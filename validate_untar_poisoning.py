import os
import pickle
import argparse

import PIL
import numpy as np
import torch

from untar_tools import *

parser = argparse.ArgumentParser(description='Pytorch Poison AT Validation')

# Load basic settings
parser.add_argument('--arch', type=str, default='resnet18',
                    help='choose the model architecture')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='choose the dataset')
parser.add_argument('--train-steps', type=int, default=15000,
                    help='set the training steps')
parser.add_argument('--batch-size', type=int, default=128,
                    help='set the batch size')

parser.add_argument('--optim', type=str, default='sgd',
                    help='select which optimizer to use')
parser.add_argument('--lr', type=float, default=0.1,
                    help='set the initial learning rate')
parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                    help='set the learning rate decay rate')
parser.add_argument('--lr-decay-freq', type=int, default=6000,
                    help='set the learning rate decay frequency')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='set the weight decay rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='set the momentum for SGD')

parser.add_argument('--pgd-radius', type=float, default=0,
                    help='set the perturbation radius in pgd')
parser.add_argument('--pgd-steps', type=int, default=0,
                    help='set the number of iteration steps in pgd')
parser.add_argument('--pgd-step-size', type=float, default=0,
                    help='set the step size in pgd')
parser.add_argument('--pgd-random-start', action='store_true',
                    help='if select, randomly choose starting points each time performing pgd')
parser.add_argument('--pgd-norm-type', type=str, default='l-infty',
                    choices=['l-infty', 'l2', 'l1'],
                    help='set the type of metric norm in pgd')

parser.add_argument('--local_rank', type=int, default=0,
                    help='for distributed data parallel')

parser.add_argument('--data-dir', type=str, default='../data',
                    help='set the path to the exp data')
parser.add_argument('--save-dir', type=str, default=None,
                    help='set which dictionary to save the experiment result')
parser.add_argument('--save-name', type=str, default='valid',
                    help='set the save name of the experiment result')

parser.add_argument('--seed', type=int, default=7,
                    help='set the random seed')

parser.add_argument('--data-mode', type=str, default='mix',
                    help='mix = clear + unlearnable data, clear = clear data only')

parser.add_argument('--noise-path', type=str, default=None,
                    help='set the path to the train images noises')
parser.add_argument('--poi-idx-path', type=str, default=None,
                    help='set the path to the poisoned indices')
parser.add_argument('--resume-path', type=str, default=None,
                    help='set where to resume the model')

parser.add_argument('--perturb-freq', type=int, default=1,
                    help='set the perturbation frequency')
parser.add_argument('--report-freq', type=int, default=200,
                    help='set the report frequency')
parser.add_argument('--save-freq', type=int, default=5000,
                    help='set the checkpoint saving frequency')

parser.add_argument('--mask-path', type=str, default=None)
parser.add_argument('--noise-rate', type=float, default=1.0)
parser.add_argument('--noise-type', type=str, default='patch')

# Device
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()

# Set device and show args-info
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.save_dir is None:
    poison_name = args.noise_path.split('/')[-3]
    valid_name = '{}_pr{}_sd{}'.format(args.save_name, args.pgd_radius, args.seed)
    args.save_dir = os.path.join('exp_data', 'untargeted', poison_name, valid_name)
logger = init_logger(args)
setup = system_startup(logger)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def save_checkpoint(save_dir, save_name, model, optim):
    ckpt_path = os.path.join(save_dir, 'checkpoints')

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    torch.save({
        'model_state_dict': get_model_state(model),
        'optim_state_dict': optim.state_dict(),
        }, os.path.join(ckpt_path, '{}-model.pkl'.format(save_name)))



''' init model / optim / dataloader / loss func '''
model = get_arch(args.arch, args.dataset)

if args.resume_path is not None:
    state_dict = torch.load(args.resume_path, map_location=torch.device('cpu'))
    model.load_state_dict( state_dict['model_state_dict'] )
    del state_dict

criterion = torch.nn.CrossEntropyLoss()

model.to(**setup)
criterion.to(**setup)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optim = get_optim(
    args.optim, model.parameters(),
    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

if args.data_mode == 'mix':
    train_loader = get_poisoned_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True,
        noise_path=args.noise_path, noise_rate=args.noise_rate,
        poisoned_indices_path=args.poi_idx_path, mask_path=args.mask_path, type=args.noise_type)
elif args.data_mode == 'clear':
    train_loader = get_clear_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True,
        poisoned_indices_path=args.poi_idx_path)
else:
    raise NotImplementedError

test_loader = get_poisoned_loader(
    args.dataset, batch_size=args.batch_size, root=args.data_dir, train=False)

attacker = PGDAttacker(
    radius = args.pgd_radius,
    steps = args.pgd_steps,
    step_size = args.pgd_step_size,
    random_start = args.pgd_random_start,
    norm_type = args.pgd_norm_type,
    ascending = True,
)

for step in range(args.train_steps):
    lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
    for group in optim.param_groups:
        group['lr'] = lr

    x, y = next(train_loader)
    x, y = x.to(**setup), y.to(device=setup['device'], dtype=torch.long, non_blocking=True)

    if (step+1) % args.perturb_freq == 0:
        adv_x = attacker.perturb(model, criterion, x, y)
    else:
        adv_x = x

    model.train()
    _y = model(adv_x)
    adv_acc = (_y.argmax(dim=1) == y).sum().item() / len(x)
    adv_loss = criterion(_y, y)
    optim.zero_grad()
    adv_loss.backward()
    optim.step()

    if (step+1) % args.save_freq == 0:
        save_checkpoint(
            args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1),
            model, optim)

    if (step+1) % args.report_freq == 0:
        test_acc, test_loss, robust_acc, robust_loss = evaluate(model, criterion, test_loader, setup, attacker)

        logger.info('step [{}/{}]:'.format(step+1, args.train_steps))
        logger.info('adv_acc {:.2%} \t adv_loss {:.3e}'
                    .format( adv_acc, adv_loss.item() ))
        logger.info('test_acc  {:.2%} \t test_loss  {:.3e}'
                    .format( test_acc, test_loss ))
        logger.info('test_robust_acc  {:.2%} \t test_robust_loss  {:.3e}'
                    .format( robust_acc, robust_loss ))
        logger.info('')

save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name), model, optim)
