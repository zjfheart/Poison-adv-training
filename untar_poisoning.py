import os
import pickle
import argparse

import numpy as np
import torch
import torchvision.transforms as transforms

from untar_tools import *


parser = argparse.ArgumentParser(description='Pytorch Poisoning Adversarial Training')

# Load basic settings
parser.add_argument('--arch', type=str, default='resnet18',
                    help='choose the model architecture')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='choose the dataset')
parser.add_argument('--train-steps', type=int, default=5000,
                    help='set the training steps')
parser.add_argument('--batch-size', type=int, default=128,
                    help='set the batch size')

parser.add_argument('--optim', type=str, default='sgd',
                    help='select which optimizer to use')
parser.add_argument('--lr', type=float, default=0.1,
                    help='set the initial learning rate')
parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                    help='set the learning rate decay rate')
parser.add_argument('--lr-decay-freq', type=int, default=2000,
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
parser.add_argument('--save-name', type=str, default='patch',
                    help='set the save name of the experiment result')

parser.add_argument('--seed', type=int, default=7,
                    help='set the random seed')

# Frequencies
parser.add_argument('--perturb-freq', type=int, default=1,
                    help='set the perturbation frequency')
parser.add_argument('--report-freq', type=int, default=1000,
                    help='set the report frequency')
parser.add_argument('--save-freq', type=int, default=1000,
                    help='set the checkpoint saving frequency')

# Privacy sticker
parser.add_argument('--patch-size', type=float, default=0.1,
                    help='set the size of patch ')
parser.add_argument('--location', type=int, default=None,
                    help='set the patch location')

# Device
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()

# Set device and show args-info
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.save_dir is None:
    poison_name = '{}_{}_loc{}_ss{}_ps{}'.format(args.save_name, args.dataset, args.location, args.pgd_step_size,
                                                 args.patch_size)
    args.save_dir = os.path.join('exp_data', 'untargeted', poison_name)
logger = init_logger(args)
setup = system_startup(logger)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def regenerate_def_noise(def_noise, model, criterion, loader, defender, setup, mask):
    for x, y, ii in loader:
        x, y = x.to(**setup), y.to(device=setup['device'], dtype=torch.long, non_blocking=True)
        patch = defender.perturb(model, criterion, x, y, mask)
        def_noise[ii] = (patch.cpu().numpy() * 255).round().astype(np.uint8)


def save_checkpoint(save_dir, save_name, model, optim, def_patch=None, mask=None):
    ckpt_path = os.path.join(save_dir, 'checkpoints')
    poison_path = os.path.join(save_dir, 'poisons')

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    torch.save({
        'model_state_dict': get_model_state(model),
        'optim_state_dict': optim.state_dict(),
        }, os.path.join(ckpt_path, '{}-model.pkl'.format(save_name)))

    if def_patch is not None:
        if not os.path.exists(poison_path):
            os.makedirs(poison_path)
        def_noise = def_patch.astype(np.uint8)
        mask = mask.astype(np.uint8)
        assert (def_noise.max() <= 255 and def_noise.min() >= 0)
        with open(os.path.join(poison_path, '{}-def-noise.pkl'.format(save_name)), 'wb') as f:
            pickle.dump(def_noise, f)
        with open(os.path.join(poison_path, '{}-def-mask.pkl'.format(save_name)), 'wb') as f:
            pickle.dump(mask, f)



''' init model / optim / dataloader / loss func '''
model = get_arch(args.arch, args.dataset)
optim = get_optim(
    args.optim, model.parameters(),
    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

train_loader = get_indexed_tensor_loader(
    args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True)

''' get train transforms '''
train_trans = transforms.Compose([
    transforms.Normalize((255 * 0.5, 255 * 0.5, 255 * 0.5), (255., 255., 255.))
])

test_loader = get_indexed_loader(
    args.dataset, batch_size=args.batch_size, root=args.data_dir, train=False)
criterion = torch.nn.CrossEntropyLoss()

defender = PatchPGDAttacker(
    steps = args.pgd_steps,
    step_size = args.pgd_step_size,
    random_start = args.pgd_random_start,
    norm_type = args.pgd_norm_type,
    ascending = False,
    trans = train_trans
)

model.to(**setup)
criterion.to(**setup)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

''' initialize the defensive noise (for unlearnable examples) '''
data_nums = len( train_loader.loader.dataset )
if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    def_patch, patch_shape = init_patch_square(32, args.patch_size, data_nums)
    data_shape = (data_nums, 3, 32, 32)
    def_patch, mask = square_transform(def_patch, data_shape, patch_shape, 32, args.location)
elif args.dataset == 'imagenet-mini':
    def_patch, patch_shape = init_patch_square(256, args.patch_size, data_nums)
    data_shape = (data_nums, 3, 256, 256)
    def_patch, mask = square_transform(def_patch, data_shape, patch_shape, 256, args.location)
else:
    raise NotImplementedError

for step in range(args.train_steps):
    lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
    for group in optim.param_groups:
        group['lr'] = lr

    x, y, ii = next(train_loader)
    x, y = x.to(**setup), y.to(device=setup['device'], dtype=torch.long, non_blocking=True)

    if (step+1) % args.perturb_freq == 0:
        patch = defender.perturb(model, criterion, x, y, mask)
        def_patch[ii] = (patch.cpu().numpy() * 255).round().astype(np.uint8)

    patch_mask = torch.FloatTensor(mask).to(**setup)
    def_x = train_trans(x * (1 - patch_mask) + torch.FloatTensor(def_patch[ii]).to(**setup) * patch_mask)
    def_x.clamp_(-0.5, 0.5)

    model.train()
    _y = model(def_x)
    def_acc = (_y.argmax(dim=1) == y).sum().item() / len(x)
    def_loss = criterion(_y, y)
    optim.zero_grad()
    def_loss.backward()
    optim.step()


    if (step+1) % args.save_freq == 0:
        save_checkpoint(
            args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1),
            model, optim)

    if (step+1) % args.report_freq == 0:
        test_acc, test_loss = evaluate(model, criterion, test_loader, setup)

        logger.info('step [{}/{}]:'.format(step+1, args.train_steps))
        logger.info('def_acc {:.2%} \t def_loss {:.3e}'
                    .format( def_acc, def_loss.item() ))
        logger.info('test_acc  {:.2%} \t test_loss  {:.3e}'
                    .format( test_acc, test_loss ))
        logger.info('')

logger.info('Noise generation started')
regenerate_def_noise(
    def_patch, model, criterion, train_loader, defender, setup, mask)
logger.info('Noise generation finished')

save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name), model, optim, def_patch, mask)

