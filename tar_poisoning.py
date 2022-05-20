import argparse
import time

from tar_tools import *
from tar_tools import _pgd_step

parser = argparse.ArgumentParser(description='Pytorch Poisoning Adversarial Training')

# Central
parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
parser.add_argument('--dataset', default='CIFAR10', type=str)

# Random seed
parser.add_argument('--poisonkey', default=None, type=str)
parser.add_argument('--modelkey', default=None, type=int)

# Generation
parser.add_argument('--attackoptim', default='PGD', type=str)
parser.add_argument('--attackiter', default=250, type=int)
parser.add_argument('--init', default='randn', type=str)
parser.add_argument('--pbatch', default=512, type=int)
parser.add_argument('--budget', default=0.04, type=float, help='Fraction of training data that is poisoned')
parser.add_argument('--eps', default=16, type=float)
parser.add_argument('--tau', default=0.01, type=float)
parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')
parser.add_argument('--loss', default='similarity', type=str)
parser.add_argument('--lamb', default=0.01, type=float)

# Training
parser.add_argument('--strategy', default='adversarial', type=str)
parser.add_argument('--pretrained', default=None, type=str)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--optimizer', default='SGD', type=str)
parser.add_argument('--scheduler', default='linear', type=str)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--augmentations', default='default', type=str)
parser.add_argument('--perturb-steps', default=10, type=int)
parser.add_argument('--at-eps', default=2, type=int)
parser.add_argument('--step-size', default=0.5, type=float)
parser.add_argument('--clean-threat', default=False, type=bool)

# Validation
parser.add_argument('--vruns', default=1, type=int)
parser.add_argument('--vgg-vruns', default=1, type=int)

# Data path & Export
parser.add_argument('--save', default=None, type=str)
parser.add_argument('--exp-path', default=None, type=str)
parser.add_argument('--poison-path', default=None, type=str)
parser.add_argument('--ckpt-path', default=None, type=str)
parser.add_argument('--data-path', default='../data', type=str)

# Device
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()

# Set device and show args-info
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.exp_path is None:
    poison_name = '{}_{}_eps{}_tau{}_poisonkey{}'.format(args.dataset, args.attackoptim, args.eps,
                                                         args.tau, args.poisonkey)
    args.exp_path = os.path.join('exp_data', 'targeted', poison_name)
logger = init_logger(args)
setup = system_startup(logger)

# Init Threat Model / Training settings
model = Victim(args, setup, logger, is_victim=False)

# Init Trainset / Validset / Poisonset / Target
data = DataLoader(args, setup, logger)

start_time = time.time()

# If Threat Model does not load pretrained parameters, then train Threat Model from scratch
if args.pretrained is None:
    model.train(data)
else:
    logger.info(f'Model loaded state from {args.pretrained} successfully ...')
train_time = time.time()


""" generate poisons """
model.eval(dropout=True)

# Init target img and label
target = torch.stack([tars[0] for tars in data.targetset], dim=0).to(**setup)
intended_classes = torch.tensor(data.poison_setup['intended_class']).to(device=setup['device'], dtype=torch.long)
true_classes = torch.tensor([tars[1] for tars in data.targetset]).to(device=setup['device'], dtype=torch.long)

# Precompute target gradients
target_grad, target_gnorm = model.gradient(target, intended_classes)
logger.info(f'Target Grad Norm is {target_gnorm}')

# Restart poisoning
poisons, scores = [], torch.ones(args.restarts) * 10000
for trial in range(args.restarts):
    # Init poison delta
    poison_delta = data.initialize_poison(initializer='zero')

    # Update loop
    for iter in range(args.attackiter):
        target_losses = 0
        matching_losses = 0
        cent_losses = 0
        poison_correct = 0
        for batch, (inputs, labels, ids) in enumerate(data.poisonloader):
            # Take a batch step toward minimizing the current target loss.
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=True)
            batch_slice = []
            for image_id in ids.tolist():
                lookup = data.poison_lookup.get(image_id)
                if lookup is not None:
                    batch_slice.append(lookup)

            poison_images = inputs.clone()
            inputs += poison_delta[batch_slice].detach().to(**setup)
            inputs_ = inputs.clone().to(**setup)
            inputs_.requires_grad_()

            # Compute loss
            matching_loss, prediction = similarity_loss(args.loss, model.model, model.criterion, inputs_, labels,
                                                        target_grad, target_gnorm)
            cent_loss = torch.nn.functional.cross_entropy(model.model(inputs_), labels)
            loss = matching_loss - args.lamb * cent_loss
            loss.backward()

            matching_losses += matching_loss.item()
            cent_losses += cent_loss.item()

            target_losses += loss.item()
            poison_correct += prediction.item()

            # Update Step
            if args.dataset == 'TinyImageNet': imgnet = True
            else: imgnet = False
            inputs_.data = torch.clamp(denormalize(inputs_.data, imgnet=imgnet), min=0).to(**setup)
            poison_images = torch.clamp(denormalize(poison_images, imgnet=imgnet), min=0).to(**setup)
            X_pgd = _pgd_step(args, inputs_.data, inputs_.grad, poison_images, args.tau)
            delta_slice = normalize(X_pgd, imgnet=imgnet) - normalize(poison_images, imgnet=imgnet)
            poison_delta[batch_slice] = delta_slice.detach().to(device=torch.device('cpu'))

        target_losses = target_losses / (batch + 1)
        matching_losses = matching_losses / (batch + 1)
        cent_losses = cent_losses / (batch + 1)
        poison_acc = poison_correct / len(data.poisonloader.dataset)
        if iter % (args.attackiter // 25) == 0 or iter == (args.attackiter - 1):
            logger.info(f'Iteration {iter}: Target loss is {target_losses:2.4f}, '
                        f'Poison clean acc is {poison_acc * 100:2.2f}%, '
                        f'Gradmatching loss is {matching_losses:2.4f}, '
                        f'Crossentropy loss is {cent_losses:2.4f}')

    # Record scores and poison delta
    scores[trial] = target_losses
    poisons.append(poison_delta.detach())

# Get optimal poison delta
optimal_score = torch.argmin(scores)
stat_optimal_loss = scores[optimal_score].item()
logger.info(f'Poisons with minimal target loss {stat_optimal_loss:6.4e} selected.')
poison_delta = poisons[optimal_score]
poison_time = time.time()

# Export Dataset
if args.save is not None:
    data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

# Validate the model
if args.vruns > 0:
    model.validate(data, poison_delta)
if args.vgg_vruns > 0:
    if args.dataset == 'TinyImageNet':
        args.net = ['VGG11-TI']
    else:
        args.net = ['VGG11']
    args.vruns = args.vgg_vruns
    model = Victim(args, setup, logger, is_victim=True)
    model.validate(data, poison_delta)
test_time = time.time()


logger.info('---------------------------------------------------')
logger.info(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
logger.info(f'---------------------- poisoning time: {str(datetime.timedelta(seconds=poison_time - train_time))}')
logger.info(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - poison_time))}')
logger.info('-------------Job finished.-------------------------')
