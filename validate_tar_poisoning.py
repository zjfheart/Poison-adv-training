import os
import argparse
import time
import datetime

import torch
import torchvision
from torchvision import transforms

from tar_tools import *

parser = argparse.ArgumentParser(description='Pytorch Poison AT Validation')

# Central
parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
parser.add_argument('--dataset', default='CIFAR10', type=str)

# Random seed
parser.add_argument('--poisonkey', default=None, type=str)
parser.add_argument('--modelkey', default=None, type=str)

parser.add_argument('--budget', default=0.04, type=float, help='Fraction of training data that is poisoned')
parser.add_argument('--pbatch', default=512, type=int)

# Training setting
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
parser.add_argument('--at_eps', default=2, type=int)
parser.add_argument('--step-size', default=0.5, type=float)
parser.add_argument('--clean-threat', default=False, type=bool)

# Validate setting
parser.add_argument('--vruns', default=4, type=int)

# Data path
parser.add_argument('--poison-path', default=None, type=str)
parser.add_argument('--data-path', default='../data', type=str)
parser.add_argument('--exp-path', default=None, type=str)
parser.add_argument('--ckpt-path', default=None, type=str)

# Device
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()

# Set device and show args-info
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.exp_path is None:
    name = '{}_{}_eps{}'.format(args.dataset, args.net[0], args.at_eps)
    args.exp_path = os.path.join('exp_data', 'targeted', 'valid', name)
    if args.ckpt_path is None: args.ckpt_path = os.path.join(args.exp_path, 'checkpoints')
logger = init_logger(args)
setup = system_startup(logger)

# Init Victim Model
model = Victim(args, setup, logger, is_victim=True)

# Load Poisoned Dataset
data = DataLoader(args, setup, logger)

if args.dataset == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
elif args.dataset == 'TinyImageNet':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tiny_imagenet_mean, tiny_imagenet_std)
    ])

trainset = ImageFolder(root=os.path.join(args.poison_path, 'train'), transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)

data.trainloader = train_loader

# Validate Poison
start_time = time.time()
stats = model.validate(data)
test_time = time.time()


print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print('---------------------------------------------------')
print(f'Finished computations with  test time: {str(datetime.timedelta(seconds=test_time - start_time))}')
print('-------------Job finished.-------------------------')



