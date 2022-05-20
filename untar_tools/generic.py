import pickle
import os
import socket
import sys
import logging
import math

import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from . import data
from . import imagenet_utils


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def get_transforms(dataset, train=True, is_tensor=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_transforms(dataset, train, is_tensor)

    if train:
        if dataset == 'cifar10' or dataset == 'cifar100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), ]
        else:
            raise NotImplementedError
    else:
        comp1 = []

    if is_tensor:
        comp2 = [
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = data.ElementWiseTransform(trans)

    return trans


def get_dataset(dataset, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_dataset(dataset, root, train)

    transform = get_transforms(dataset, train=train, is_tensor=False)

    if dataset == 'cifar10':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    return data.Dataset(x, y, transform)


def get_indexed_loader(dataset, batch_size, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_indexed_loader(dataset, batch_size, root, train)

    target_set = get_dataset(dataset, root=root, train=train)

    if train:
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
    else:
        target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_indexed_tensor_loader(dataset, batch_size, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_indexed_tensor_loader(dataset, batch_size, root, train)

    target_set = get_dataset(dataset, root=root, train=train)
    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_poisoned_loader(
        dataset, batch_size, root='./data', train=True,
        noise_path=None, noise_rate=1.0, poisoned_indices_path=None, mask_path=None, type=None):

    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_poisoned_loader(
                dataset, batch_size, root, train, noise_path, noise_rate, poisoned_indices_path, mask_path, type)

    target_set = get_dataset(dataset, root=root, train=train)

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)
        assert isinstance(raw_noise, np.ndarray)
        # assert raw_noise.dtype == np.uint8

        raw_noise = raw_noise.astype(np.int16)

        if type == 'patch':
            with open(mask_path, 'rb') as f:
                raw_mask = pickle.load(f)
            raw_mask = raw_mask.astype(np.int16)
            mask = raw_mask
            mask = np.transpose(mask, [1, 2, 0])


        noise = np.zeros_like(raw_noise)

        if poisoned_indices_path is not None:
            with open(poisoned_indices_path, 'rb') as f:
                indices = pickle.load(f)
        else:
            indices = np.random.permutation(len(noise))[:int(len(noise)*noise_rate)]

        noise[indices] += raw_noise[indices]

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        if type == 'patch':
            target_set.x[indices] = target_set.x[indices].astype(np.int16) * (1 - mask) + noise[indices] * mask
            target_set.x[indices] = target_set.x[indices].clip(0, 255).astype(np.uint8)
        elif type == 'rem':
            imgs = target_set.x.astype(np.int16) + noise
            imgs = imgs.clip(0, 255).astype(np.uint8)
            target_set.x = imgs


    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_clear_loader(
        dataset, batch_size, root='./data', train=True,
        noise_rate=0, poisoned_indices_path=None):

    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_clear_loader(
                dataset, batch_size, root, train, noise_rate, poisoned_indices_path)

    target_set = get_dataset(dataset, root=root, train=train)
    data_nums = len(target_set)

    if poisoned_indices_path is not None:
        with open(poisoned_indices_path, 'rb') as f:
            poi_indices = pickle.load(f)
        indices = np.array( list( set(range(data_nums)) - set(poi_indices) ) )

    else:
        indices = np.random.permutation(range(data_nums))[: int( data_nums * (1-noise_rate) )]

    ''' select clear examples '''
    target_set.x = target_set.x[indices]
    target_set.y = np.array(target_set.y)[indices]

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_optim(optim, params, lr=0.1, weight_decay=1e-4, momentum=0.9):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    raise NotImplementedError('optimizer {} is not supported'.format(optim))


def init_logger(args):

    path = args.save_dir
    if not os.path.exists(path):
        os.makedirs(path)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        os.path.join(path, 'log.txt'), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    return logger

def system_startup(logger=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=True)

    logger.info(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        logger.info(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup


def evaluate(model, criterion, loader, setup, attacker=None):
    acc = AverageMeter()
    loss = AverageMeter()
    if attacker is not None:
        robust_acc = AverageMeter()
        robust_loss = AverageMeter()

    model.eval()
    for x, y in loader:
        x, y = x.to(**setup), y.to(device=setup['device'], dtype=torch.long, non_blocking=True)
        if attacker is not None:
            x_adv = attacker.perturb(model, criterion, x, y)
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
            acc.update(ac, len(x))
            loss.update(lo, len(x))

            if attacker is not None:
                _y_adv = model(x_adv)
                adv_acc = (_y_adv.argmax(dim=1) == y).sum().item() / len(x)
                adv_loss = criterion(_y_adv, y)
                robust_acc.update(adv_acc, len(x))
                robust_loss.update(adv_loss, len(x))

    if attacker is None:
        return acc.average(), loss.average()
    else:
        return acc.average(), loss.average(), robust_acc.average(), robust_loss.average()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_model_state(model):
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    return model_state

def init_patch_square(image_size, patch_size, patch_nums):
    patches = []
    img_size = image_size
    image_size = img_size ** 2
    noise_size = image_size * patch_size
    noise_dim = int(noise_size ** (0.5))

    for _ in range(patch_nums):
        patch = np.random.uniform(0, 1, [1, 3, noise_dim, noise_dim])
        patches.append(patch)
    patches = np.concatenate(patches, axis=0)
    return patches, patches.shape


def square_transform(patch, data_shape, patch_shape, image_size, fixed_location=None):
    # get dummy image
    x = np.zeros(data_shape)

    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):

        # random location
        if fixed_location is None:
            random_x = np.random.choice(image_size)
            random_y = np.random.choice(image_size)
        else:
            random_x = fixed_location
            random_y = fixed_location
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)

        # apply patch to dummy image
        x[i][0][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x + patch_shape[-1], random_y:random_y + patch_shape[-1]] = patch[i][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0

    return x.astype(np.uint8), mask[0]

def get_arch(arch, dataset):
    if dataset == 'cifar10':
        in_dims, out_dims = 3, 10
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
    elif dataset == 'imagenet-mini':
        in_dims, out_dims = 3, 100
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    if arch == 'resnet18':
        return resnet18(in_dims, out_dims)

    elif arch == 'resnet50':
        return resnet50(in_dims, out_dims)

    elif arch == 'vgg11-bn':
        if dataset == 'imagenet' or dataset == 'imagenet-mini':
            raise NotImplementedError
        return vgg11_bn(in_dims, out_dims)

    elif arch == 'vgg16-bn':
        if dataset == 'imagenet' or dataset == 'imagenet-mini':
            raise NotImplementedError
        return vgg16_bn(in_dims, out_dims)

    else:
        raise NotImplementedError('architecture {} is not supported'.format(arch))


''' ref:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(BasicBlock, self).__init__()
        planes = planes * wide
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(Bottleneck, self).__init__()
        mid_planes = planes * wide
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_dims, out_dims, wide=1):
        super(ResNet, self).__init__()
        self.wide = wide
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_dims, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, out_dims)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wide))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def feature_extract(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


class WRN(nn.Module):
    def __init__(self, num_blocks, in_dims, out_dims, wide=10):
        super(WRN, self).__init__()
        self.in_planes = 16
        self.wide = wide

        block = BasicBlock

        self.conv1 = nn.Conv2d(in_dims, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64*wide, out_dims)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wide))
            self.in_planes = planes * self.wide * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, 8)
        # out = out.view(out.shape[0], -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def resnet18(in_dims, out_dims):
    return ResNet(BasicBlock, [2,2,2,2], in_dims, out_dims, 1)

def resnet50(in_dims, out_dims):
    return ResNet(Bottleneck, [3,4,6,3], in_dims, out_dims, 1)

'''
Modified from https://github.com/pytorch/vision.git
'''

__all__ = [
    'VGG', 'vgg11_bn', 'vgg16_bn'
]

class VGG(nn.Module):
    ''' VGG model '''
    def __init__(self, features, out_channels):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, out_channels),
        )

        ''' Initialize weights '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, in_dims=3, batch_norm=False):
    layers = []
    in_channels = in_dims
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11_bn(in_dims=3, out_dims=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], in_dims, batch_norm=True), out_dims)

def vgg16_bn(in_dims=3, out_dims=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], in_dims, batch_norm=True), out_dims)

