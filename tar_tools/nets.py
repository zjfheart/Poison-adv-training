"""Model definitions."""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

import numpy as np


def get_model(model_name, dataset_name, pretrained=None, is_victim=False):
    """Retrieve an appropriate architecture."""
    if 'CIFAR' in dataset_name:
        # if pretrained:
        #     raise ValueError('Loading pretrained models is only supported for ImageNet.')
        in_channels = 3
        num_classes = 10 if dataset_name in ['CIFAR10'] else 100
        if 'ResNet' in model_name:
            model = resnet_picker(model_name, dataset_name)
        elif 'VGG' in model_name:
            model = VGG(model_name)
        elif model_name == 'MobileNetV2':
            model = MobileNetV2(num_classes=num_classes, train_dp=0, test_dp=0, droplayer=0, bdp=0)
        else:
            raise ValueError(f'Architecture {model_name} not implemented for dataset {dataset_name}.')

        if pretrained is not None and is_victim is False:
            state = torch.load(pretrained, map_location="cuda:0")['state_dict']
            for k in list(state.keys()):
                if 'module' in k:
                    state[k[7:]] = state.pop(k)
            model.load_state_dict(state)

    elif 'TinyImageNet' in dataset_name:
        in_channels = 3
        num_classes = 200

        if 'ResNet' in model_name:
            model = resnet_picker(model_name, dataset_name)
        elif 'VGG' in model_name:
            model = VGG(model_name, in_channels=in_channels, num_classes=num_classes)
        elif model_name == 'MobileNetV2':
            model = MobileNetV2(num_classes=num_classes)
        else:
            raise ValueError(f'Model {model_name} not implemented for TinyImageNet')

        if pretrained is not None and is_victim is False:
            state = torch.load(pretrained, map_location="cuda:0")['state_dict']
            for k in list(state.keys()):
                if 'module' in k:
                    state[k[7:]] = state.pop(k)
            model.load_state_dict(state)


    return model


def resnet_picker(arch, dataset):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""
    in_channels = 1 if dataset == 'MNIST' else 3
    num_classes = 10
    if dataset in ['CIFAR10', 'MNIST', 'SVHN']:
        num_classes = 10
        initial_conv = [3, 1, 1]
    elif dataset == 'CIFAR100':
        num_classes = 100
        initial_conv = [3, 1, 1]
    elif dataset == 'TinyImageNet':
        num_classes = 200
        initial_conv = [7, 2, 3]
    else:
        raise ValueError(f'Unknown dataset {dataset} for ResNet.')

    if arch == 'ResNet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif 'ResNet20-' in arch and arch[-1].isdigit():
        width_factor = int(arch[-1])
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * width_factor, initial_conv=initial_conv)
    elif arch == 'ResNet28-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet32':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet32-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10, initial_conv=initial_conv)
    elif arch == 'ResNet44':
        return ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet56':
        return ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet110':
        return ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16, initial_conv=initial_conv)
    elif arch == 'ResNet18':
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif 'ResNet18-' in arch:  # this breaks the usual notation, but is nicer for now!!
        new_width = int(arch.split('-')[1])
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=new_width, initial_conv=initial_conv)
    elif arch == 'ResNet34':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet50':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet101':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    elif arch == 'ResNet152':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64, initial_conv=initial_conv)
    else:
        raise ValueError(f'Invalid ResNet [{dataset}] model chosen: {arch}.')


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR-like thingies.

    This is a minor modification of
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
    adding additional options.
    """

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
                 norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
        self._norm_layer = norm_layer

        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=initial_conv[0],
                                     stride=initial_conv[1], padding=initial_conv[2], bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11-TI': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16-TI': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels=3, num_classes=10):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        in_planes,
        out_planes,
        expansion,
        stride,
        train_dp,
        test_dp,
        droplayer=0,
        bdp=0,
    ):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

        self.train_dp = train_dp
        self.test_dp = test_dp

        self.droplayer = droplayer
        self.bdp = bdp

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if self.stride == 1 and action == 1:
            # if stride is not 1, then there is no skip connection. so we keep this layer unchanged
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            if self.test_dp > 0 or (self.training and self.train_dp > 0):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)

            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = (
                    torch.bernoulli(
                        self.bdp
                        * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(
                            out.device
                        )
                    )
                    / self.bdp
                )
                out = bdp_mask * out

            out = self.bn3(self.conv3(out))
            out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layers = self._make_layers(
            in_planes=32,
            train_dp=train_dp,
            test_dp=test_dp,
            droplayer=droplayer,
            bdp=bdp,
        )
        self.conv2 = nn.Conv2d(
            320, 1280, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.test_dp = test_dp
        self.bdp = bdp

    def _make_layers(self, in_planes, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        layers = []

        # get the total number of blocks
        nblks = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            nblks += num_blocks

        dl_step = droplayer / nblks

        blkidx = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                dl = dl_step * blkidx
                blkidx += 1

                layers.append(
                    Block(
                        in_planes,
                        out_planes,
                        expansion,
                        stride,
                        train_dp=train_dp,
                        test_dp=test_dp,
                        droplayer=dl,
                        bdp=bdp,
                    )
                )
                in_planes = out_planes
        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.layers:
            layer.test_dp = dp

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        if out.shape[-1] == 2:
            out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, penu=False):
        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if "linear" in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()
