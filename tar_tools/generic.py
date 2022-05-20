"""Various utilities."""
import os
import socket
import datetime
import logging
import sys

import PIL
import torch
import random
import numpy as np
from PIL import Image
from resizeimage import resizeimage

from .consts import NON_BLOCKING, cifar10_mean, cifar10_std, tiny_imagenet_mean, tiny_imagenet_std

def init_logger(args):
    path = args.exp_path
    if not os.path.exists(path):
        os.makedirs(path)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(os.path.join(path, 'log.txt'), mode='w')
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
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)

    logger.info(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        logger.info(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup

def print_and_save_stats(epoch, logger, current_lr, train_loss, train_acc, valid_acc, valid_loss, adv_acc, adv_loss,
                         target_acc, target_ori_acc, adv_target_acc, adv_target_ori_acc):

    if valid_acc is not None:
        logger.info(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
                    f'Training loss is {train_loss:7.4f}, train acc: {train_acc:7.2%} | ')

        logger.info(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
                    f'Validation loss is {valid_loss:7.4f}, valid acc: {valid_acc:7.2%} | '
                    f'Target fool acc: {target_acc:7.2%}, orig. acc: {target_ori_acc:7.2%} | ')

        if adv_acc is not None:
            logger.info(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
                        f'Robust validation loss is {adv_loss:7.4f}, robust valid acc: {adv_acc:7.2%} | '
                        f'Adv target fool acc: {adv_target_acc:7.2%}, adv orig. acc: {adv_target_ori_acc:7.2%} | ')

    else:
        logger.info(f'Epoch: {epoch:<3}| lr: {current_lr:.4f} | '
                    f'Training loss is {train_loss:7.4f}, train acc: {train_acc:7.2%} | ')

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def denormalize(image, device='cuda', batch=True, imgnet=False):
    if not imgnet:
        m, s = cifar10_mean, cifar10_std
    else:
        m, s = tiny_imagenet_mean, tiny_imagenet_std
    if device == 'cpu':
        mean = torch.tensor(m)[:, None, None]
        std = torch.tensor(s)[:, None, None]
    else:
        mean = torch.tensor(m)[:, None, None].cuda()
        std = torch.tensor(s)[:, None, None].cuda()
    if batch:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return image * std + mean

def normalize(image, device='cuda', batch=True, imgnet=False):
    if not imgnet:
        m, s = cifar10_mean, cifar10_std
    else:
        m, s = tiny_imagenet_mean, tiny_imagenet_std
    if device == 'cpu':
        mean = torch.tensor(m)[:, None, None]
        std = torch.tensor(s)[:, None, None]
    else:
        mean = torch.tensor(m)[:, None, None].cuda()
        std = torch.tensor(s)[:, None, None].cuda()
    if batch:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (image - mean) / std

def vis_cifar10_image(torch_tensor, normalized, show=False, save=None, imgnet=False):
    torch_image = torch_tensor.clone().cpu()
    if len(torch_image.size()) != 3:
        if torch_image.size(0) == 1:
            torch_image = torch.squeeze(torch_image, dim=0)
        else:
            raise ValueError("not a batch...")
    if normalized:
        torch_image = torch.clamp(denormalize(torch_image, device='cpu', batch=False, imgnet=imgnet), 0, 1)
    torch_image_uint8 = torch_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
    image_PIL = PIL.Image.fromarray(torch_image_uint8.numpy())
    if show:
        image_PIL.show()
    if save is not None:
        filename = os.path.join("./vis", save + '.png')
        image_PIL.save(filename)
