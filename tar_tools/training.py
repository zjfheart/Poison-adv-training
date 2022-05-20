"""Repeatable code parts concerning optimization and training schedules."""
import os

import torch

import datetime

from .generic import print_and_save_stats
from .poison_optim import pgd

from .consts import NON_BLOCKING, BENCHMARK, DEBUG_TRAINING
torch.backends.cudnn.benchmark = BENCHMARK


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)
    return optimizer, scheduler


def run_step(dataloader, poison_delta, loss_fn, epoch, logger, model, defs, criterion, optimizer, scheduler, net_seed=None):

    epoch_loss, total_preds, correct_preds = 0, 0, 0

    loader = dataloader.trainloader

    for batch, (inputs, labels, ids) in enumerate(loader):
        # Transfer to GPU
        inputs = inputs.to(**dataloader.setup)
        labels = labels.to(dtype=torch.long, device=dataloader.setup['device'], non_blocking=NON_BLOCKING)

        # Add adversarial pattern
        if poison_delta is not None:
            poison_slices, batch_positions = [], []
            for batch_id, image_id in enumerate(ids.tolist()):
                lookup = dataloader.poison_lookup.get(image_id)
                if lookup is not None:
                    poison_slices.append(lookup)
                    batch_positions.append(batch_id)
            # Python 3.8:
            # twins = [(b, l) for b, i in enumerate(ids.tolist()) if l:= kettle.poison_lookup.get(i)]
            # poison_slices, batch_positions = zip(*twins)

            if batch_positions:
                inputs[batch_positions] += poison_delta[poison_slices].to(**dataloader.setup)

        # Add data augmentation
        if defs.augmentations:  # defs.augmentations is actually a string, but it is False if --noaugment
            inputs = dataloader.augment(inputs)

        if not (dataloader.args.clean_threat and poison_delta is None):
            if defs.adversarial_steps:
                inputs = pgd(inputs, labels, model, loss_fn, dataloader.dm, dataloader.ds, eps=dataloader.args.at_eps,
                             steps=defs.adversarial_steps, alpha=dataloader.args.step_size, rand_init=True)

        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        # Get loss
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs.data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

        if defs.scheduler == 'cyclic':
            scheduler.step()

    if defs.scheduler == 'linear':
        scheduler.step()

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(model, criterion, dataloader.validloader, dataloader.setup)
        target_acc, target_ori_acc, adv_target_acc, adv_target_ori_acc = check_targets(model, criterion, dataloader)
        adv_acc, adv_loss = run_adv_validation(model, criterion, dataloader, loss_fn)
        if epoch == (defs.epochs - 1):
            if dataloader.args.ckpt_path is None:
                poison_name = '{}_{}_eps{}_tau{}_poisonkey{}'.format(dataloader.args.dataset, dataloader.args.attackoptim,
                                                                     dataloader.args.eps, dataloader.args.tau, dataloader.args.poisonkey)
                path = os.path.join('exp_data', 'targeted', poison_name, 'checkpoints')
            else:
                path = dataloader.args.ckpt_path
            if not os.path.exists(path):
                os.makedirs(path)
            if poison_delta is None:
                filename = 'threat-' + str(net_seed) + '.pth.tar'
            else:
                filename = 'victim-' + str(net_seed) + '.pth.tar'
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
            }, path=path, filename=filename)
    else:
        valid_acc, valid_loss = None, None
        adv_acc, adv_loss = None, None
        target_acc, target_ori_acc, adv_target_acc, adv_target_ori_acc = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, logger, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         valid_acc, valid_loss, adv_acc, adv_loss,
                         target_acc, target_ori_acc, adv_target_acc, adv_target_ori_acc)


def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if dryrun:
                break

    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg

def check_targets(model, criterion, dataloader):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    if len(dataloader.targetset) > 0:

        target_images = torch.stack([data[0] for data in dataloader.targetset]).to(**dataloader.setup)
        intended_labels = torch.tensor(dataloader.poison_setup['intended_class']).to(device=dataloader.setup['device'], dtype=torch.long)
        original_labels = torch.stack([torch.as_tensor(data[1], device=dataloader.setup['device'], dtype=torch.long) for data in dataloader.targetset])
        with torch.no_grad():
            outputs = model(target_images)
            # print(f"Target Logit is {outputs[0].tolist()}")
            # print(f"{outputs[0].tolist()[dataloader.poison_setup['target_class']]}  "
            #       f"{outputs[0].tolist()[dataloader.poison_setup['poison_class']]}  "
            #       f"{dataloader.trainset.classes[dataloader.poison_setup['poison_class']]}")
            predictions = torch.argmax(outputs, dim=1)

            loss_intended = criterion(outputs, intended_labels)
            accuracy_intended = (predictions == intended_labels).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            accuracy_clean = (predictions == original_labels).sum().float() / predictions.size(0)
            # print(f'Raw softmax output is {torch.softmax(outputs, dim=1)}, intended: {intended_class}')

        target_images_adv = pgd(target_images, original_labels, model, criterion, dataloader.dm, dataloader.ds,
                                      eps=dataloader.args.at_eps, steps=dataloader.args.perturb_steps, alpha=dataloader.args.step_size, rand_init=True)
        with torch.no_grad():
            outputs = model(target_images_adv)
            # print(f"Adversarial Target Logit is {outputs[0].tolist()[dataloader.poison_setup['target_class']]}  "
            #       f"{outputs[0].tolist()[dataloader.poison_setup['poison_class']]}")
            # print(f"Adversarial Target Logit is {outputs[0].tolist()}")
            # print(f"{outputs[0].tolist()[dataloader.poison_setup['target_class']]}  "
            #       f"{outputs[0].tolist()[dataloader.poison_setup['poison_class']]}")
            predictions = torch.argmax(outputs, dim=1)
            accuracy_intended_adv = (predictions == intended_labels).sum().float() / predictions.size(0)
            accuracy_clean_adv = (predictions == original_labels).sum().float() / predictions.size(0)
            # print(f'Adv fool acc: {accuracy_intended_adv:7.2%}, orig  acc: {accuracy_clean_adv:7.2%}')


        return accuracy_intended.item(), accuracy_clean.item(), accuracy_intended_adv.item(), accuracy_clean_adv.item()
    else:
        return 0, 0, 0, 0

def run_adv_validation(model, criterion, dataloader, loss_fn):
    """Get adversarial accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for i, (inputs, targets, _) in enumerate(dataloader.validloader):
        inputs = inputs.to(**dataloader.setup)
        targets = targets.to(device=dataloader.setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
        inputs = pgd(inputs, targets, model, loss_fn, dataloader.dm, dataloader.ds, eps=dataloader.args.at_eps,
                     steps=dataloader.args.perturb_steps, alpha=dataloader.args.step_size, rand_init=True)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss += criterion(outputs, targets).item()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg

def save_checkpoint(state, path=None, filename='checkpoint.pth.tar'):
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)
