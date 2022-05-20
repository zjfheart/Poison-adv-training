"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np

import pickle

import os
import warnings
import PIL

from .datasets import construct_datasets, Subset

from .diff_data_augmentation import RandomTransform

from .consts import PIN_MEMORY, BENCHMARK, SHARING_STRATEGY, MAX_THREADING
from .generic import set_random_seed
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class DataLoader():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - poisonloader
    - poison_ids
    - trainset/poisonset/targetset

    Most notably .poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison
    - export_poison

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float), logger=None):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.logger = logger
        self.batch_size = args.batch_size
        self.augmentations = args.augmentations
        self.trainset, self.validset = self.prepare_data(normalize=True)
        num_workers = self.get_num_workers()

        self.random_construction()


        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        validated_batch_size = max(min(args.pbatch, len(self.poisonset)), 1)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=False, drop_last=False, num_workers=num_workers,
                                                        pin_memory=PIN_MEMORY)

        self.print_status()


    """ STATUS METHODS """

    def print_status(self):
        class_names = self.trainset.classes
        self.logger.info(
            f'Poisoning setup generated for threat model single-class and '
            f'budget of {self.args.budget * 100}% - {len(self.poisonset)} images:')
        self.logger.info(
            f'--Target images drawn from class {", ".join([class_names[self.targetset[i][1]] for i in range(len(self.targetset))])}'
            f' with ids {self.target_ids}.')
        self.logger.info(f'--Target images assigned intended class {", ".join([class_names[i] for i in self.poison_setup["intended_class"]])}.')

        if self.poison_setup["poison_class"] is not None:
            self.logger.info(f'--Poison images drawn from class {class_names[self.poison_setup["poison_class"]]}.')
        else:
            self.logger.info(f'--Poison images drawn from all classes.')


    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        self.logger.info(f'Data is loaded with {worker_count} workers.')
        return worker_count

    """ CONSTRUCTION METHODS """

    def prepare_data(self, normalize=True):
        trainset, validset = construct_datasets(self.args.dataset, self.args.data_path, normalize)


        # Prepare data mean and std for later:
        self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup)
        self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup)


        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations is not None or self.args.paugment:
            if 'CIFAR' in self.args.dataset or 'SVHN' in self.args.dataset:
                params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
            elif 'MNIST' in self.args.dataset:
                params = dict(source_size=28, target_size=28, shift=4, fliplr=True)
            elif 'TinyImageNet' in self.args.dataset:
                params = dict(source_size=64, target_size=64, shift=64 // 4, fliplr=True)
            elif 'ImageNet' in self.args.dataset:
                params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)

            if self.augmentations == 'default':
                self.augment = RandomTransform(**params, mode='bilinear')
            elif not self.defs.augmentations:
                self.logger.info('Data augmentations are disabled.')
                self.augment = RandomTransform(**params, mode='bilinear')
            else:
                raise ValueError(f'Invalid diff. transformation given: {self.augmentations}.')

        return trainset, validset

    def random_construction(self):
        """Construct according to random selection.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup
         - poisonset / targetset / validset

        """
        if self.args.poisonkey is None:
            self.init_seed = np.random.randint(0, 2 ** 32 - 1)
        else:
            self.init_seed = int(self.args.poisonkey)
        set_random_seed(self.init_seed)
        self.logger.info(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {self.init_seed}')

        # Parse threat model
        self.poison_setup = self._parse_threats_randomly()
        self.poisonset, self.targetset, self.validset = self._choose_poisons_randomly()

    def _parse_threats_randomly(self):
        """Parse the different threat models.

        The threat-models are [In order of expected difficulty]:

        single-class replicates the threat model of feature collision attacks,
        third-party draws all poisons from a class that is unrelated to both target and intended label.
        random-subset draws poison images from all classes.
        random-subset draw poison images from all classes and draws targets from different classes to which it assigns
        different labels.
        """
        num_classes = len(self.trainset.classes)

        target_class = np.random.randint(num_classes)
        list_intentions = list(range(num_classes))
        list_intentions.remove(target_class)
        intended_class = [np.random.choice(list_intentions)]

        poison_class = intended_class[0]
        poison_setup = dict(poison_budget=self.args.budget, target_num=1,
                            poison_class=poison_class, target_class=target_class, intended_class=intended_class)

        return poison_setup

    def _choose_poisons_randomly(self):
        """Subconstruct poison and targets.

        The behavior is different for poisons and targets. We still consider poisons to be part of the original training
        set and load them via trainloader (And then add the adversarial pattern Delta)
        The targets are fully removed from the validation set and returned as a separate dataset, indicating that they
        should not be considered during clean validation using the validloader

        """
        # Poisons:
        if self.poison_setup['poison_class'] is not None:
            class_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.trainset.get_target(index)
                if target == self.poison_setup['poison_class']:
                    class_ids.append(idx)

            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(class_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(class_ids)}')
                poison_num = len(class_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                class_ids, size=poison_num, replace=False), dtype=torch.long)
        else:
            total_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.trainset.get_target(index)
                total_ids.append(idx)
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(total_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(total_ids)}')
                poison_num = len(total_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                total_ids, size=poison_num, replace=False), dtype=torch.long)

        # Targets:
        if self.poison_setup['target_class'] is not None:
            class_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.validset.get_target(index)
                if target == self.poison_setup['target_class']:
                    class_ids.append(idx)
            self.target_ids = np.random.choice(class_ids, size=1, replace=False)
        else:
            total_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.validset.get_target(index)
                total_ids.append(idx)
            self.target_ids = np.random.choice(total_ids, size=1, replace=False)

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        return poisonset, targetset, validset

    def initialize_poison(self, initializer=None):
        """Initialize according to args.init.

        Propagate initialization in distributed settings.
        """
        if initializer is None:
            initializer = self.args.init

        # ds has to be placed on the default (cpu) device, not like self.ds
        ds = torch.tensor(self.trainset.data_std)[None, :, None, None]
        if initializer == 'zero':
            init = torch.zeros(len(self.poison_ids), *self.trainset[0][0].shape)
        elif initializer == 'rand':
            init = (torch.rand(len(self.poison_ids), *self.trainset[0][0].shape) - 0.5) * 2
            init *= self.args.eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
            init *= self.args.eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
        else:
            raise NotImplementedError()

        init.data = torch.max(torch.min(init, self.args.eps / ds / 255), -self.args.eps / ds / 255)

        return init

    """ EXPORT METHODS """

    def export_poison(self, poison_delta, path=None, mode='full'):
        if path is None:
            poison_name = '{}_{}_eps{}_tau{}_poisonkey{}'.format(self.args.dataset, self.args.attackoptim, self.args.eps,
                                                                 self.args.tau, self.args.poisonkey)
            path = os.path.join('exp_data', 'targeted', poison_name, 'poisons')

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input = poison_delta[lookup, :, :, :] * 3
            _torch_to_PIL(input).save(filename)

        if mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'poisons', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)

            # Poisons
            for lookup, key in enumerate(self.poison_lookup.keys()):
                input, label, idx = self.trainset[key]
                input += poison_delta[lookup, :, :, :]
                _save_image(input, label, idx, location=os.path.join(path, 'poisons', names[label]), train=False)

        elif mode == 'numpy':
            _, h, w = self.trainset[0][0].shape
            training_data = np.zeros([len(self.trainset), h, w, 3])
            labels = np.zeros(len(self.trainset))
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    input += poison_delta[lookup, :, :, :]
                training_data[idx] = np.asarray(_torch_to_PIL(input))
                labels[idx] = label

            np.save(os.path.join(path, 'poisoned_training_data.npy'), training_data)
            np.save(os.path.join(path, 'poisoned_training_labels.npy'), labels)
        else:
            raise NotImplementedError()

        self.logger.info('Dataset fully exported.')
