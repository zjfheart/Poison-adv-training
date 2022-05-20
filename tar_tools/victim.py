"""Base victim class."""

import torch
import numpy as np

from .nets import get_model
from .training import get_optimizers, run_step
from .optimization_strategy import training_strategy
from .generic import set_random_seed


class Victim:
    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float), logger=None, is_victim=None):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        self.logger = logger
        self.initialize(is_victim=is_victim)

    def gradient(self, images, labels, criterion=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        if criterion is None:
            loss = self.criterion(self.model(images), labels)
        else:
            loss = criterion(self.model(images), labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), only_inputs=True)
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
        return gradients, grad_norm

    """ Methods to initialize a model."""

    def initialize(self, seed=None, is_victim=None):
        if self.args.modelkey is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.modelkey
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.criterion, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], is_victim)

        self.model.to(**self.setup)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        if self.args.pretrained is None or is_victim:
            self.logger.info(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, dataloader, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        self.logger.info('Starting training ...')
        self._iterate(dataloader, poison_delta=None, max_epoch=max_epoch)

    def validate(self, dataloader, poison_delta=None):
        """Check poison on a new initialization(s)."""
        for runs in range(self.args.vruns):
            self.initialize(is_victim=True)
            self._iterate(dataloader, poison_delta=poison_delta)

    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, dataloader, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""

        if max_epoch is None:
            max_epoch = self.defs.epochs

        def loss_fn(outputs, labels):
            return self.criterion(outputs, labels)

        single_setup = (self.model, self.defs, self.criterion, self.optimizer, self.scheduler)
        for self.epoch in range(max_epoch):
            run_step(dataloader, poison_delta, loss_fn, self.epoch, self.logger, *single_setup, net_seed=self.model_init_seed)

    def _initialize_model(self, model_name, is_victim):

        model = get_model(model_name, self.args.dataset, pretrained=self.args.pretrained, is_victim=is_victim)
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizers(model, self.args, defs)

        return model, defs, criterion, optimizer, scheduler


