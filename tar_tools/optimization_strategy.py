"""Optimization setups."""

from dataclasses import dataclass

BRITTLE_NETS = ['convnet', 'vgg']

def training_strategy(model_name, args):
    """Parse training strategy."""
    if args.strategy == 'standard':
        defs = ConservativeStrategy(model_name, args)
    elif args.strategy == 'adversarial':
        defs = AdversarialStrategy(model_name, args)
    return defs


@dataclass
class Strategy:
    """Default usual parameters, not intended for parsing."""

    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    augmentations : bool
    privacy : dict
    validate : int

    def __init__(self, model_name, args):
        """Defaulted parameters. Apply overwrites from args."""
        if any(net in model_name.lower() for net in BRITTLE_NETS):
            self.lr *= 0.1

@dataclass
class ConservativeStrategy(Strategy):
    """Default usual parameters, defines a config object."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.weight_decay = args.weight_decay
        self.augmentations = args.augmentations
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = 0
        self.validate = 10

        super().__init__(model_name, args)


@dataclass
class AdversarialStrategy(Strategy):
    """Implement adversarial training to defend against the poisoning."""

    def __init__(self, model_name, args):
        """Initialize training hyperparameters."""
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.weight_decay = args.weight_decay
        self.augmentations = args.augmentations
        self.privacy = dict(clip=None, noise=None)
        self.adversarial_steps = args.perturb_steps
        self.validate = 10

        super().__init__(model_name, args)

