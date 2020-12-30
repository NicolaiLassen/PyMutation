from torch.nn import Module
from torch.optim import Optimizer


class _Mutator(object):
    def __init__(self, model, optimizer, verbose=False):

        # Attach module
        if not isinstance(model, Module):
            raise TypeError('{} is not an Module'.format(
                type(model).__name__))

        self.model = model

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.verbose = verbose

        def step(self, epoch=None):
            print()


class Plasticity(_Mutator):

    def __init__(self, model, optimizer, hebb=0, verbose=False):
        self.model = model
        self.optimizer = optimizer


class Plasticity(_Mutator):

    def __init__(self, optimizer, hebb=0, verbose=False):
        self.optimizer = optimizer

