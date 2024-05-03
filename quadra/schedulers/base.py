from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateScheduler(_LRScheduler):
    """Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer: Optimizer, init_lr: tuple[float, ...]):
        # pylint: disable=super-init-not-called
        self.optimizer = optimizer
        self.init_lr = init_lr

    def step(self, *args, **kwargs):
        """Base method, must be implemented by the sub classes."""
        raise NotImplementedError

    def set_lr(self, lr: tuple[float, ...]):
        """Set the learning rate for the optimizer."""
        if self.optimizer is not None:
            for i, g in enumerate(self.optimizer.param_groups):
                if "fix_lr" in g and g["fix_lr"]:
                    if len(lr) == 1:
                        lr_to_set = self.init_lr[0]
                    else:
                        lr_to_set = self.init_lr[i]
                elif len(lr) == 1:
                    lr_to_set = lr[0]
                else:
                    lr_to_set = lr[i]
                g["lr"] = lr_to_set

    def get_lr(self):
        """Get the current learning rate if the optimizer is available."""
        if self.optimizer is not None:
            for g in self.optimizer.param_groups:
                return g["lr"]

        return None
