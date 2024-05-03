from __future__ import annotations

import math

import torch

from quadra.schedulers.base import LearningRateScheduler
from quadra.utils.utils import get_logger

log = get_logger(__name__)


def cosine_annealing_with_warmup(
    init_lrs: list[float],
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_reduce_factor: float = 0.001,
) -> list[float]:
    """Cosine learning rate scheduler with linear warmup helper function.

    Args:
        init_lrs: The initial learning rate, one for every `param_group`.
        step: the current step
        total_steps: the total steps
        warmup_steps: total linear warmup steps
        lr_reduce_factor: reduce factor for the initial learning
            rate. This is used to set the minimum learning rate as
            `init_lr[i] * lr_reduce_factor`
            Defaults to 0.001.

    Returns:
        Annealed learning rate for this `step`
    """
    lrs = []
    for init_lr in init_lrs:
        if step < warmup_steps:
            lr = init_lr * step / warmup_steps
        else:
            step -= warmup_steps
            total_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            end_lr = init_lr * lr_reduce_factor
            lr = init_lr * q + end_lr * (1 - q)
        lrs.append(lr)
    return lrs


class CosineAnnealingWithLinearWarmUp(LearningRateScheduler):
    """Cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: optimizer for which the learning rate
            has to be optimized. If your are using this scheduler, than you have
            set the learning rate of the optimizer to 0
        batch_size: global batch size of the data loader. For more information please take a look at
            https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html?highlight=batch%20size#batch-size
        total_epochs: the total number of epochs
        init_lr: The initial learning rate, one for every `param_group`. Mind that the
            learning rate it's linearly scaled by `batch_size` / `lr_scale`, as
            specified by https://arxiv.org/abs/1706.02677.
            Defaults to 0.01.
        lr_scale: the learning rate scheduler. Mind that the learning rate it's linearly
            scaled by `batch_size` / `lr_scale` as specified by https://arxiv.org/abs/1706.02677.
            Defaults to 256.
        linear_warmup_epochs: how many epochs for the initial
            linear learning rate scaling.
            Defaults to 10.
        lr_reduce_factor: factor to be multiplied by scaled
            lr (init_lr * batch_size / lr_scale) to avoid reaching 0 lr
            at the end of training.
        len_loader: number of batches in a given dataloader. Remind that
            the `len_loader` must be divided by total number of gpus used during the
            training. If one specifies the `len_loader` parameter, then the unit measure
            for the lr update will be in steps (number of batches), not in epochs.
            Defaults to None.
        scheduler_interval: 'step' or 'epoch'. If 'step' then the
            scheduler expects 'len_loader' to be not None.
            Defaults to `epoch`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        total_epochs: int,
        init_lr: tuple[float, ...] = (0.01,),
        lr_scale: float = 256.0,
        linear_warmup_epochs: int = 10,
        lr_reduce_factor: float = 0.001,
        len_loader: int | None = None,
        scheduler_interval: str = "epoch",
    ) -> None:
        super().__init__(optimizer, init_lr)
        assert batch_size > 0
        assert total_epochs > 0
        assert lr_scale != 0
        assert linear_warmup_epochs >= 0
        assert lr_reduce_factor > 0
        assert scheduler_interval.lower() in ["step", "epoch"]
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.linear_warmup_epochs = linear_warmup_epochs
        self.base_lr_scale = self.batch_size / lr_scale
        self.updates_counter = 1
        self.init_lr = tuple(lr * self.base_lr_scale for lr in init_lr)
        self.lr = self.init_lr
        self.lr_reduce_factor = lr_reduce_factor

        self.scheduler_interval = scheduler_interval.lower()
        if self.scheduler_interval == "step":
            assert len_loader is not None and len_loader > 0
            self.total_epochs = len_loader * self.total_epochs
            self.linear_warmup_epochs = len_loader * self.linear_warmup_epochs

    def step(self):
        """Update the learning rate for the current step."""
        self.lr = cosine_annealing_with_warmup(
            self.init_lr,
            self.updates_counter,
            self.total_epochs,
            self.linear_warmup_epochs,
            self.lr_reduce_factor,
        )
        self.set_lr(self.lr)
        self.updates_counter += 1
        return self.lr
