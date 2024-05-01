from __future__ import annotations

import sklearn
import torch
from torch import nn, optim

from quadra.modules.base import SSLModule


class BarlowTwins(SSLModule):
    """BarlowTwins model.

    Args:
        model: Network Module used for extract features
        projection_mlp: Module to project extracted features
        criterion: SSL loss to be applied
        classifier: Standard sklearn classifier. Defaults to None.
        optimizer: optimizer of the training. If None a default Adam is used. Defaults to None.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used. Defaults to None.
        lr_scheduler_interval: interval at which the lr scheduler is updated. Defaults to "epoch".
    """

    def __init__(
        self,
        model: nn.Module,
        projection_mlp: nn.Module,
        criterion: nn.Module,
        classifier: sklearn.base.ClassifierMixin | None = None,
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: object | None = None,
        lr_scheduler_interval: str | None = "epoch",
    ):
        super().__init__(model, criterion, classifier, optimizer, lr_scheduler, lr_scheduler_interval)
        # self.save_hyperparameters()
        self.projection_mlp = projection_mlp
        self.criterion = criterion

    def forward(self, x):
        x = self.model(x)
        z = self.projection_mlp(x)
        return z

    def training_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int):
        # pylint: disable=unused-argument
        # Compute loss
        (im_x, im_y), _ = batch
        z1 = self(im_x)
        z2 = self(im_y)
        loss = self.criterion(z1, z2)

        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return loss
