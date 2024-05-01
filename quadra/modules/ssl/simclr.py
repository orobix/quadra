from __future__ import annotations

import sklearn
import torch
from torch import nn

from quadra.modules.base import SSLModule


class SimCLR(SSLModule):
    """SIMCLR class.

    Args:
        model: Feature extractor as pytorch `torch.nn.Module`
        projection_mlp: projection head as
            pytorch `torch.nn.Module`
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
        criterion: torch.nn.Module,
        classifier: sklearn.base.ClassifierMixin | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: object | None = None,
        lr_scheduler_interval: str | None = "epoch",
    ):
        super().__init__(
            model,
            criterion,
            classifier,
            optimizer,
            lr_scheduler,
            lr_scheduler_interval,
        )
        self.projection_mlp = projection_mlp

    def forward(self, x):
        x = self.model(x)
        x = self.projection_mlp(x)
        return x

    def training_step(
        self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Args:
            batch: The batch of data
            batch_idx: The index of the batch.

        Returns:
            The computed loss
        """
        # pylint: disable=unused-argument
        (im_x, im_y), _ = batch
        emb_x = self(im_x)
        emb_y = self(im_y)
        loss = self.criterion(emb_x, emb_y)

        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return loss
