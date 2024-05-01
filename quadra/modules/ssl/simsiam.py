from __future__ import annotations

import sklearn
import torch

from quadra.modules.base import SSLModule


class SimSIAM(SSLModule):
    """SimSIAM model.

    Args:
        model: Feature extractor as pytorch `torch.nn.Module`
        projection_mlp: optional projection head as pytorch `torch.nn.Module`
        prediction_mlp: optional predicition head as pytorch `torch.nn.Module`
        criterion: loss to be applied.
        classifier: Standard sklearn classifier.
        optimizer: optimizer of the training. If None a default Adam is used.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used.
        lr_scheduler_interval: interval at which the lr scheduler is updated.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        projection_mlp: torch.nn.Module,
        prediction_mlp: torch.nn.Module,
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
        # self.save_hyperparameters()
        self.projection_mlp = projection_mlp
        self.prediction_mlp = prediction_mlp

    def forward(self, x):
        x = self.model(x)
        z = self.projection_mlp(x)
        p = self.prediction_mlp(z)
        return p, z.detach()

    def training_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int):
        # pylint: disable=unused-argument
        # Compute loss
        (im_x, im_y), _ = batch
        p1, z1 = self(im_x)
        p2, z2 = self(im_y)
        loss = self.criterion(p1, p2, z1, z2)

        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return loss
