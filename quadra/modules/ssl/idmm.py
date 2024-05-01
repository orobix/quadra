from __future__ import annotations

import sklearn
import timm
import torch
import torch.nn.functional as F

from quadra.modules.base import SSLModule


class IDMM(SSLModule):
    """IDMM model.

    Args:
        model: backbone model
        prediction_mlp: student prediction MLP
        criterion: loss function
        multiview_loss: whether to use the multiview loss as definied in https://arxiv.org/abs/2201.10728.
            Defaults to True.
        mixup_fn: the mixup/cutmix function to be applied to a batch of images.
            Defaults to None.
        classifier: Standard sklearn classifier
        optimizer: optimizer of the training. If None a default Adam is used.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used.
        lr_scheduler_interval: interval at which the lr scheduler is updated.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prediction_mlp: torch.nn.Module,
        criterion: torch.nn.Module,
        multiview_loss: bool = True,
        mixup_fn: timm.data.Mixup | None = None,
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
        self.prediction_mlp = prediction_mlp
        self.mixup_fn = mixup_fn
        self.multiview_loss = multiview_loss

    def forward(self, x):
        z = self.model(x)
        p = self.prediction_mlp(z)
        return z, p

    def training_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        # Compute loss
        if self.multiview_loss:
            im_x, im_y, target = batch

            # Contrastive loss
            za, _ = self(im_x)
            zb, _ = self(im_y)
            za = F.normalize(za, dim=-1)
            zb = F.normalize(zb, dim=-1)
            s_aa = za.T @ za
            s_ab = za.T @ zb
            contrastive = (
                torch.log(torch.exp(s_aa).sum(-1))
                - torch.diagonal(s_aa)
                + torch.log(torch.exp(s_ab).sum(-1))
                - torch.diagonal(s_ab)
            )

            # Instance discrimination
            if self.mixup_fn is not None:
                im_x, target = self.mixup_fn(im_x, target)
            _, pred = self(im_x)
            loss = self.criterion(pred, target) + contrastive.mean()
        else:
            im_x, target = batch
            if self.mixup_fn is not None:
                im_x, target = self.mixup_fn(im_x, target)
            pred = self(im_x)
            loss = self.criterion(pred, target)

        self.log(
            "loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return loss
