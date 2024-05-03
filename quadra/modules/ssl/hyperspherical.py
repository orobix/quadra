from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn, optim

from quadra.losses.ssl import hyperspherical as loss
from quadra.modules.base import BaseLightningModule


class AlignLoss(Enum):
    """Align loss enum."""

    L2 = 1
    COSINE = 2


class TLHyperspherical(BaseLightningModule):
    """Hyperspherical model: maps features extracted from a pretrained backbone into
    an hypersphere.

    Args:
        model: Feature extractor as pytorch `torch.nn.Module`
        optimizer: optimizer of the training.
            If None a default Adam is used.
        lr_scheduler: lr scheduler.
            If None a default ReduceLROnPlateau is used.
        align_weight: Weight for the align loss component for the
            hyperspherical loss.
            Defaults to 1.
        unifo_weight: Weight for the uniform loss component for the
            hyperspherical loss.
            Defaults to 1.
        classifier_weight: Weight for the classifier loss component for the
            hyperspherical loss.
            Defaults to 1.
        align_loss_type: Which type of align loss to use.
            Defaults to AlignLoss.L2.
        classifier_loss: Whether to compute a classifier loss to 'enhance'
            the hyperpsherical loss with the classification loss.
            It True, model.classifier must be defined
            Defaults to False.
        num_classes: Number of classes for a classification problem.
            Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: object | None = None,
        align_weight: float = 1,
        unifo_weight: float = 1,
        classifier_weight: float = 1,
        align_loss_type: AlignLoss = AlignLoss.L2,
        classifier_loss: bool = False,
        num_classes: int | None = None,
    ):
        super().__init__(model, optimizer, lr_scheduler)
        self.align_loss_fun: (
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
            | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        )
        self.align_weight = align_weight
        self.unifo_weight = unifo_weight
        self.classifier_weight = classifier_weight
        self.align_loss_type = align_loss_type
        if align_loss_type == AlignLoss.L2:
            self.align_loss_fun = loss.align_loss
        elif align_loss_type == AlignLoss.COSINE:
            self.align_loss_fun = loss.cosine_align_loss
        else:
            raise ValueError("The align loss must be one of 'AlignLoss.L2' (L2 distance) or AlignLoss.COSINE")

        if classifier_loss and model.classifier is None:
            raise AssertionError("Classifier is not defined")

        self.classifier_loss = classifier_loss
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        im_x, im_y, target = batch
        emb_x, emb_y = self(torch.cat([im_x, im_y])).chunk(2)

        align_loss = 0.0
        if self.align_weight > 0:
            align_loss = self.align_loss_fun(emb_x, emb_y)

        unifo_loss = 0.0
        if self.unifo_weight > 0:
            unifo_loss = (loss.uniform_loss(emb_x) + loss.uniform_loss(emb_y)) / 2

        classifier_loss = 0.0
        if self.classifier_loss:
            pred = self.model.classifier(emb_x)
            classifier_loss = F.cross_entropy(pred, target)

        total_loss = (
            self.align_weight * align_loss + self.unifo_weight * unifo_loss + self.classifier_weight * classifier_loss
        )

        self.log(
            "t_loss",
            total_loss,
            on_epoch=True,
            logger=True,
            prog_bar=False,
        )
        self.log(
            "t_align",
            align_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        self.log(
            "t_classifier",
            classifier_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "t_unif",
            unifo_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        return {"loss": total_loss}

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        return {"loss": avg_loss}

    def validation_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        im_x, im_y, target = batch
        emb_x, emb_y = self(torch.cat([im_x, im_y])).chunk(2)

        align_loss = 0.0
        if self.align_weight > 0:
            align_loss = self.align_loss_fun(emb_x, emb_y)

        unifo_loss = 0.0
        if self.unifo_weight > 0:
            unifo_loss = (loss.uniform_loss(emb_x) + loss.uniform_loss(emb_y)) / 2

        classifier_loss = 0.0
        if self.classifier_loss:
            pred = self.model.classifier(emb_x)
            classifier_loss = F.cross_entropy(pred, target)

        total_loss = (
            self.align_weight * align_loss + self.unifo_weight * unifo_loss + self.classifier_weight * classifier_loss
        )

        self.log(
            "val_loss",
            total_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        self.log(
            "v_classifier",
            classifier_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "v_align",
            align_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        self.log(
            "v_unif",
            unifo_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        return {"val_loss": total_loss}

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        return {"val_loss": avg_loss}
