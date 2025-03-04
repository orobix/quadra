from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import sklearn
import torch
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.optim import Optimizer

from quadra.models.base import ModelSignatureWrapper

__all__ = ["BaseLightningModule", "SSLModule"]


class BaseLightningModule(pl.LightningModule):
    """Base lightning module.

    Args:
        model: Network Module used for extract features
        optimizer: optimizer of the training. If None a default Adam is used.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        lr_scheduler: object | None = None,
        lr_scheduler_interval: str | None = "epoch",
    ):
        super().__init__()
        self.model = ModelSignatureWrapper(model)
        self.optimizer = optimizer
        self.schedulers = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method
        Args:
            x: input tensor.

        Returns:
            model inference
        """
        return self.model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Get default optimizer if not passed a value.

        Returns:
            optimizer and lr scheduler as Tuple containing a list of optimizers and a list of lr schedulers
        """
        # get default optimizer
        if getattr(self, "optimizer", None) is None or not self.optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        # get default scheduler
        if getattr(self, "schedulers", None) is None or not self.schedulers:
            self.schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=30)

        lr_scheduler_conf = {
            "scheduler": self.schedulers,
            "interval": self.lr_scheduler_interval,
            "monitor": "val_loss",
            "strict": False,
        }
        return [self.optimizer], [lr_scheduler_conf]  # type: ignore[return-value]

    # pylint: disable=unused-argument
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx: int = 0):
        """Redefine optimizer zero grad."""
        optimizer.zero_grad(set_to_none=True)


class SSLModule(BaseLightningModule):
    """Base module for self supervised learning.

    Args:
        model: Network Module used for extract features
        criterion: SSL loss to be applied
        classifier: Standard sklearn classifiers
        optimizer: optimizer of the training. If None a default Adam is used.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        classifier: sklearn.base.ClassifierMixin | None = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: object | None = None,
        lr_scheduler_interval: str | None = "epoch",
    ):
        super().__init__(model, optimizer, lr_scheduler, lr_scheduler_interval)
        self.criterion = criterion
        self.classifier_train_loader: torch.utils.data.DataLoader | None
        if classifier is None:
            self.classifier = LogisticRegression(max_iter=10000, n_jobs=8, random_state=42)
        else:
            self.classifier = classifier

        self.val_acc = torchmetrics.Accuracy()

    def fit_estimator(self):
        """Fit a classifier on the embeddings extracted from the current trained model."""
        targets = []
        train_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for im, target in self.classifier_train_loader:
                emb = self.model(im.to(self.device))
                targets.append(target)
                train_embeddings.append(emb)
        targets = torch.cat(targets, dim=0).cpu().numpy()
        train_embeddings = torch.cat(train_embeddings, dim=0).cpu().numpy()
        self.classifier.fit(train_embeddings, targets)

    def calculate_accuracy(self, batch):
        """Calculate accuracy on a batch of data."""
        images, labels = batch
        with torch.no_grad():
            embedding = self.model(images).cpu().numpy()

        predictions = self.classifier.predict(embedding)
        labels = labels.detach()
        acc = self.val_acc(torch.tensor(predictions, device=self.device), labels)

        return acc

    # TODO: In multiprocessing mode, this function is called multiple times, how can we avoid this?
    def on_validation_start(self) -> None:
        if not hasattr(self, "classifier_train_loader") and hasattr(self.trainer, "datamodule"):
            self.classifier_train_loader = self.trainer.datamodule.classifier_train_dataloader()

        if self.classifier_train_loader is not None:
            self.fit_estimator()

    def validation_step(self, batch: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], batch_idx: int) -> None:
        # pylint: disable=unused-argument
        if self.classifier_train_loader is None:
            # Compute loss
            (im_x, im_y), _ = batch
            z1 = self(im_x)
            z2 = self(im_y)
            loss = self.criterion(z1, z2)

            self.log(
                "val_loss",
                loss,
                on_epoch=True,
                on_step=True,
                logger=True,
                prog_bar=True,
            )
            return loss

        acc = self.calculate_accuracy(batch)
        self.log("val_acc", acc, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        return None


class SegmentationModel(BaseLightningModule):
    """Generic segmentation model.

    Args:
        model: segmentation model to be used.
        loss_fun: loss function to be used.
        optimizer: Optimizer to be used. Defaults to None.
        lr_scheduler: lr scheduler to be used. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun: Callable,
        optimizer: Optimizer | None = None,
        lr_scheduler: object | None = None,
    ):
        super().__init__(model, optimizer, lr_scheduler)
        self.loss_fun = loss_fun

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method
        Args:
            x: input tensor.

        Returns:
            model inference
        """
        x = self.model(x)
        return x

    def step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss
        Args:
            batch: batch.

        Returns:
            Prediction and target masks
        """
        images, target_masks, _ = batch
        pred_masks = self(images)
        if len(pred_masks.shape) == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if len(target_masks.shape) == 3:
            target_masks = target_masks.unsqueeze(1)
        assert pred_masks.shape == target_masks.shape

        return pred_masks, target_masks

    def compute_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Compute loss
        Args:
            pred_masks: predicted masks
            target_masks: target masks.

        Returns:
            The computed loss

        """
        loss = self.loss_fun(pred_masks, target_masks)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step."""
        # pylint: disable=unused-argument
        pred_masks, target_masks = self.step(batch)
        loss = self.compute_loss(pred_masks, target_masks)
        self.log_dict(
            {"loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        """Validation step."""
        # pylint: disable=unused-argument
        pred_masks, target_masks = self.step(batch)
        loss = self.compute_loss(pred_masks, target_masks)
        self.log_dict(
            {"val_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step."""
        # pylint: disable=unused-argument
        pred_masks, target_masks = self.step(batch)
        loss = self.compute_loss(pred_masks, target_masks)
        self.log_dict(
            {"test_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> Any:
        """Predict step."""
        # pylint: disable=unused-argument
        images, masks, labels = batch
        pred_masks = self(images)
        return images.cpu(), masks.cpu(), pred_masks.cpu(), labels.cpu()


class SegmentationModelMulticlass(SegmentationModel):
    """Generic multiclass segmentation model.

    Args:
        model: segmentation model to be used.
        loss_fun: loss function to be used.
        optimizer: Optimizer to be used. Defaults to None.
        lr_scheduler: lr scheduler to be used. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fun: Callable,
        optimizer: Optimizer | None = None,
        lr_scheduler: object | None = None,
    ):
        super().__init__(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_fun=loss_fun)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute step
        Args:
            batch: batch.

        Returns:
            prediction, target

        """
        images, target_masks, _ = batch
        pred_masks = self(images)

        return pred_masks, target_masks
