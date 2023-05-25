from typing import Any, List, Optional, Tuple, Union, cast

import timm
import torch
import torchmetrics
import torchmetrics.functional as TMF
from pytorch_grad_cam import GradCAM
from torch import nn, optim

from quadra.models.classification import BaseNetworkBuilder
from quadra.modules.base import BaseLightningModule
from quadra.utils.utils import get_logger

log = get_logger(__name__)


class ClassificationModule(BaseLightningModule):
    """Lightning module for classification tasks.

    Args:
        model: Feature extractor as PyTorch `torch.nn.Module`
        criterion: the loss to be applied as a PyTorch `torch.nn.Module`.
        optimizer: optimizer of the training. Defaults to None.
        lr_scheduler: Pytorch learning rate scheduler.
            If None a default ReduceLROnPlateau is used.
            Defaults to None.
        lr_scheduler_interval: the learning rate scheduler interval.
            Defaults to "epoch".
        gradcam (bool): Whether to compute gradcam during prediction step
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Union[None, optim.Optimizer] = None,
        lr_scheduler: Union[None, object] = None,
        lr_scheduler_interval: Optional[str] = "epoch",
        gradcam: bool = False,
    ):
        super().__init__(model, optimizer, lr_scheduler, lr_scheduler_interval)

        self.criterion = criterion
        self.gradcam = gradcam
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.cam: GradCAM

        if not isinstance(self.model.features_extractor, timm.models.resnet.ResNet):
            log.warning(
                "Backbone must be compatible with gradcam, at the moment only ResNets supported, disabling gradcam"
            )
            self.gradcam = False

        self.original_requires_grads: List[bool] = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # pylint: disable=unused-argument
        im, target = batch
        outputs = self(im)
        loss = self.criterion(outputs, target)

        self.log_dict(
            {"train_loss": loss},
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        self.log_dict(
            {"train_acc": self.train_acc(outputs.argmax(1), target)},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # pylint: disable=unused-argument
        im, target = batch
        outputs = self(im)
        loss = self.criterion(outputs, target)

        self.log_dict(
            {"val_loss": loss},
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        self.log_dict(
            {"val_acc": self.val_acc(outputs.argmax(1), target)},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        # pylint: disable=unused-argument
        im, target = batch
        outputs = self(im)

        loss = self.criterion(outputs, target)

        self.log_dict(
            {"test_loss": loss},
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=False,
        )
        self.log_dict(
            {"test_acc": self.test_acc(outputs.argmax(1), target)},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
        )

    def on_predict_start(self) -> None:
        """If gradcam will be computed, saves all requires_grad values and set them to True before the predict."""
        if self.gradcam:
            target_layers = [cast(BaseNetworkBuilder, self.model).features_extractor.layer4[-1]]  # type: ignore[index]
            self.cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

            for p in self.model.parameters():
                self.original_requires_grads.append(p.requires_grad)
                p.requires_grad = True

        return super().on_predict_start()

    def on_predict_end(self) -> None:
        """If we computed gradcam, requires_grad values are reset to original value."""
        if self.gradcam:
            for i, p in enumerate(self.model.parameters()):
                p.requires_grad = self.original_requires_grads[i]

            self.cam.activations_and_grads.release()
        return super().on_predict_end()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Prediction step.

        Args:
            batch: Tuple composed by (image, target)
            batch_idx: Batch index
            dataloader_idx: Dataloader index
        Returns:
            Tuple containing:
                predicted_classes: indexes of predicted classes
                grayscale_cam: gray scale gradcams
        """
        im, _ = batch
        # inference_mode set to false because gradcam needs gradients
        if self.gradcam:
            with torch.inference_mode(False):
                im = im.clone()
                outputs = self(im)
                probs = torch.softmax(outputs, dim=1)
                predicted_classes = torch.max(probs, dim=1).indices.tolist()
                grayscale_cam = self.cam(input_tensor=im, targets=None)
        else:
            outputs = self(im)
            probs = torch.softmax(outputs, dim=1)
            predicted_classes = torch.max(probs, dim=1).indices.tolist()
            grayscale_cam = None
        return predicted_classes, grayscale_cam


class MultilabelClassificationModule(BaseLightningModule):
    """SklearnClassification model: train a generic SklearnClassification model for a multilabel
    problem.

    Args:
        model: Feature extractor as PyTorch `torch.nn.Module`
        criterion: the loss to be applied as a PyTorch `torch.nn.Module`.
        optimizer: optimizer of the training. Defaults to None.
        lr_scheduler: Pytorch learning rate scheduler.
            If None a default ReduceLROnPlateau is used.
            Defaults to None.
        lr_scheduler_interval: the learning rate scheduler interval.
            Defaults to "epoch".
    """

    def __init__(
        self,
        model: nn.Sequential,
        criterion: nn.Module,
        optimizer: Union[None, optim.Optimizer] = None,
        lr_scheduler: Union[None, object] = None,
        lr_scheduler_interval: Optional[str] = "epoch",
        gradcam: bool = False,
    ):
        super().__init__(model, optimizer, lr_scheduler, lr_scheduler_interval)
        self.criterion = criterion
        self.gradcam = gradcam

        # TODO: can we use gradcam with more backbones?
        if self.gradcam:
            if not isinstance(model[0].features_extractor, timm.models.resnet.ResNet):
                log.warning(
                    "Backbone must be compatible with gradcam, at the moment only ResNets supported, disabling gradcam"
                )
                self.gradcam = False
            else:
                target_layers = [model[0].features_extractor.layer4[-1]]
                self.cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        im, target = batch
        outputs = self(im)
        with torch.no_grad():
            outputs_sig = torch.sigmoid(outputs)
        loss = self.criterion(outputs, target)

        self.log_dict(
            {
                "t_loss": loss,
                "t_map": TMF.label_ranking_average_precision(outputs_sig, target.bool()),
                "t_f1": TMF.f1_score(outputs_sig, target.bool(), average="samples"),
            },
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        im, target = batch
        outputs = self(im)
        with torch.no_grad():
            outputs_sig = torch.sigmoid(outputs)
        loss = self.criterion(outputs, target)

        self.log_dict(
            {
                "val_loss": loss,
                "val_map": TMF.label_ranking_average_precision(outputs_sig, target.bool()),
                "val_f1": TMF.f1_score(outputs_sig, target.bool(), average="samples"),
            },
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        im, target = batch
        outputs = self(im)
        with torch.no_grad():
            outputs_sig = torch.sigmoid(outputs)
        loss = self.criterion(outputs, target)

        self.log_dict(
            {
                "test_loss": loss,
                "test_map": TMF.label_ranking_average_precision(outputs_sig, target.bool()),
                "test_f1": TMF.f1_score(outputs_sig, target.bool(), average="samples"),
            },
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=False,
        )
        return loss
