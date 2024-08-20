from __future__ import annotations

from typing import Any, cast

import numpy as np
import timm
import torch
import torchmetrics
import torchmetrics.functional as TMF
from pytorch_grad_cam import GradCAM
from scipy import ndimage
from torch import nn, optim

from quadra.models.classification import BaseNetworkBuilder
from quadra.modules.base import BaseLightningModule
from quadra.utils.models import is_vision_transformer
from quadra.utils.utils import get_logger
from quadra.utils.vit_explainability import VitAttentionGradRollout

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
        optimizer: None | optim.Optimizer = None,
        lr_scheduler: None | object = None,
        lr_scheduler_interval: str | None = "epoch",
        gradcam: bool = False,
    ):
        super().__init__(model, optimizer, lr_scheduler, lr_scheduler_interval)

        self.criterion = criterion
        self.gradcam = gradcam
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.cam: GradCAM | None = None
        self.grad_rollout: VitAttentionGradRollout | None = None

        if not isinstance(self.model.features_extractor, timm.models.resnet.ResNet) and not is_vision_transformer(
            cast(BaseNetworkBuilder, self.model).features_extractor
        ):
            log.warning(
                "Backbone not compatible with gradcam. Only timm ResNets, timm ViTs and TorchHub dinoViTs supported",
            )
            self.gradcam = False

        self.original_requires_grads: list[bool] = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
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

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
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

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
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

    def prepare_gradcam(self) -> None:
        """Instantiate gradcam handlers."""
        if isinstance(self.model.features_extractor, timm.models.resnet.ResNet):
            target_layers = [cast(BaseNetworkBuilder, self.model).features_extractor.layer4[-1]]

            self.cam = GradCAM(
                model=self.model,
                target_layers=target_layers,
            )
            # Activating gradients
            for p in self.model.features_extractor.layer4[-1].parameters():
                p.requires_grad = True
        elif is_vision_transformer(cast(BaseNetworkBuilder, self.model).features_extractor):
            self.grad_rollout = VitAttentionGradRollout(self.model)
        else:
            log.warning("Gradcam not implemented for this backbone, it won't be computed")
            self.original_requires_grads.clear()
            self.gradcam = False

    def on_predict_start(self) -> None:
        """If gradcam, prepares gradcam and saves params requires_grad state."""
        if self.gradcam:
            # Saving params requires_grad state
            for p in self.model.parameters():
                self.original_requires_grads.append(p.requires_grad)
            self.prepare_gradcam()

        return super().on_predict_start()

    def on_predict_end(self) -> None:
        """If we computed gradcam, requires_grad values are reset to original value."""
        if self.gradcam:
            # Get back to initial state
            for i, p in enumerate(self.model.parameters()):
                p.requires_grad = self.original_requires_grads[i]

            # We are using GradCAM package only for resnets at the moment
            if isinstance(self.model.features_extractor, timm.models.resnet.ResNet) and self.cam is not None:
                # Needed to solve jitting bug
                self.cam.activations_and_grads.release()
            elif (
                is_vision_transformer(cast(BaseNetworkBuilder, self.model).features_extractor)
                and self.grad_rollout is not None
            ):
                for handle in self.grad_rollout.f_hook_handles:
                    handle.remove()
                for handle in self.grad_rollout.b_hook_handles:
                    handle.remove()

        return super().on_predict_end()

    # pylint: disable=unused-argument
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
        outputs = self(im)
        probs = torch.softmax(outputs, dim=1)
        predicted_classes = torch.max(probs, dim=1).indices.tolist()
        if self.gradcam:
            # inference_mode set to false because gradcam needs gradients
            with torch.inference_mode(False):
                im = im.clone()

                if isinstance(self.model.features_extractor, timm.models.resnet.ResNet) and self.cam:
                    grayscale_cam = self.cam(input_tensor=im, targets=None)
                elif (
                    is_vision_transformer(cast(BaseNetworkBuilder, self.model).features_extractor) and self.grad_rollout
                ):
                    grayscale_cam_low_res = self.grad_rollout(input_tensor=im, targets_list=predicted_classes)
                    orig_shape = grayscale_cam_low_res.shape
                    new_shape = (orig_shape[0], im.shape[2], im.shape[3])
                    zoom_factors = tuple(np.array(new_shape) / np.array(orig_shape))
                    grayscale_cam = ndimage.zoom(grayscale_cam_low_res, zoom_factors, order=1)
        else:
            grayscale_cam = None
        return predicted_classes, grayscale_cam, torch.max(probs, dim=1)[0].tolist()


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
        optimizer: None | optim.Optimizer = None,
        lr_scheduler: None | object = None,
        lr_scheduler_interval: str | None = "epoch",
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
                self.cam = GradCAM(model=model, target_layers=target_layers)

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
