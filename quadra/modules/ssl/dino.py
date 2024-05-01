from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sklearn
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.optim import Optimizer

from quadra.modules.ssl import BYOL
from quadra.utils.models import clip_gradients
from quadra.utils.utils import get_logger

log = get_logger(__name__)


class Dino(BYOL):
    """DINO pytorch-lightning module.

    Args:
        student : student model
        teacher : teacher model
        student_projection_mlp : student projection MLP
        teacher_projection_mlp : teacher projection MLP
        criterion : loss function
        freeze_last_layer : number of layers to freeze in the student model. Default: 1
        classifier: Standard sklearn classifier
        optimizer: optimizer of the training. If None a default Adam is used.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used.
        lr_scheduler_interval: interval at which the lr scheduler is updated.
        teacher_momentum: momentum of the teacher parameters
        teacher_momentum_cosine_decay: whether to use cosine decay for the teacher momentum
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_projection_mlp: nn.Module,
        teacher_projection_mlp: nn.Module,
        criterion: nn.Module,
        freeze_last_layer: int = 1,
        classifier: sklearn.base.ClassifierMixin | None = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: object | None = None,
        lr_scheduler_interval: str | None = "epoch",
        teacher_momentum: float = 0.9995,
        teacher_momentum_cosine_decay: bool | None = True,
    ):
        super().__init__(
            student=student,
            teacher=teacher,
            student_projection_mlp=student_projection_mlp,
            student_prediction_mlp=nn.Identity(),
            teacher_projection_mlp=teacher_projection_mlp,
            criterion=criterion,
            teacher_momentum=teacher_momentum,
            teacher_momentum_cosine_decay=teacher_momentum_cosine_decay,
            classifier=classifier,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
        )
        self.freeze_last_layer = freeze_last_layer

    def initialize_teacher(self):
        """Initialize teacher from the state dict of the student one,
        checking also that student model requires greadient correctly.
        """
        self.teacher_projection_mlp.load_state_dict(self.student_projection_mlp.state_dict())
        for p in self.teacher_projection_mlp.parameters():
            p.requires_grad = False

        self.teacher.load_state_dict(self.model.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        all_frozen = True
        for p in self.model.parameters():
            all_frozen = all_frozen and (not p.requires_grad)

        if all_frozen:
            log.warning(
                "All parameters of the student model are frozen, the model will not be trained, automatically"
                " unfreezing all the layers"
            )

            for p in self.model.parameters():
                p.requires_grad = True

        for name, p in self.student_projection_mlp.named_parameters():
            if name != "last_layer.weight_g":
                assert p.requires_grad is True

        self.teacher_initialized = True

    def student_multicrop_forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Student forward on the multicrop imges.

        Args:
            x: List of torch.Tensor containing multicropped augmented images

        Returns:
            torch.Tensor: a tensor of shape NxBxD, where N is the number crops
                corresponding to the length of the input list `x`, B is the batch size
                and D is the output dimension
        """
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, C, H, W)
        embedding = self.model(concatenated)  # (n_samples * n_crops, in_dim)
        logits = self.student_projection_mlp(embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)
        return chunks

    def teacher_multicrop_forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Teacher forward on the multicrop imges.

        Args:
            x: List of torch.Tensor containing multicropped augmented images

        Returns:
            torch.Tensor: a tensor of shape NxBxD, where N is the number crops
                corresponding to the length of the input list `x`, B is the batch size
                and D is the output dimension
        """
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)  # (n_samples * n_crops, C, H, W)
        embedding = self.teacher(concatenated)  # (n_samples * n_crops, in_dim)
        logits = self.teacher_projection_mlp(embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)
        return chunks

    def cancel_gradients_last_layer(self, epoch: int, freeze_last_layer: int):
        """Zero out the gradient of the last layer, as specified in the paper.

        Args:
            epoch: current epoch
            freeze_last_layer: maximum freeze epoch: if `epoch` >= `freeze_last_layer`
                then the gradient of the last layer will not be freezed
        """
        if epoch >= freeze_last_layer:
            return
        for n, p in self.student_projection_mlp.named_parameters():
            if "last_layer" in n:
                p.grad = None

    def training_step(self, batch: tuple[list[torch.Tensor], torch.Tensor], *args: Any) -> torch.Tensor:
        images, _ = batch
        with torch.no_grad():
            teacher_output = self.teacher_multicrop_forward(images[:2])

        student_output = self.student_multicrop_forward(images)
        loss = self.criterion(self.current_epoch, student_output, teacher_output)

        self.log(name="loss", value=loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ):
        """Configure gradient clipping for the optimizer."""
        if gradient_clip_algorithm is not None and gradient_clip_val is not None:
            clip_gradients(self.model, gradient_clip_val)
            clip_gradients(self.student_projection_mlp, gradient_clip_val)
        self.cancel_gradients_last_layer(self.current_epoch, self.freeze_last_layer)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """Override optimizer step to update the teacher parameters."""
        super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure=optimizer_closure,
        )
        self.update_teacher()
