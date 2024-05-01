from __future__ import annotations

import math
from collections.abc import Callable, Sized
from typing import Any

import sklearn
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.optim import Optimizer

from quadra.modules.base import SSLModule


class BYOL(SSLModule):
    """BYOL module, inspired by https://arxiv.org/abs/2006.07733.

    Args:
        student : student model.
        teacher : teacher model.
        student_projection_mlp : student projection MLP.
        student_prediction_mlp : student prediction MLP.
        teacher_projection_mlp : teacher projection MLP.
        criterion : loss function.
        classifier: Standard sklearn classifier.
        optimizer: optimizer of the training. If None a default Adam is used.
        lr_scheduler: lr scheduler. If None a default ReduceLROnPlateau is used.
        lr_scheduler_interval: interval at which the lr scheduler is updated.
        teacher_momentum: momentum of the teacher parameters.
        teacher_momentum_cosine_decay: whether to use cosine decay for the teacher momentum. Default: True
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_projection_mlp: nn.Module,
        student_prediction_mlp: nn.Module,
        teacher_projection_mlp: nn.Module,
        criterion: nn.Module,
        classifier: sklearn.base.ClassifierMixin | None = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: object | None = None,
        lr_scheduler_interval: str | None = "epoch",
        teacher_momentum: float = 0.9995,
        teacher_momentum_cosine_decay: bool | None = True,
    ):
        super().__init__(
            model=student,
            criterion=criterion,
            classifier=classifier,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
        )
        # Student model
        self.max_steps: int
        self.student_projection_mlp = student_projection_mlp
        self.student_prediction_mlp = student_prediction_mlp

        # Teacher model
        self.teacher = teacher
        self.teacher_projection_mlp = teacher_projection_mlp
        self.teacher_initialized = False
        self.teacher_momentum = teacher_momentum
        self.teacher_momentum_cosine_decay = teacher_momentum_cosine_decay

        self.initialize_teacher()

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

        for p in self.student_projection_mlp.parameters():
            assert p.requires_grad is True
        for p in self.student_prediction_mlp.parameters():
            assert p.requires_grad is True

        self.teacher_initialized = True

    def update_teacher(self):
        """Update teacher given `self.teacher_momentum` by an exponential moving average
        of the student parameters, that is: theta_t * tau + theta_s * (1 - tau), where
        `theta_{s,t}` are the parameters of the student and the teacher model, while `tau` is the
        teacher momentum. If `self.teacher_momentum_cosine_decay` is True, then the teacher
        momentum will follow a cosine scheduling from `self.teacher_momentum` to 1:
        tau = 1 - (1 - tau) * (cos(pi * t / T) + 1) / 2, where `t` is the current step and
        `T` is the max number of steps.
        """
        with torch.no_grad():
            if self.teacher_momentum_cosine_decay:
                teacher_momentum = (
                    1
                    - (1 - self.teacher_momentum)
                    * (math.cos(math.pi * self.trainer.global_step / self.max_steps) + 1)
                    / 2
                )
            else:
                teacher_momentum = self.teacher_momentum
            self.log("teacher_momentum", teacher_momentum, prog_bar=True)
            for student_ps, teacher_ps in zip(
                list(self.model.parameters()) + list(self.student_projection_mlp.parameters()),
                list(self.teacher.parameters()) + list(self.teacher_projection_mlp.parameters()),
                strict=False,
            ):
                teacher_ps.data = teacher_ps.data * teacher_momentum + (1 - teacher_momentum) * student_ps.data

    def on_train_start(self) -> None:
        if isinstance(self.trainer.train_dataloader, Sized) and isinstance(self.trainer.max_epochs, int):
            self.max_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        else:
            raise ValueError("BYOL requires `max_epochs` to be set and `train_dataloader` to be initialized.")

    def training_step(self, batch: tuple[list[torch.Tensor], torch.Tensor], *args: Any) -> torch.Tensor:
        [image1, image2], _ = batch

        online_pred_one = self.student_prediction_mlp(self.student_projection_mlp(self.model(image1)))
        online_pred_two = self.student_prediction_mlp(self.student_projection_mlp(self.model(image2)))

        with torch.no_grad():
            target_proj_one = self.teacher_projection_mlp(self.teacher(image1))
            target_proj_two = self.teacher_projection_mlp(self.teacher(image2))

        loss_one = self.criterion(online_pred_one, target_proj_two.detach())
        loss_two = self.criterion(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two

        self.log(name="loss", value=loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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

    def calculate_accuracy(self, batch):
        """Calculate accuracy on the given batch."""
        images, labels = batch
        embedding = self.model(images).detach().cpu().numpy()
        predictions = self.classifier.predict(embedding)
        labels = labels.detach()
        acc = self.val_acc(torch.tensor(predictions, device=self.device), labels)

        return acc

    def on_test_epoch_start(self) -> None:
        self.fit_estimator()

    def test_step(self, batch, *args: list[Any]) -> None:
        """Calculate accuracy on the test set for the given batch."""
        acc = self.calculate_accuracy(batch)
        self.log(name="test_acc", value=acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc
