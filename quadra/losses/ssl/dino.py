import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


def dino_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    center_vector: torch.Tensor,
    teacher_temp: float = 0.04,
    student_temp: float = 0.1,
) -> torch.Tensor:
    """Compute the DINO distillation loss.

    Args:
        student_output: tensor of the student output
        teacher_output: tensor of the teacher output
        center_vector: center vector of distribution
        teacher_temp: temperature teacher
        student_temp: temperature student.

    Returns:
        The computed loss
    """
    student_temp = [s / student_temp for s in student_output]
    teacher_temp = [(t - center_vector) / teacher_temp for t in teacher_output]

    student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
    teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

    total_loss = torch.tensor(0.0, device=student_output[0].device)
    n_loss_terms = torch.tensor(0.0, device=student_output[0].device)

    for t_ix, t in enumerate(teacher_sm):
        for s_ix, s in enumerate(student_sm):
            if t_ix == s_ix:
                continue

            loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
            total_loss += loss.mean()  # scalar
            n_loss_terms += 1

    total_loss /= n_loss_terms
    return total_loss


class DinoDistillationLoss(nn.Module):
    """Dino distillation loss module.

    Args:
        output_dim: output dim.
        max_epochs: max epochs.
        warmup_teacher_temp: warmup temperature.
        teacher_temp: teacher temperature.
        warmup_teacher_temp_epochs: warmup teacher epocs.
        student_temp: student temperature.
        center_momentum: center momentum.
    """

    def __init__(
        self,
        output_dim: int,
        max_epochs: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center: torch.Tensor
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning

        if warmup_teacher_temp_epochs >= max_epochs:
            raise ValueError(
                f"Number of warmup epochs ({warmup_teacher_temp_epochs}) must be smaller than max_epochs ({max_epochs})"
            )

        if warmup_teacher_temp_epochs < 30:
            log.warning("Warmup teacher epochs is very small (< 30). This may cause instabilities in the training")

        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(max_epochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(
        self,
        current_epoch: int,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """Runs forward."""
        teacher_temp = self.teacher_temp_schedule[current_epoch]
        loss = dino_distillation_loss(
            student_output,
            teacher_output,
            center_vector=self.center,
            teacher_temp=teacher_temp,
            student_temp=self.student_temp,
        )

        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Update center of the distribution of the teacher
        Args:
            teacher_output: teacher output.

        Returns:
            None
        """
        # TODO: check if this is correct
        # torch.cat expects a list of tensors but teacher_output is a tensor
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)  # type: ignore[call-overload]
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
