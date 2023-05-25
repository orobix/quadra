import torch
import torch.nn.functional as F
from torch import nn


def byol_regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Byol regression loss
    Args:
        x: tensor
        y: tensor.

    Returns:
        tensor
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=1).mean()


class BYOLRegressionLoss(nn.Module):
    """BYOL regression loss module."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the BYOL regression loss.

        Args:
            x: First Tensor
            y: Second Tensor

        Returns:
            BYOL regression loss
        """
        return byol_regression_loss(x, y)
