import torch
import torch.nn.functional as F


def idmm_loss(
    p1: torch.Tensor,
    y1: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """IDMM loss described in https://arxiv.org/abs/2201.10728.

    Args:
        p1: Prediction labels for `z1`
        y1: Instance labels for `z1`
        smoothing: smoothing factor used for label smoothing.
            Defaults to 0.1.

    Returns:
        IDMM loss
    """
    loss = F.cross_entropy(p1, y1, label_smoothing=smoothing)
    return loss


class IDMMLoss(torch.nn.Module):
    """IDMM loss described in https://arxiv.org/abs/2201.10728."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        p1: torch.Tensor,
        y1: torch.Tensor,
    ) -> torch.Tensor:
        """IDMM loss described in https://arxiv.org/abs/2201.10728.

        Args:
            p1: Prediction labels for `z1`
            y1: Instance labels for `z1`

        Returns:
            IDMM loss
        """
        return idmm_loss(
            p1,
            y1,
            self.smoothing,
        )
