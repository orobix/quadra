import math

import torch
import torch.nn.functional as F

from quadra.utils.utils import AllGatherSyncFunction


def simclr_loss(
    features1: torch.Tensor,
    features2: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """SimCLR loss described in https://arxiv.org/pdf/2002.05709.pdf.

    Args:
        temperature: optional temperature
        features1: First augmented features (i.e. T(features))
        features2: Second augmented features (i.e. T'(features))

    Returns:
        SimCLR loss
    """
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        features1_dist = AllGatherSyncFunction.apply(features1)
        features2_dist = AllGatherSyncFunction.apply(features2)
    else:
        features1_dist = features1
        features2_dist = features2
    features = torch.cat([features1, features2], dim=0)  # [2B, d]
    features_dist = torch.cat([features1_dist, features2_dist], dim=0)  # [2B * DIST_SIZE, d]

    # Similarity matrix
    sim = torch.exp(torch.div(torch.mm(features, features_dist.t()), temperature))  # [2B, 2B * DIST_SIZE]

    # Negatives
    neg = sim.sum(dim=-1)

    # From each row, subtract e^(1/temp) to remove similarity measure for zi * zi, since
    # (zi^T * zi) / ||zi||^2 = 1
    row_sub = torch.full_like(neg, math.e ** (1 / temperature), device=neg.device)
    neg = torch.clamp(neg - row_sub, min=1e-6)  # clamp for numerical stability

    # Positive similarity, pos becomes [2 * batch_size]
    pos = torch.exp(torch.div(torch.sum(features1 * features2, dim=-1), temperature))
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + 1e-6)).mean()
    return loss


class SimCLRLoss(torch.nn.Module):
    """SIMCLRloss module.

    Args:
        temperature: temperature of SIM loss.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss."""
        return simclr_loss(x1, x2, temperature=self.temperature)
