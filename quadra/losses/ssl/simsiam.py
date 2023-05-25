import torch
import torch.nn.functional as F


def simsiam_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    """SimSIAM loss described in https://arxiv.org/abs/2011.10566.

    Args:
        p1: First `predicted` features (i.e. h(f(T(x1))))
        p2: Second `predicted` features (i.e. h(f(T'(x2))))
        z1: First 'projected features (i.e. f(T(x1)))
        z2: Second 'projected features (i.e. f(T(x2)))

    Returns:
        SimSIAM loss
    """
    return -(F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean()) * 0.5


class SimSIAMLoss(torch.nn.Module):
    """SimSIAM loss module."""

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the SimSIAM loss."""
        return simsiam_loss(p1, p2, z1, z2)
