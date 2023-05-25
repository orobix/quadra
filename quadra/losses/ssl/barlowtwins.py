import torch


def barlowtwins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float,
) -> torch.Tensor:
    """BarlowTwins loss described in https://arxiv.org/abs/2103.03230.

    Args:
        z1: First `augmented` normalized features (i.e. f(T(x))).
            The normalization can be obtained with
            z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2: Second `augmented` normalized features (i.e. f(T(x))).
            The normalization can be obtained with
            z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        lambd: lambda multiplier for redundancy term.

    Returns:
        BarlowTwins loss
    """
    z1 = (z1 - z1.mean(0)) / z1.std(0)
    z1 = (z2 - z2.mean(0)) / z2.std(0)
    cov = z1.T @ z2
    cov.div_(z1.size(0))
    n = cov.size(0)
    invariance_term = torch.diagonal(cov).add_(-1).pow_(2).sum()
    off_diag = cov.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    redundancy_term = off_diag.pow_(2).sum()
    return invariance_term + lambd * redundancy_term


class BarlowTwinsLoss(torch.nn.Module):
    """BarlowTwin loss.

    Args:
        lambd: lambda of the loss.
    """

    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the BarlowTwins loss."""
        return barlowtwins_loss(z1, z2, self.lambd)
