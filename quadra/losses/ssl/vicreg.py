import torch


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float,
    mu: float,
    nu: float = 1,
    gamma: float = 1,
) -> torch.Tensor:
    """VICReg loss described in https://arxiv.org/abs/2105.04906.

    Args:
        z1: First `augmented` normalized features (i.e. f(T(x))). The normalization can be obtained with
            z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2: Second `augmented` normalized features (i.e. f(T(x))). The normalization can be obtained with
            z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        lambd: lambda multiplier for redundancy term.
        mu: mu multiplier for similarity term.
        nu: nu multiplier for variance term. Default: 1
        gamma: gamma multiplier for covariance term. Default: 1

    Returns:
        VICReg loss
    """
    # Variance loss
    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    v_z1 = torch.nn.functional.relu(gamma - std_z1).mean()
    v_z2 = torch.nn.functional.relu(gamma - std_z2).mean()
    var_loss = v_z1 + v_z2

    # Similarity loss
    sim_loss = torch.nn.functional.mse_loss(z1, z2)

    # Covariance loss
    n = z1.size(0)
    d = z1.size(1)
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (n - 1)
    cov_z2 = (z2.T @ z2) / (n - 1)
    off_diagonal_cov_z1 = cov_z1.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    off_diagonal_cov_z2 = cov_z2.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    cov_loss = off_diagonal_cov_z1.pow_(2).sum() / d + off_diagonal_cov_z2.pow_(2).sum() / d

    return lambd * sim_loss + mu * var_loss + nu * cov_loss


class VICRegLoss(torch.nn.Module):
    """VIC regression loss module.

    Args:
        lambd: lambda multiplier for redundancy term.
        mu: mu multiplier for similarity term.
        nu: nu multiplier for variance term. Default: 1.
        gamma: gamma multiplier for covariance term. Default: 1.
    """

    def __init__(
        self,
        lambd: float,
        mu: float,
        nu: float = 1,
        gamma: float = 1,
    ):
        super().__init__()
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.gamma = gamma

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes VICReg loss."""
        return vicreg_loss(z1, z2, self.lambd, self.mu, self.nu, self.gamma)
