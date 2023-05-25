import torch
from torch.nn.functional import cosine_similarity


def cosine_align_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes mean of cosine distance based on similarity mean(1 - cosine_similarity).

    Args:
        x: feature n1
        y: feature n2.

    Returns:
        cosine align loss
    """
    cos = 1 - cosine_similarity(x, y, dim=1)
    return torch.mean(cos)


# Source: https://arxiv.org/pdf/2005.10242.pdf
def align_loss(x: torch.Tensor, y: torch.Tensor, alpha: int = 2) -> torch.Tensor:
    """Mean(l2^alpha).

    Args:
        x: feature n1
        y: feature n2
        alpha: pow of the norm loss.

    Returns:
        Align loss
    """
    norm = torch.norm(x - y, p=2, dim=1)
    return torch.mean(torch.pow(norm, alpha))


def uniform_loss(x: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """log(mean(exp(-t*dist_p2))).

    Args:
        x: feature tensor
        t: temperature of the dist_p2.

    Returns:
        Uniform loss
    """
    return torch.log(torch.mean(torch.exp(torch.pow(torch.pdist(x, p=2), 2) * -t)))
