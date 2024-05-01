from __future__ import annotations

import torch
from torch import nn

from quadra.utils.models import trunc_normal_


class ProjectionHead(torch.nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (linear_layer, batch_norm_layer, non_linearity_layer).
            `batch_norm` layer can be possibly None, the same happens for
            `non_linearity_layer`.
    """

    def __init__(self, blocks: list[tuple[torch.nn.Module | None, ...]]):
        super().__init__()

        layers: list[nn.Module] = []
        for linear, batch_norm, non_linearity in blocks:
            if linear:
                layers.append(linear)
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class ExpanderReducer(ProjectionHead):
    """Expander followed by a reducer."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (
                    torch.nn.Linear(hidden_dim, output_dim, bias=False),
                    torch.nn.BatchNorm1d(output_dim, affine=False),
                    torch.nn.ReLU(inplace=True),
                ),
            ]
        )


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.
    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." https://arxiv.org/abs/2103.03230.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (torch.nn.Linear(hidden_dim, output_dim, bias=False), None, None),
            ]
        )


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.
    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." https://arxiv.org/abs/2002.05709.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim),
                    None,
                    torch.nn.ReLU(inplace=True),
                ),
                (torch.nn.Linear(hidden_dim, output_dim), None, None),
            ]
        )


class SimCLRPredictionHead(ProjectionHead):
    """Prediction head used for SimCLR.
    "We set g(h) = W(2)σ(W(1)h), with the same input and output dimensionality (i.e. 2048)."
    https://arxiv.org/abs/2002.05709.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (torch.nn.Linear(hidden_dim, output_dim, bias=False), None, None),
            ]
        )


class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam.
    "The projection MLP (in f) has BN applied to each fully-connected (fc)
    layer, including its output fc. Its output fc has no ReLU. The hidden fc is
    2048-d. This MLP has 3 layers." https://arxiv.org/abs/2011.10566.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim, affine=False),
                    torch.nn.ReLU(inplace=True),
                ),
                (
                    torch.nn.Linear(hidden_dim, output_dim, bias=False),
                    torch.nn.BatchNorm1d(output_dim, affine=False),
                    None,
                ),
            ]
        )


class SimSiamPredictionHead(ProjectionHead):
    """Prediction head used for SimSiam.
    "The prediction MLP (h) has BN applied to its hidden fc layers. Its output
    fc does not have BN (...) or ReLU. This MLP has 2 layers." https://arxiv.org/abs/2011.10566.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (torch.nn.Linear(hidden_dim, output_dim, bias=False), None, None),
            ]
        )


class BYOLPredictionHead(ProjectionHead):
    """Prediction head used for BYOL."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (torch.nn.Linear(hidden_dim, output_dim, bias=False), None, None),
            ]
        )


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__(
            [
                (
                    torch.nn.Linear(input_dim, hidden_dim, bias=False),
                    torch.nn.BatchNorm1d(hidden_dim),
                    torch.nn.ReLU(inplace=True),
                ),
                (torch.nn.Linear(hidden_dim, output_dim, bias=False), None, None),
            ]
        )


class DinoProjectionHead(nn.Module):
    """Projection head used for Dino. This projection head does not have
    a batch norm layer.

    Args:
        input_dim: Input dimension for MLP head.
        output_dim: Output dimension (projection dimension) for MLP head.
        hidden_dim: Hidden dimension. Defaults to 512.
        bottleneck_dim: Bottleneck dimension. Defaults to 256.
        num_layers: Number of hidden layers used in MLP. Defaults to 3.
        norm_last_layer: Decides applying normalization before last layer.
            Defaults to False.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        num_layers: int = 3,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        num_layers = max(num_layers, 1)
        self.mlp: nn.Linear | nn.Sequential
        if num_layers == 1:
            self.mlp = nn.Linear(input_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        """Initialize the weights of the projection head."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropModel(nn.Module):
    """MultiCrop model for DINO augmentation.

    It takes 2 global crops and N (possible) local crops as a single tensor.

    Args:
        backbone: Backbone model.
        head: Head model.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        n_crops = len(x)
        # (n_samples * n_crops, 3, size, size)
        concatenated = torch.cat(x, dim=0)
        # (n_samples * n_crops, in_dim)
        embedding = self.backbone(concatenated)
        logits = self.head(embedding)  # (n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops)  # n_crops * (n_samples, out_dim)

        return chunks
