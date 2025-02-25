from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
import timm
import torch
import torch.nn.functional as F
import tqdm
from pytorch_grad_cam import GradCAM
from scipy import ndimage
from sklearn.linear_model._base import ClassifierMixin
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch import nn

from quadra.models.evaluation import (
    BaseEvaluationModel,
    ONNXEvaluationModel,
    TorchEvaluationModel,
    TorchscriptEvaluationModel,
)
from quadra.utils import utils
from quadra.utils.vit_explainability import VitAttentionGradRollout

log = utils.get_logger(__name__)


def net_hat(input_size: int, output_size: int) -> torch.nn.Sequential:
    """Create a linear layer with input and output neurons.

    Args:
        input_size: Number of input neurons
        output_size: Number of output neurons.

    Returns:
        A sequential containing a single Linear layer taking input neurons and producing output neurons

    """
    return torch.nn.Sequential(torch.nn.Linear(input_size, output_size))


def create_net_hat(dims: list[int], act_fun: Callable = torch.nn.ReLU, dropout_p: float = 0) -> torch.nn.Sequential:
    """Create a sequence of linear layers with activation functions and dropout.

    Args:
        dims: Dimension of hidden layers and output
        act_fun: activation function to use between layers, default ReLU
        dropout_p: Dropout probability. Defaults to 0.

    Returns:
        Sequence of linear layers of dimension specified by the input, each linear layer is followed
            by an activation function and optionally a dropout layer with the input probability
    """
    components: list[nn.Module] = []
    for i, _ in enumerate(dims[:-2]):
        if dropout_p > 0:
            components.append(torch.nn.Dropout(dropout_p))
        components.append(net_hat(dims[i], dims[i + 1]))
        components.append(act_fun())
    components.append(net_hat(dims[-2], dims[-1]))
    components.append(L2Norm())
    return torch.nn.Sequential(*components)


class L2Norm(torch.nn.Module):
    """Compute L2 Norm."""

    def forward(self, x: torch.Tensor):
        return x / torch.norm(x, p=2, dim=1, keepdim=True)


def init_weights(m):
    """Basic weight initialization."""
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def get_feature(
    feature_extractor: torch.nn.Module | BaseEvaluationModel,
    dl: torch.utils.data.DataLoader,
    iteration_over_training: int = 1,
    gradcam: bool = False,
    classifier: ClassifierMixin | None = None,
    input_shape: tuple[int, int, int] | None = None,
    limit_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Given a dataloader and a PyTorch model, extract features with the model and return features and labels.

    Args:
        dl: PyTorch dataloader
        feature_extractor: Pretrained PyTorch backbone
        iteration_over_training: Extract feature iteration_over_training times for each image
            (best if used with augmentation)
        gradcam: Whether to compute gradcams. Notice that it will slow the function
        classifier: Scikit-learn classifier
        input_shape: [H,W,C], backbone input shape, needed by classifier's pytorch wrapper
        limit_batches: Limit the number of batches to be processed

    Returns:
        Tuple containing:
            features: Model features
            labels: input_labels
            grayscale_cams: Gradcam output maps, None if gradcam arg is False
    """
    if isinstance(feature_extractor, TorchEvaluationModel | TorchscriptEvaluationModel):
        # If we are working with torch based evaluation models we need to extract the model
        feature_extractor = feature_extractor.model
    elif isinstance(feature_extractor, ONNXEvaluationModel):
        gradcam = False

    feature_extractor.eval()

    # Setup gradcam
    if gradcam:
        if not hasattr(feature_extractor, "features_extractor"):
            gradcam = False
        elif isinstance(feature_extractor.features_extractor, timm.models.resnet.ResNet):
            target_layers = [feature_extractor.features_extractor.layer4[-1]]
            cam = GradCAM(
                model=feature_extractor,
                target_layers=target_layers,
            )
            for p in feature_extractor.features_extractor.layer4[-1].parameters():
                p.requires_grad = True
        elif is_vision_transformer(feature_extractor.features_extractor):
            grad_rollout = VitAttentionGradRollout(
                feature_extractor.features_extractor,
                classifier=classifier,
                example_input=None if input_shape is None else torch.randn(1, *input_shape),
            )
        else:
            gradcam = False

        if not gradcam:
            log.warning("Gradcam not implemented for this backbone, it will not be computed")

    # Extract features from data

    for iteration in range(iteration_over_training):
        for i, b in enumerate(tqdm.tqdm(dl)):
            x1, y1 = b

            if hasattr(feature_extractor, "parameters"):
                # Move input to the correct device and dtype
                parameter = next(feature_extractor.parameters())
                x1 = x1.to(parameter.device).to(parameter.dtype)
            elif isinstance(feature_extractor, BaseEvaluationModel):
                x1 = x1.to(feature_extractor.device).to(feature_extractor.model_dtype)

            if gradcam:
                y_hat = cast(list[torch.Tensor] | tuple[torch.Tensor] | torch.Tensor, feature_extractor(x1).detach())
                # mypy can't detect that gradcam is true only if we have a features_extractor
                if is_vision_transformer(feature_extractor.features_extractor):  # type: ignore[union-attr]
                    grayscale_cam_low_res = grad_rollout(
                        input_tensor=x1, targets_list=y1
                    )  # TODO: We are using labels (y1) but it would be better to use preds
                    orig_shape = grayscale_cam_low_res.shape
                    new_shape = (orig_shape[0], x1.shape[2], x1.shape[3])
                    zoom_factors = tuple(np.array(new_shape) / np.array(orig_shape))
                    grayscale_cam = ndimage.zoom(grayscale_cam_low_res, zoom_factors, order=1)
                else:
                    grayscale_cam = cam(input_tensor=x1, targets=None)
                feature_extractor.zero_grad(set_to_none=True)  # type: ignore[union-attr]
            else:
                with torch.no_grad():
                    y_hat = cast(list[torch.Tensor] | tuple[torch.Tensor] | torch.Tensor, feature_extractor(x1))
                grayscale_cams = None

            if isinstance(y_hat, list | tuple):
                y_hat = y_hat[0].cpu()
            else:
                y_hat = y_hat.cpu()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i == 0 and iteration == 0:
                features = torch.cat([y_hat], dim=0)
                labels = np.concatenate([y1])
                if gradcam:
                    grayscale_cams = grayscale_cam
            else:
                features = torch.cat([features, y_hat], dim=0)
                labels = np.concatenate([labels, y1], axis=0)
                if gradcam:
                    grayscale_cams = np.concatenate([grayscale_cams, grayscale_cam], axis=0)

            if limit_batches is not None and (i + 1) >= limit_batches:
                break

    return features.detach().numpy(), labels, grayscale_cams


def is_vision_transformer(model: torch.nn.Module) -> bool:
    """Verify if pytorch module is a Vision Transformer.
    This check is primarily needed for gradcam computation in classification tasks.

    Args:
        model: Model
    """
    return type(model).__name__ == "VisionTransformer"


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, a: float, b: float):
    """Cut & paste from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff
        b: the maximum cutoff
    """

    def norm_cdf(x: float):
        """Computes standard normal cumulative distribution function."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            (
                "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect."
            ),
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    """Call `_no_grad_trunc_normal_` with `torch.no_grad()`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff
        b: the maximum cutoff
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def clip_gradients(model: nn.Module, clip: float) -> list[float]:
    """Args:
        model: The model
        clip: The clip value.

    Returns:
        The norms of the gradients
    """
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


# TODO: do not use this implementation for new models


class AttentionExtractor(torch.nn.Module):
    """General attention extractor.

    Args:
        model: Backbone model which contains the attention layer.
            attention_layer_name: Attention layer for extracting attention maps.
            Defaults to "attn_drop".
        attention_layer_name: Attention layer for extracting attention maps.
    """

    def __init__(self, model: torch.nn.Module, attention_layer_name: str = "attn_drop"):
        super().__init__()
        self.model = model
        modules = [module for module_name, module in self.model.named_modules() if attention_layer_name in module_name]
        if modules:
            modules[-1].register_forward_hook(self.get_attention)
        self.attentions = torch.zeros((1, 0))

    def clear(self):
        """Clear the grabbed attentions."""
        self.attentions = torch.zeros((1, 0))

    def get_attention(self, module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor):  # pylint: disable=unused-argument
        """Method to be registered to grab attentions."""
        self.attentions = output.detach().clone().cpu()

    @staticmethod
    def process_attention_maps(attentions: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
        """Preprocess attentions maps to be visualized.

        Args:
            attentions: grabbed attentions
            img_width: image width
            img_height: image height

        Returns:
            torch.Tensor: preprocessed attentions, with the shape equal to the one of the image from
            which attentions has been computed
        """
        if len(attentions.shape) == 4:
            # vit
            # batch, heads, N, N (class atention layer)
            attentions = attentions[:, :, 0, 1:]  # batch, heads, height-1

        else:
            # xcit
            # batch, heads, N
            attentions = attentions[:, :, 1:]  # batch, heads, dim-1
        nh = attentions.shape[1]
        patch_size = int(math.sqrt(img_width * img_height / attentions.shape[-1]))
        w_featmap = img_width // patch_size
        h_featmap = img_height // patch_size

        # we keep only the output patch attention we dont want cls
        attentions = attentions.reshape(attentions.shape[0], nh, w_featmap, h_featmap)
        attentions = F.interpolate(attentions, scale_factor=patch_size, mode="nearest")
        return attentions

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.clear()
        out = self.model(t)
        return (out, self.attentions)  # torch.jit.trace does not complain


# TODO: do not use this implementation for new models


class PositionalEncoding1D(torch.nn.Module):
    """Standard sine-cosine positional encoding from https://arxiv.org/abs/2010.11929.

    Args:
        d_model: Embedding dimension
        temperature: Temperature for the positional encoding. Defaults to 10000.0.
        dropout: Dropout rate. Defaults to 0.0.
        max_len: Maximum length of the sequence. Defaults to 5000.
    """

    def __init__(self, d_model: int, temperature: float = 10000.0, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout: torch.nn.Dropout | torch.nn.Identity
        if dropout > 0:
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = torch.nn.Identity()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(temperature) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.permute(1, 0, 2)
        self.pe = torch.nn.Parameter(self.pe)
        self.pe.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the positional encoding.

        Args:
            x: torch tensor [batch_size, seq_len, embedding_dim].
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LSABlock(torch.nn.Module):
    """Local Self Attention Block from https://arxiv.org/abs/2112.13492.

    Args:
        dim: embedding dimension
        num_heads: number of attention heads
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: enable bias for qkv if True
        drop: dropout rate
        attn_drop: attention dropout rate
        drop_path: stochastic depth rate
        act_layer: activation layer
        norm_layer:: normalization layer
        mask_diagonal: whether to mask Q^T x K diagonal with -infinity so not to
            count self relationship between tokens. Defaults to True
        learnable_temperature: whether to use a learnable temperature as specified in
            https://arxiv.org/abs/2112.13492. Defaults to True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = torch.nn.GELU,
        norm_layer: type[torch.nn.LayerNorm] = torch.nn.LayerNorm,
        mask_diagonal: bool = True,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LocalSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_diagonal=mask_diagonal,
            learnable_temperature=learnable_temperature,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LocalSelfAttention(torch.nn.Module):
    """Local Self Attention from https://arxiv.org/abs/2112.13492.

    Args:
        dim: embedding dimension.
        num_heads: number of attention heads.
        qkv_bias: enable bias for qkv if True.
        attn_drop: attention dropout rate.
        proj_drop: projection dropout rate.
        mask_diagonal: whether to mask Q^T x K diagonal with -infinity
            so not to count self relationship between tokens. Defaults to True.
        learnable_temperature: whether to use a learnable temperature as specified in
            https://arxiv.org/abs/2112.13492. Defaults to True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_diagonal: bool = True,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.mask_diagonal = mask_diagonal
        if learnable_temperature:
            self.register_parameter("scale", torch.nn.Parameter(torch.tensor(head_dim**-0.5, requires_grad=True)))
        else:
            self.scale = head_dim**-0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the local self attention.

        Args:
            x: input tensor

        Returns:
            Output of the local self attention.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.mask_diagonal:
            attn[torch.eye(N, device=attn.device, dtype=torch.bool).repeat(B, self.num_heads, 1, 1)] = -float("inf")
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
