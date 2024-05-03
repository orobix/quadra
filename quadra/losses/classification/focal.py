from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import nn


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
            Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
        eps: a value added to the returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError(f"The number of classes must be bigger than one. Got: {num_classes}")

    shape = labels.shape
    one_hot_output = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot_output.scatter_(1, labels.unsqueeze(1), 1.0) + eps


# based on: # https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float | None = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input_tensor: Logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: Labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be divided by
            the number of elements in the output, ``'sum'``: the output will be
            summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

    Returns:
        The computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input_tensor)}")

    if not len(input_tensor.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input_tensor.shape}")

    if input_tensor.size(0) != target.size(0):
        raise ValueError(
            f"Expected input batch_size ({input_tensor.size(0)}) to match target batch_size ({target.size(0)})."
        )

    n = input_tensor.size(0)
    out_size = (n,) + input_tensor.size()[2:]
    if target.size()[1:] != input_tensor.size()[2:]:
        raise ValueError(f"Expected target size {out_size}, got {target.size()}")

    if not input_tensor.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input_tensor.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input_tensor, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input_tensor, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input_tensor.shape[1], device=input_tensor.device, dtype=input_tensor.dtype
    )

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum("bc...,bc...->b...", (target_one_hot, focal))

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be divided by
            the number of elements in the output, ``'sum'``: the output will be
            summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
            used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = "none", eps: float | None = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float | None = eps

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward call computation."""
        return focal_loss(input_tensor, target, self.alpha, self.gamma, self.reduction, self.eps)


def binary_focal_loss_with_logits(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float | None = None,
) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input_tensor: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be divided by
            the number of elements in the output, ``'sum'``: the output will be
            summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.

    Returns:
        the computed loss.

    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input_tensor)}")

    if not len(input_tensor.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input_tensor.shape}")

    if input_tensor.size(0) != target.size(0):
        raise ValueError(
            f"Expected input batch_size ({input_tensor.size(0)}) to match target batch_size ({target.size(0)})."
        )

    probs_pos = torch.sigmoid(input_tensor)
    probs_neg = torch.sigmoid(-input_tensor)
    loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(input_tensor) - (1 - alpha) * torch.pow(
        probs_pos, gamma
    ) * (1.0 - target) * F.logsigmoid(-input_tensor)

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be divided by
            the number of elements in the output, ``'sum'``: the output will be
            summed.

    Shape:
        - Input: :math:`(N, *)`.
        - Target: :math:`(N, *)`.

    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = "none") -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward call computation."""
        return binary_focal_loss_with_logits(input_tensor, target, self.alpha, self.gamma, self.reduction)
