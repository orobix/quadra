from typing import List, Optional

import numpy as np
import torch
from sklearn.linear_model._base import ClassifierMixin


def rollout(attentions: List[torch.Tensor], discard_ratio: float = 0.9, head_fusion: str = "mean") -> np.ndarray:
    """Apply rollout on Attention matrices.

    Args:
        attentions: List of Attention matrices coming from different blocks
        discard_ratio: Percentage of elements to discard
        head_fusion: Strategy of fusion of attention heads
    Returns:
        mask: Output mask, still needs a resize
    """
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError("Attention head fusion type Not supported")
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            flat.scatter_(-1, indices, 0)
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1).unsqueeze(1)
            result = torch.matmul(a, result)
    # Look at the total attention between the class token and the image patches
    mask = result[:, 0, 1:]
    batch_size = mask.size(0)
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(batch_size, width, width).numpy()
    mask = mask / mask.max(axis=(1, 2), keepdims=True)
    return mask


class VitAttentionRollout:
    """Attention gradient rollout class. Constructor registers hooks to the model's specified layers.
    Only 4 layers by default given the high load on gpu. Best gradcams obtained using all blocks.

    Args:
        model: Model
        attention_layer_names: On which layers to register the hook
        head_fusion: Strategy of fusion for attention heads
        discard_ratio: Percentage of elements to discard
    """

    def __init__(
        self,
        model: torch.nn.Module,
        attention_layer_names: Optional[List[str]] = None,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ):
        if attention_layer_names is None:
            attention_layer_names = [
                "blocks.6.attn.attn_drop",
                "blocks.7.attn.attn_drop",
                "blocks.10.attn.attn_drop",
                "blocks.11.attn.attn_drop",
            ]
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.f_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in self.model.named_modules():
            for layer_name in attention_layer_names:
                if layer_name in name:
                    self.f_hook_handles.append(module.register_forward_hook(self.get_attention))
        self.attentions: List[torch.Tensor] = []

    def get_attention(
        self, module: torch.nn.Module, inpt: torch.Tensor, out: torch.Tensor  # pylint: disable=W0613
    ) -> None:
        """Hook to return attention.

        Args:
            module: Torch module
            inpt: Input tensor
            out: Output tensor, in this case the attention
        """
        self.attentions.append(out.detach().clone().cpu())

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Called when the class instance is used as a function.

        Args:
            input_tensor: Input tensor
        Returns:
            out: Batch of output masks
        """
        self.attentions.clear()
        with torch.no_grad():
            self.model(input_tensor)
        out = rollout(self.attentions, self.discard_ratio, self.head_fusion)
        return out


############################################### GRAD ROLLOUT ##########################################################


def grad_rollout(
    attentions: List[torch.Tensor], gradients: List[torch.Tensor], discard_ratio: float = 0.9
) -> np.ndarray:
    """Apply gradient rollout on Attention matrices.

    Args:
        attentions: Attention matrices
        gradients: Target class gradient matrices
        discard_ratio: Percentage of elements to discard
    Returns:
        mask: Output mask, still needs a resize
    """
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = torch.mean((attention * weights), dim=1)
            attention_heads_fused[attention_heads_fused < 0] = 0
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            flat.scatter_(-1, indices, 0)
            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1).unsqueeze(1)
            result = torch.matmul(a, result)
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[:, 0, 1:]
    batch_size = mask.size(0)
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(batch_size, width, width).numpy()
    mask = mask / mask.max(axis=(1, 2), keepdims=True)
    return mask


class VitAttentionGradRollout:
    """Attention gradient rollout class. Constructor registers hooks to the model's specified layers.
    Only 4 layers by default given the high load on gpu. Best gradcams obtained using all blocks.

    Args:
        model: Pytorch model
        attention_layer_names: On which layers to register the hooks
        discard_ratio: Percentage of elements to discard
        classifier: Scikit-learn classifier. Leave it to None if model already has a classifier on top.
    """

    def __init__(  # pylint: disable=W0102
        self,
        model: torch.nn.Module,
        attention_layer_names: Optional[List[str]] = None,
        discard_ratio: float = 0.9,
        classifier: Optional[ClassifierMixin] = None,
    ):
        if attention_layer_names is None:
            attention_layer_names = [
                "blocks.4.attn.attn_drop",
                "blocks.5.attn.attn_drop",
                "blocks.6.attn.attn_drop",
                "blocks.7.attn.attn_drop",
                "blocks.8.attn.attn_drop",
                "blocks.9.attn.attn_drop",
                "blocks.10.attn.attn_drop",
                "blocks.11.attn.attn_drop",
            ]

        if classifier is not None:
            self.model = ClassifierPytorchWrapper(
                backbone=model, lr_classifier=classifier, device=next(model.parameters()).device
            )
        else:
            self.model = model  # type: ignore[assignment]

        self.discard_ratio = discard_ratio
        self.f_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.b_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in self.model.named_modules():
            for layer_name in attention_layer_names:
                if layer_name in name:
                    self.f_hook_handles.append(module.register_forward_hook(self.get_attention))
                    self.b_hook_handles.append(module.register_backward_hook(self.get_attention_gradient))
        self.attentions: List[torch.Tensor] = []
        self.attention_gradients: List[torch.Tensor] = []
        # Activate gradients
        blocks_list = [x.split("blocks")[1].split(".attn")[0] for x in attention_layer_names]
        for name, module in model.named_modules():
            for p in module.parameters():
                if "blocks" in name and any(x in name for x in blocks_list):
                    p.requires_grad = True

    def get_attention(
        self, module: torch.nn.Module, inpt: torch.Tensor, out: torch.Tensor  # pylint: disable=W0613
    ) -> None:
        """Hook to return attention.

        Args:
            module: Torch module
            inpt: Input tensor
            out: Output tensor, in this case the attention
        """
        self.attentions.append(out.detach().clone().cpu())

    def get_attention_gradient(
        self, module: torch.nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor  # pylint: disable=W0613
    ) -> None:
        """Hook to return attention.

        Args:
            module: Torch module
            grad_input: Gradients' input tensor
            grad_output: Gradients' output tensor, in this case the attention
        """
        self.attention_gradients.append(grad_input[0].detach().clone().cpu())

    def __call__(self, input_tensor: torch.Tensor, targets_list: List[int]) -> np.ndarray:
        """Called when the class instance is used as a function.

        Args:
            input_tensor: Model's input tensor
            targets_list: List of targets. If None, argmax is used

        Returns:
            out: Batch of output masks
        """
        self.attentions.clear()
        self.attention_gradients.clear()

        self.model.zero_grad(set_to_none=True)
        self.model.to(input_tensor.device)
        output = self.model(input_tensor).cpu()

        class_mask = torch.zeros(output.shape)
        if targets_list is None:
            targets_list = output.argmax(dim=1)
        class_mask[torch.arange(output.shape[0]), targets_list] = 1
        loss = (output * class_mask).sum()
        loss.backward()
        out = grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)

        return out


class ClassifierPytorchWrapper(torch.nn.Module):
    """Pytorch wrapper for scikit-learn classifiers.

    Args:
        backbone: Backbone
        num_classes: Number of classes
        lr_classifier: ScikitLearn classifier model
        device: The device to use. Defaults to "cpu".
    """

    def __init__(self, backbone: torch.nn.Module, lr_classifier: ClassifierMixin, device: torch.device):
        super().__init__()
        self.device = device
        self.backbone = backbone.to(device)
        self.num_classes = len(lr_classifier.classes_)
        self.lr_classifier = lr_classifier
        self.example_input_array = torch.rand(1, 3, 224, 224).to(device)  # TODO: Should not be hardcoded
        with torch.no_grad():
            output = self.backbone(self.example_input_array)
            num_filters = output.shape[-1]

        self.classifier = torch.nn.Linear(num_filters, self.num_classes).to(device)
        self.classifier.weight.data = torch.from_numpy(lr_classifier.coef_).float()
        self.classifier.bias.data = torch.from_numpy(lr_classifier.intercept_).float()

    def forward(self, x):
        return torch.nn.Softmax(dim=1)(self.classifier(self.backbone(x)))