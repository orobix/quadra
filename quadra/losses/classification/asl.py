import torch


class AsymmetricLoss(torch.nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations.

    Args:
        gamma_neg: gamma for negative samples
        gamma_pos: gamma for positive samples
        m: bias value added to negative samples
        eps: epsilon to avoid division by zero
        disable_torch_grad_focal_loss: if True, disables torch grad for focal loss
        apply_sigmoid: if True, applies sigmoid to input before computing loss
    """

    def __init__(
        self,
        gamma_neg: float = 4,
        gamma_pos: float = 0,
        m: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False,
        apply_sigmoid: bool = True,
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.m = m
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.apply_sigmoid = apply_sigmoid

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets: torch.Tensor
        self.anti_targets: torch.Tensor
        self.xs_pos: torch.Tensor
        self.xs_neg: torch.Tensor
        self.asymmetric_w: torch.Tensor
        self.loss: torch.Tensor

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the asymmetric loss.

        Args:
            x: input logits (after sigmoid)
            y: targets (multi-label binarized vector)

        Returns:
            asymettric loss
        """
        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = x
        if self.apply_sigmoid:
            self.xs_pos = torch.sigmoid(self.xs_pos)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric clipping
        if self.m is not None and self.m > 0:
            self.xs_neg.add_(self.m).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
