from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Parameter


class SAM(torch.optim.Optimizer):
    """PyTorch implementation of Sharpness-Aware-Minization paper: https://arxiv.org/abs/2010.01412
    and https://arxiv.org/abs/2102.11600.
    Taken from: https://github.com/davda54/sam.

    Args:
        params: model parameters.
        base_optimizer: optimizer to use.
        rho: Postive float value used to scale the gradients.
        adaptive: Boolean flag indicating whether to use adaptive step update.
        **kwargs: Additional parameters for the base optimizer.
    """

    def __init__(
        self,
        params: list[Parameter],
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = True,
        **kwargs: Any,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = {"rho": rho, "adaptive": adaptive, **kwargs}
        super().__init__(params, defaults)

        if callable(base_optimizer):
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        else:
            self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """First step for SAM optimizer.

        Args:
            zero_grad: Boolean flag indicating whether to zero the gradients.

        Returns:
            None
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Second step for SAM optimizer.

        Args:
            zero_grad: Boolean flag indicating whether to zero the gradients.

        Returns:
            None

        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:  # type: ignore[override]
        """Step for SAM optimizer.

        Args:
            closure: The Optional closure for enable grad.

        Returns:
            None

        """
        if closure is not None:
            closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        if closure is not None:
            closure()
        self.second_step(zero_grad=False)

    def _grad_norm(self) -> torch.Tensor:
        """Put everything on the same device, in case of model parallelism
        Returns:
            Grad norm.
        """
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
