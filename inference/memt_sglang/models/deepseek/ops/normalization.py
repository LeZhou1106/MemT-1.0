"""L2 normalization used by DeepSeek's compiled Q/K preparation."""

import torch


def l2norm(inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the last dimension with the configured epsilon."""
    return inputs * torch.rsqrt(inputs.square().sum(-1, keepdim=True) + eps)
