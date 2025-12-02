import torch

import extension_cpp  # noqa: F401

def muladd(a: torch.Tensor, b: torch.Tensor, c: float) -> torch.Tensor:
    return torch.ops.extension_cpp.muladd_cpp(a, b, c)
