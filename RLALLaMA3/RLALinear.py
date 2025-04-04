import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .RLACore import projection_sketch_mm

class RLALinear(nn.Module):
    r"""Applies a linear transformation using either standard or randomized matrix multiplication.

    When deterministic=False, approximates y = x @ W.T + b using projection_sketch_mm.
    When deterministic=True, computes y = x @ W.T + b exactly using F.linear.

    Initialization is identical to torch.nn.Linear.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        sketch_size: The intermediate dimension for the random projection sketch. Required if deterministic=False.
        sketch_mode: The type of random projection ('rademacher' or 'gaussian'). Default: 'rademacher'.
        deterministic: If ``True``, performs standard deterministic matrix multiplication. Default: ``False``.
        device: Device for tensors.
        dtype: Data type for tensors.

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. Initialized identically to nn.Linear.
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                Initialized identically to nn.Linear if bias=True.
        sketch_size: The sketch dimension used for randomized multiplication.
        sketch_mode: The mode for random projection ('rademacher' or 'gaussian').
        deterministic: Flag to switch between deterministic and randomized computation.
    """
    __constants__ = ["in_features", "out_features", "sketch_size", "sketch_mode", "deterministic"]
    in_features: int
    out_features: int
    sketch_size: int
    sketch_mode: str
    deterministic: bool
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sketch_size: int = 0, # Set default, but raise error if needed
        sketch_mode: str = 'rademacher',
        deterministic: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deterministic = deterministic
        self.sketch_mode = sketch_mode

        if not deterministic and sketch_size <= 0:
            raise ValueError("sketch_size must be positive when deterministic is False")
        self.sketch_size = sketch_size

        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Replicates nn.Linear's initialization
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Calculate fan_in from the weight tensor
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # Calculate the bound based on fan_in
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # Initialize bias with uniform distribution within the calculated bounds
            torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            # Use standard F.linear for deterministic computation
            return F.linear(input, self.weight, self.bias)
        else:
            # Use randomized matrix multiplication
            # F.linear computes input @ weight.T + bias
            # So we need projection_sketch_mm(input, weight.T, ...)
            output_no_bias = projection_sketch_mm(
                input, self.weight.T, self.sketch_size, mode=self.sketch_mode
            )
            if self.bias is not None:
                # Add bias if it exists
                output_no_bias = output_no_bias + self.bias

            return output_no_bias
    
    def deterministic_mode(self, enable: bool = True) -> "RLALinear":
        # Set the deterministic mode
        self.deterministic = enable
        return self

    def extra_repr(self) -> str:
        # Provides a string representation including custom parameters
        repr = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        if not self.deterministic:
            repr += f", sketch_size={self.sketch_size}, sketch_mode='{self.sketch_mode}'"
        repr += f", deterministic={self.deterministic}"
        return repr