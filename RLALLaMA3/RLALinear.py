import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Assuming RLACore.py contains sample_and_project_mm and projection_sketch_mm
from .RLACore import sample_and_project_mm # Import the new function

class RLALinear(nn.Module):
    r"""Applies a linear transformation using either standard matrix multiplication
    or a combination of exact sampling and randomized projection.

    When deterministic=False, approximates y = x @ W.T + b using sample_and_project_mm.
    When deterministic=True, computes y = x @ W.T + b exactly using F.linear.

    Initialization is identical to torch.nn.Linear.

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
        sample_exact_dim: The number of dimensions from in_features to be computed
                          exactly via sampling. Used when deterministic=False. Default: 0.
        projection_dim: The intermediate dimension for the random projection sketch
                        for the remaining (in_features - sample_exact_dim) dimensions.
                        Used when deterministic=False. Default: 0.
        projection_mode: The type of random projection ('rademacher' or 'gaussian').
                         Default: 'rademacher'.
        deterministic: If ``True``, performs standard deterministic matrix multiplication.
                       Default: ``False``.
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
        sample_exact_dim: Number of exact dimensions sampled.
        projection_dim: The sketch dimension used for randomized projection.
        projection_mode: The mode for random projection ('rademacher' or 'gaussian').
        deterministic: Flag to switch between deterministic and randomized computation.
    """
    __constants__ = [
        "in_features",
        "out_features",
        "sample_exact_dim",
        "projection_dim",
        "projection_mode",
        "deterministic",
    ]
    in_features: int
    out_features: int
    sample_exact_dim: int
    projection_dim: int
    projection_mode: str
    deterministic: bool
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sample_exact_dim: int = 0,
        projection_dim: int = 0,
        projection_mode: str = 'rademacher',
        deterministic: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deterministic = deterministic
        self.projection_mode = projection_mode

        if not deterministic:
            if sample_exact_dim < 0 or sample_exact_dim > in_features:
                raise ValueError(
                    f"sample_exact_dim ({sample_exact_dim}) must be between 0 and in_features ({in_features})"
                )
            if projection_dim < 0:
                raise ValueError("projection_dim must be non-negative.")
            if sample_exact_dim == in_features and projection_dim > 0:
                # If all dimensions are sampled exactly, projection_dim for the remainder is moot.
                # We could warn or silently set projection_dim to 0.
                # For now, let's allow it, sample_and_project_mm should handle it.
                pass
            if sample_exact_dim < in_features and projection_dim == 0:
                # This means some dimensions will be dropped if not deterministic
                # This might be intentional for some forms of feature selection/dropping
                pass
            if sample_exact_dim == 0 and projection_dim == 0 and in_features > 0:
                # This means if not deterministic, the output will be zero before bias.
                # Could be a user error or intentional.
                # This case is problematic as sample_and_project_mm might not behave as expected
                # if both sample_exact_dim and (in_features - sample_exact_dim) with projection_dim are zero.
                # The original projection_sketch_mm required sketch_size > 0.
                # sample_and_project_mm will handle if dim_to_project > 0 and projection_dim == 0 (drops).
                # If in_features > 0 and sample_exact_dim == 0 and projection_dim == 0,
                # it means all features are meant for projection, but projection_dim is 0, effectively dropping all.
                pass


        self.sample_exact_dim = sample_exact_dim
        self.projection_dim = projection_dim


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
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.deterministic or \
           (self.sample_exact_dim == self.in_features and self.projection_dim == 0):
            # Perform exact multiplication if deterministic OR
            # if all features are sampled exactly and no projection is applied to a zero-size remainder.
            return F.linear(input, self.weight, self.bias)
        else:
            # Use randomized matrix multiplication (sampling + optional projection)
            # F.linear computes input @ weight.T + bias
            # So we need sample_and_project_mm(input, weight.T, ...)
            output_no_bias = sample_and_project_mm(
                A=input,                 # Shape: (..., in_features)
                B=self.weight.T,         # Shape: (in_features, out_features)
                sample_exact_dim=self.sample_exact_dim, # along in_features
                projection_dim=self.projection_dim,     # for the remainder of in_features
                projection_mode=self.projection_mode
            )
            if self.bias is not None:
                output_no_bias = output_no_bias + self.bias
            return output_no_bias

    def deterministic_mode(self, enable: bool = True) -> "RLALinear":
        self.deterministic = enable
        return self

    def extra_repr(self) -> str:
        repr_str = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        if not self.deterministic:
            repr_str += f", sample_exact_dim={self.sample_exact_dim}"
            # Only show projection_dim if there are dimensions left to project
            if self.in_features - self.sample_exact_dim > 0:
                 repr_str += f", projection_dim={self.projection_dim}"
            repr_str += f", projection_mode='{self.projection_mode}'"
        repr_str += f", deterministic={self.deterministic}"
        return repr_str