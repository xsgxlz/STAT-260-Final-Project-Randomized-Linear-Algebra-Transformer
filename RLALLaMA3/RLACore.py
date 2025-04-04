import torch
import math
from typing import Optional

def projection_sketch_mm(A, B, sketch_size, mode='rademacher'):
    """
    Approximates the matrix product A @ B using a projection sketch.
    Handles batched inputs (any number of leading dimensions).

    Args:
        A: The first matrix (or batch of matrices) (..., m, d).
        B: The second matrix (or batch of matrices) (..., d, n).
        sketch_size: The size of the sketch (inner dimension k).
        mode: Type of random projection ('rademacher' or 'gaussian').

    Returns:
        The approximated matrix product (..., m, n).
    """
    # Ensure inputs are tensors
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("Inputs A and B must be torch tensors.")

    # Basic dimension check for the matrix multiply dimensions
    if A.shape[-1] != B.shape[-2]:
         raise RuntimeError(
             f"Matrix A last dim {A.shape[-1]} doesn't match B second-to-last dim {B.shape[-2]}"
         )

    # Determine shape for S: A(..., m, d), B(..., d, n) -> S(..., d, k)
    S_shape = (*A.shape[:-1], sketch_size) # (*batch_A, m, d) -> (*batch_A, m, k) is wrong!
                                           # S should project the common dim 'd'
                                           # Shape should be independent of 'm'
    S_shape = (*A.shape[:-2], A.shape[-1], sketch_size) # (*batch_A, d, k)
    device = A.device
    dtype = A.dtype

    if mode == 'rademacher':
        S = (torch.randint(0, 2, S_shape, device=device, dtype=dtype) * 2 - 1)
    elif mode == 'gaussian':
        S = torch.randn(S_shape, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Scaling factor for unbiased estimate E[S @ S.T / sketch_size] = I_d
    scaling_factor = sketch_size

    # Calculate (A @ S / sketch_size) @ (S.T @ B)
    # Note: torch.matmul handles batching and broadcasting
    AS = torch.matmul(A, S) / scaling_factor
    SB = torch.matmul(S.transpose(-1, -2), B)

    # Check intermediate dimension for final matmul
    if AS.shape[-1] != SB.shape[-2]:
         raise RuntimeError(f"Intermediate matrix AS last dim {AS.shape[-1]} doesn't match SB second-to-last dim {SB.shape[-2]}")

    AB_bar = torch.matmul(AS, SB)
    return AB_bar

def rla_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    n_rep: int = 1, # Number of times to repeat K, V for GQA
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    # RLA Specific Args
    deterministic: bool = False, # If True, overrides sketch sizes and uses matmul
    sketch_size_qk: int = 0,     # Sketch size for QK^T, 0 means deterministic matmul
    sketch_size_sv: int = 0,     # Sketch size for Scores@V, 0 means deterministic matmul
    sketch_mode: str = 'rademacher'
) -> torch.Tensor:
    """
    Computes scaled dot product attention, optionally using Randomized Linear Algebra
    for QK^T and Scores@V multiplications. Handles GQA internally via n_rep.
    """
    L, S = query.size(-2), key.size(-2)
    E_q, E_k, E_v = query.size(-1), key.size(-1), value.size(-1)
    *batch_q, num_heads_q, _, _ = query.shape if query.ndim >=4 else (*([1]* (4-query.ndim)), *query.shape) # Handle cases < 4D
    *batch_k, num_heads_kv, _, _ = key.shape if key.ndim >=4 else (*([1]* (4-key.ndim)), *key.shape)

    scale_factor = 1.0 / math.sqrt(E_q) if scale is None else scale

    # --- Grouped Query Attention (GQA) Handling ---
    if n_rep > 1:
        if num_heads_q != num_heads_kv * n_rep:
             raise ValueError(f"GQA: n_heads_q ({num_heads_q}) must be n_heads_kv ({num_heads_kv}) * n_rep ({n_rep})")
        key = key.repeat_interleave(n_rep, dim=-3)
        value = value.repeat_interleave(n_rep, dim=-3)

    # --- Attention Mask / Bias ---
    # Determine target shape for bias broadcasting
    target_bias_shape = (*batch_q, num_heads_q, L, S)
    attn_bias = torch.zeros(target_bias_shape, dtype=query.dtype, device=query.device)

    if is_causal:
        if attn_mask is not None:
             raise ValueError("Cannot specify both attn_mask and is_causal=True")
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        # Expand mask to match target bias shape for broadcasting
        causal_mask_shape = (1,) * (len(target_bias_shape) - 2) + (L,S)
        temp_mask = temp_mask.view(causal_mask_shape)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        # Ensure attn_mask is broadcastable to attn_bias shape
        if attn_mask.dtype == torch.bool:
            # Expand mask shape if necessary for broadcasting
            while attn_mask.ndim < attn_bias.ndim:
                 attn_mask = attn_mask.unsqueeze(0)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
             # Expand mask shape if necessary for broadcasting
            while attn_mask.ndim < attn_bias.ndim:
                 attn_mask = attn_mask.unsqueeze(0)
            attn_bias = attn_mask + attn_bias

    # --- Attention Weight Calculation (QK^T) ---
    use_rla_qk = not deterministic and sketch_size_qk > 0
    if use_rla_qk:
        attn_weight = projection_sketch_mm(
            query, key.transpose(-2, -1), sketch_size_qk, mode=sketch_mode
        ) * scale_factor
    else:
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # --- Weighted Value Calculation (Scores @ V) ---
    use_rla_sv = not deterministic and sketch_size_sv > 0
    if use_rla_sv:
        output = projection_sketch_mm(
            attn_weight, value, sketch_size_sv, mode=sketch_mode
        )
    else:
        output = torch.matmul(attn_weight, value)

    return output