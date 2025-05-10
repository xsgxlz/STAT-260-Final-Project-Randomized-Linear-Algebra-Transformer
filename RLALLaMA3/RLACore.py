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

def broadcast_and_sample_tensors(A: torch.Tensor, B: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slices tensors A and B based on the index tensor idx, supporting PyTorch-style
    broadcasting for their batch dimensions.

    The core operations, after broadcasting A, B, and idx to a common batch shape, are:
    For A (original shape e.g., (*batch_dims_A, dim1_A, dim2_A)):
      The effective A used is A_eff (shape *common_batch_shape, dim1_A, dim2_A).
      The effective idx used is idx_eff (shape *common_batch_shape, num_indices).
      sampled_A[*common_batch_shape, i, k] = A_eff[*common_batch_shape, i, idx_eff[*common_batch_shape, k]]
      where i is an index for dim1_A, k is an index for num_indices. The value from idx_eff selects along dim2_A.

    For B (original shape e.g., (*batch_dims_B, dim1_B, dim2_B)):
      The effective B used is B_eff (shape *common_batch_shape, dim1_B, dim2_B).
      sampled_B[*common_batch_shape, k, j] = B_eff[*common_batch_shape, idx_eff[*common_batch_shape, k], j]
      where k is an index for num_indices, j is an index for dim2_B. The value from idx_eff selects along dim1_B.

    Args:
        A (torch.Tensor): The first tensor. Must have at least 2 dimensions.
                          Shape: e.g., (*batch_dims_A, dim1_A, dim2_A).
                          Example values for (dim1_A, dim2_A): (4, 5).
        B (torch.Tensor): The second tensor. Must have at least 2 dimensions.
                          Shape: e.g., (*batch_dims_B, dim1_B, dim2_B).
                          Example values for (dim1_B, dim2_B): (5, 4).
        idx (torch.Tensor): The index tensor. Must have at least 2 dimensions.
                            Shape: e.g., (*batch_dims_idx, num_indices).
                            Example value for num_indices: 2.
                            Indices must be valid for the dimensions being gathered:
                            - For A: indices for dim2_A (original A.size(-1)).
                            - For B: indices for dim1_B (original B.size(-2)).

    Returns:
        tuple: A tuple containing:
            - sampled_A (torch.Tensor): The sliced tensor A. Shape:
                                        (*common_batch_shape, dim1_A, num_indices).
            - sampled_B (torch.Tensor): The sliced tensor B. Shape:
                                        (*common_batch_shape, num_indices, dim2_B).
    Raises:
        ValueError: If tensors don't have minimum required dimensions or if
                    batch dimensions are not broadcastable.
    """
    # Validate minimum dimensions
    if not (A.ndim >= 2 and B.ndim >= 2 and idx.ndim >= 2):
        raise ValueError(
            f"A (ndim {A.ndim}), B (ndim {B.ndim}), and idx (ndim {idx.ndim}) "
            "must all have at least 2 dimensions."
        )

    # 1. Determine batch dimensions for each tensor
    # If ndim=2, batch_dims is empty, e.g. A=(4,5) -> A.shape[:-2] = torch.Size([])
    A_batch_dims = A.shape[:-2]
    B_batch_dims = B.shape[:-2]
    idx_batch_dims = idx.shape[:-1] # idx is (*batch_dims, num_indices)

    # 2. Compute the final common broadcasted batch shape
    try:
        # This function takes multiple shapes and finds their broadcasted result
        common_batch_shape = torch.broadcast_shapes(A_batch_dims, B_batch_dims, idx_batch_dims)
    except RuntimeError as e:
        raise ValueError(
            f"Batch dimensions of A {A_batch_dims}, B {B_batch_dims}, "
            f"and idx {idx_batch_dims} are not broadcastable. PyTorch error: {e}"
        )

    # 3. Expand A, B, and idx to effectively have this common_batch_shape
    # These are the "data" dimensions, not affected by batch broadcasting.
    A_data_dims = A.shape[-2:] # (dim1_A, dim2_A) from original A
    B_data_dims = B.shape[-2:] # (dim1_B, dim2_B) from original B
    idx_data_dim = idx.shape[-1:]  # (num_indices,) from original idx
    
    A_eff_shape = common_batch_shape + A_data_dims
    A_eff = A.expand(*A_eff_shape)

    B_eff_shape = common_batch_shape + B_data_dims
    B_eff = B.expand(*B_eff_shape)
    
    idx_eff_shape = common_batch_shape + idx_data_dim
    idx_eff = idx.expand(*idx_eff_shape)

    # --- Prepare sampled_A ---
    # Output shape for sampled_A will be: common_batch_shape + (dim1_A, num_indices)
    
    # Unsqueeze idx_eff to prepare for expansion for A.
    # Current idx_eff shape: (*common_batch_shape, num_indices)
    # After unsqueeze: (*common_batch_shape, 1, num_indices)
    idx_A_unsqueezed = idx_eff.unsqueeze(-2)
    
    # Target shape for idx_A_expanded (this is also the shape of sampled_A):
    # (*common_batch_shape, dim1_A, num_indices)
    # A_eff.size(-2) is dim1_A. idx_eff.size(-1) is num_indices.
    shape_for_idx_A_expansion = common_batch_shape + (A_eff.size(-2), idx_eff.size(-1))
    idx_A_expanded = idx_A_unsqueezed.expand(*shape_for_idx_A_expansion)
    
    # Gather along the last dimension of A_eff (which corresponds to original dim2_A)
    gather_dim_A = A_eff.ndim - 1 
    sampled_A = torch.gather(A_eff, gather_dim_A, idx_A_expanded)

    # --- Prepare sampled_B ---
    # Output shape for sampled_B will be: common_batch_shape + (num_indices, dim2_B)

    # Unsqueeze idx_eff to prepare for expansion for B.
    # Current idx_eff shape: (*common_batch_shape, num_indices)
    # After unsqueeze: (*common_batch_shape, num_indices, 1)
    idx_B_unsqueezed = idx_eff.unsqueeze(-1)

    # Target shape for idx_B_expanded (this is also the shape of sampled_B):
    # (*common_batch_shape, num_indices, dim2_B)
    # idx_eff.size(-1) is num_indices. B_eff.size(-1) is dim2_B.
    shape_for_idx_B_expansion = common_batch_shape + (idx_eff.size(-1), B_eff.size(-1))
    idx_B_expanded = idx_B_unsqueezed.expand(*shape_for_idx_B_expansion)

    # Gather along the second to last dim of B_eff (which corresponds to original dim1_B)
    gather_dim_B = B_eff.ndim - 2 
    sampled_B = torch.gather(B_eff, gather_dim_B, idx_B_expanded)
    
    return sampled_A, sampled_B

def sample_and_project_mm(A, B, sample_without_dim, projection_dim, projection_mode='rademacher'):
    if A.ndim == 2:
        A = A.unsqueeze(0)
    if sample_without_dim > A.size(-1):
        raise ValueError(f"sample_without_dim ({sample_without_dim}) must be less than or equal to A.size(-1) ({A.size(-1)})")
    if A.shape[-1] != B.shape[-2]:
        raise ValueError(f"A.shape[-1] ({A.shape[-1]}) must match B.shape[-2] ({B.shape[-2]})")
    
    device = A.device
    
    random_tensor = torch.rand(*A.shape[:-2], A.size(-1), device=device)
    permutations = torch.argsort(random_tensor, dim=-1, descending=False)
    sample_part = permutations[..., :sample_without_dim]
    projection_part = permutations[..., sample_without_dim:]
    sampled_A, sampled_B = broadcast_and_sample_tensors(A, B, sample_part)
    A_to_be_projected, B_to_be_projected = broadcast_and_sample_tensors(A, B, projection_part)
    if projection_dim != 0:
        projection_mm = projection_sketch_mm(A_to_be_projected, B_to_be_projected, projection_dim, mode=projection_mode)
        return sampled_A @ sampled_B + projection_mm
    else:
        return sampled_A @ sampled_B

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
    sample_exact_dim_qk: int = 0, # Number of exact dimensions for QK^T
    sketch_size_qk: int = 0,     # Sketch size for QK^T projection part
    sample_exact_dim_sv: int = 0, # Number of exact dimensions for Scores@V
    sketch_size_sv: int = 0,     # Sketch size for Scores@V projection part
    sketch_mode: str = 'rademacher'
) -> torch.Tensor:
    """
    Computes scaled dot product attention, optionally using Randomized Linear Algebra
    (sampling + projection) for QK^T and Scores@V multiplications.
    Handles GQA internally via n_rep.
    """
    L, S = query.size(-2), key.size(-2)
    E_q = query.size(-1) # Dimension of query features, also common dim for QK^T
    # E_k = key.size(-1) # Usually E_k = E_q
    # E_v = value.size(-1) # Dimension of value features

    # Ensure correct batch and head dimensions
    query_orig_ndim = query.ndim
    key_orig_ndim = key.ndim
    value_orig_ndim = value.ndim

    if query.ndim < 3: query = query.view(1, *query.shape) # Add batch if 2D
    if key.ndim < 3: key = key.view(1, *key.shape)
    if value.ndim < 3: value = value.view(1, *value.shape)

    if query.ndim < 4: query = query.unsqueeze(1) # Add head if 3D (batch, seq, feat)
    if key.ndim < 4: key = key.unsqueeze(1)
    if value.ndim < 4: value = value.unsqueeze(1)

    *batch_dims_q, num_heads_q, _, _ = query.shape
    *batch_dims_k, num_heads_kv, _, _ = key.shape
    # *batch_dims_v, num_heads_v_val, _, _ = value.shape # num_heads_v_val should match num_heads_kv or num_heads_q after GQA

    scale_factor = 1.0 / math.sqrt(E_q) if scale is None else scale

    # --- Grouped Query Attention (GQA) Handling ---
    if n_rep > 1:
        if num_heads_q != num_heads_kv * n_rep:
             raise ValueError(f"GQA: n_heads_q ({num_heads_q}) must be n_heads_kv ({num_heads_kv}) * n_rep ({n_rep})")
        # key: (..., num_heads_kv, S, E_k) -> (..., num_heads_q, S, E_k)
        key = key.repeat_interleave(n_rep, dim=-3)
        # value: (..., num_heads_kv, S, E_v) -> (..., num_heads_q, S, E_v)
        value = value.repeat_interleave(n_rep, dim=-3)
    
    # After GQA, num_heads_kv is effectively num_heads_q for key and value
    num_heads_eff = num_heads_q


    # --- Attention Mask / Bias ---
    # Determine target shape for bias broadcasting
    # Use torch.broadcast_shapes for robust batch dim handling
    common_batch_dims = torch.broadcast_shapes(query.shape[:-3], key.shape[:-3]) # Ignores last 3 dims (heads, seq, feat)
    target_bias_shape = (*common_batch_dims, num_heads_eff, L, S)

    attn_bias = torch.zeros(target_bias_shape, dtype=query.dtype, device=query.device)

    if is_causal:
        if attn_mask is not None:
             raise ValueError("Cannot specify both attn_mask and is_causal=True")
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        causal_mask_shape = (1,) * (len(target_bias_shape) - 2) + (L,S) # for heads, L, S
        temp_mask_expanded = temp_mask.view(causal_mask_shape) # No need to expand to full batch, broadcasting handles it
        attn_bias.masked_fill_(temp_mask_expanded.logical_not(), float("-inf"))


    if attn_mask is not None:
        # Ensure attn_mask is broadcastable to attn_bias.shape
        # We need to align batch_dims, head_dim, L_dim, S_dim
        # attn_mask could be (L,S), (H,L,S), (B,H,L,S), etc.
        mask_s = attn_mask.shape
        bias_s = attn_bias.shape
        
        # Prepend 1s to attn_mask's shape until its ndim matches attn_bias's ndim
        expanded_mask_shape = (1,) * (attn_bias.ndim - attn_mask.ndim) + attn_mask.shape
        attn_mask_expanded = attn_mask.view(expanded_mask_shape)

        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask_expanded.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask_expanded + attn_bias


    # --- Attention Weight Calculation (QK^T) ---
    # Q (..., H, L, E_q), K.T (..., H, E_q, S) -> attn_weight (..., H, L, S)
    # Common dimension for QK^T is E_q (query.size(-1))
    use_rla_qk = not deterministic and (sample_exact_dim_qk < query.size(-1) or sketch_size_qk > 0)
    
    if use_rla_qk:
        attn_weight = sample_and_project_mm(
            query, # (..., H, L, E_q)
            key.transpose(-2, -1), # (..., H, E_q, S)
            sample_without_dim=sample_exact_dim_qk,
            projection_dim=sketch_size_qk,
            projection_mode=sketch_mode
        ) * scale_factor
    else:
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1) # attn_weight shape: (..., H, L, S)

    # --- Weighted Value Calculation (Scores @ V) ---
    # attn_weight (..., H, L, S), value (..., H, S, E_v) -> output (..., H, L, E_v)
    # Common dimension for Scores@V is S (key.size(-2) or value.size(-2))
    use_rla_sv = not deterministic and (sample_exact_dim_sv < attn_weight.size(-1) or sketch_size_sv > 0)

    if use_rla_sv:
        output = sample_and_project_mm(
            attn_weight, # (..., H, L, S)
            value,       # (..., H, S, E_v)
            sample_without_dim=sample_exact_dim_sv,
            projection_dim=sketch_size_sv,
            projection_mode=sketch_mode
        )
    else:
        output = torch.matmul(attn_weight, value)

    # Restore original ndim if changed
    if query_orig_ndim < 4: output = output.squeeze(1) # Remove head if added
    if query_orig_ndim < 3: output = output.squeeze(0) # Remove batch if added

    return output