import torch
import pytest
from RLACore import rla_scaled_dot_product_attention

def test_rla_sdpa_large_sample_projection_correctness():
    B = 1  # Batch size
    H = 2  # Number of heads
    L = 16 # Query sequence length
    S = 32 # Key/Value sequence length
    E_q = 64 # Query/Key feature dim
    E_v = 128 # Value feature dim

    dtype = torch.float32
    device = 'cpu'

    # RLA parameters
    num_projected_dims_qk = 4
    sample_exact_dim_qk_val = E_q - num_projected_dims_qk
    sketch_size_qk_val = 32

    num_projected_dims_sv = 4
    sample_exact_dim_sv_val = S - num_projected_dims_sv
    sketch_size_sv_val = 16

    # Input tensors
    query = torch.randn(B, H, L, E_q, dtype=dtype, device=device)
    key = torch.randn(B, H, S, E_q, dtype=dtype, device=device)
    value = torch.randn(B, H, S, E_v, dtype=dtype, device=device)

    torch.manual_seed(42)
    output_rla = rla_scaled_dot_product_attention(
        query.clone(), key.clone(), value.clone(),
        n_rep=1,
        attn_mask=None,
        is_causal=False,
        deterministic=False,
        sample_exact_dim_qk=sample_exact_dim_qk_val,
        sketch_size_qk=sketch_size_qk_val,
        sample_exact_dim_sv=sample_exact_dim_sv_val,
        sketch_size_sv=sketch_size_sv_val,
        sketch_mode='rademacher'
    )

    output_pytorch = torch.nn.functional.scaled_dot_product_attention(
        query.clone(), key.clone(), value.clone(),
        attn_mask=None,
        is_causal=False
    )

    # Print comparison metrics
    print(f"RLA output sample (first 5 flat): {output_rla.flatten()[:5]}")
    print(f"Pytorch output sample (first 5 flat): {output_pytorch.flatten()[:5]}")

    # Shape check
    assert output_rla.shape == output_pytorch.shape, \
        f"Shape mismatch: RLA is {output_rla.shape}, PyTorch is {output_pytorch.shape}"

    # Calculate metrics
    mse = torch.mean((output_rla - output_pytorch)**2)
    print(f"Mean Squared Error: {mse.item()}")

    cos_sim = torch.nn.functional.cosine_similarity(output_rla.flatten(), output_pytorch.flatten(), dim=0)
    print(f"Cosine Similarity: {cos_sim.item()}")

    # Primary quality check: Cosine similarity should be high
    assert cos_sim.item() > 0.95, \
        f"Cosine similarity {cos_sim.item()} is below threshold 0.95. MSE: {mse.item()}"

    # Secondary quality check: MSE should be small
    assert mse.item() < 0.01, \
        f"MSE {mse.item()} is too large. CosSim: {cos_sim.item()}"

    print("Test passed: RLA SDPA output is close to PyTorch SDPA output")

if __name__ == '__main__':
    pytest.main([__file__])
