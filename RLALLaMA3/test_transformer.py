import torch
import pytest
from RLALLaMA3.LLaMA3 import ModelArgs, Transformer

"""
To run this test:
# source your virtual env
cd RLALLaMA3
python -m pytest test_transformer.py -s
"""


def test_transformer_rla():
    # Model parameters
    dim = 64
    n_heads = 4
    n_layers = 2
    vocab_size = 1000
    max_seq_len = 32

    # Calculate RLA parameters
    # For attention QKV and output projections (dim -> dim)
    attn_sample_exact = int(dim * 0.9)  # 90% of dim
    attn_proj_dim = int((dim - attn_sample_exact) * 0.5)  # 50% of remaining dim

    # For FFN input projection (dim -> hidden_dim)
    hidden_dim = int(3 * dim / 2)  # Standard SwiGLU scaling
    ffn_in_sample_exact = int(dim * 0.9)  # 90% of dim
    ffn_in_proj_dim = int((dim - ffn_in_sample_exact) * 0.5)  # 50% of remaining dim

    # For FFN output projection (hidden_dim -> dim)
    ffn_out_sample_exact = int(hidden_dim * 0.9)  # 90% of hidden_dim
    ffn_out_proj_dim = int((hidden_dim - ffn_out_sample_exact) * 0.5)  # 50% of remaining dim

    # For SDPA internal matmuls
    head_dim = dim // n_heads
    sdpa_qk_sample_exact = int(head_dim * 0.9)  # 90% of head_dim
    sdpa_qk_proj_dim = int((head_dim - sdpa_qk_sample_exact) * 0.5)  # 50% of remaining dim

    # Use actual sequence length instead of max_seq_len for SDPA SV parameters
    seq_len = 16  # This matches the sequence length we use in the test
    sdpa_sv_sample_exact = int(seq_len * 0.9)  # 90% of seq_len
    sdpa_sv_proj_dim = int((seq_len - sdpa_sv_sample_exact) * 0.5)  # 50% of remaining dim

    # Create model args with RLA settings
    args = ModelArgs(
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        deterministic=False,  # Enable RLA
        sketch_mode='rademacher',
        
        # Attention QKV and output projections
        rla_attn_qkv_sample_exact_dim=attn_sample_exact,
        rla_attn_qkv_projection_dim=attn_proj_dim,
        rla_attn_out_sample_exact_dim=attn_sample_exact,
        rla_attn_out_projection_dim=attn_proj_dim,
        
        # FFN projections
        rla_ffn_in_sample_exact_dim=ffn_in_sample_exact,
        rla_ffn_in_projection_dim=ffn_in_proj_dim,
        rla_ffn_out_sample_exact_dim=ffn_out_sample_exact,
        rla_ffn_out_projection_dim=ffn_out_proj_dim,
        
        # SDPA internal matmuls
        rla_sdpa_qk_sample_exact_dim=sdpa_qk_sample_exact,
        rla_sdpa_qk_projection_dim=sdpa_qk_proj_dim,
        rla_sdpa_sv_sample_exact_dim=sdpa_sv_sample_exact,
        rla_sdpa_sv_projection_dim=sdpa_sv_proj_dim
    )

    # Create model
    model = Transformer(args)
    
    # Create input
    batch_size = 2
    seq_len = 16
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Run model in RLA mode
    output_rla = model(tokens)
    
    # Run model in deterministic mode
    model.deterministic_mode(True)
    output_det = model(tokens)
    
    # Print shapes and metrics
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output_rla.shape}")
    
    # Calculate metrics
    mse = torch.mean((output_rla - output_det)**2)
    print(f"Mean Squared Error: {mse.item()}")
    
    cos_sim = torch.nn.functional.cosine_similarity(
        output_rla.flatten(), 
        output_det.flatten(), 
        dim=0
    )
    print(f"Cosine Similarity: {cos_sim.item()}")
    
    # Assertions
    assert output_rla.shape == output_det.shape, \
        f"Shape mismatch: RLA is {output_rla.shape}, Deterministic is {output_det.shape}"
    
    # Check that outputs are reasonably close
    assert cos_sim.item() > 0.90, \
        f"Cosine similarity {cos_sim.item()} is below threshold 0.90. MSE: {mse.item()}"
    
    assert mse.item() < 0.1, \
        f"MSE {mse.item()} is too large. CosSim: {cos_sim.item()}"
    
    print("Test passed: Transformer with RLA produces reasonable approximations")

if __name__ == '__main__':
    pytest.main([__file__]) 