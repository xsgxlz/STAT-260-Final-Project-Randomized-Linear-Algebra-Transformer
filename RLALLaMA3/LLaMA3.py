import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these are in a separate file or defined above
from .RLALinear import RLALinear # Import RLALinear
from .RLACore import rla_scaled_dot_product_attention # Import the SDPA function

# Modified ModelArgs without rlamhafrom dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArgs:
    """Configuration class for Transformer model hyperparameters, including RLA settings."""
    # Standard Transformer Args
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None # For Grouped Query Attention
    hidden_dim: Optional[int] = None # Dimension of the hidden layer in FFN, typically 4*dim or 2/3 * 4 * dim for SwiGLU
    vocab_size: int = -1  # To be set based on tokenizer
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 2048

    # --- Randomized Linear Algebra (RLA) Specific Args ---

    # Global RLA settings
    deterministic: bool = False # If True, all RLA approximations are disabled, and exact computations are used.
                                # RLALinear layers behave like nn.Linear.
                                # RLA SDPA behaves like standard SDPA.
    sketch_mode: str = 'rademacher' # Default projection mode ('rademacher' or 'gaussian') for all RLA projections.

    # RLA settings for Linear layers in Attention (Wq, Wk, Wv, Wo)
    # These control RLALinear for Q, K, V, and Output projections.
    # Common dimension for these is `dim` (model dimension) -> `dim` (or head_dim * n_heads)
    rla_attn_qkv_sample_exact_dim: int = 0 # Num exact dims for Wq, Wk, Wv input features (dim).
    rla_attn_qkv_projection_dim: int = 0 # Proj. dim for remaining input features of Wq, Wk, Wv.
                                         # If 0, remaining features are effectively dropped if not deterministic.

    rla_attn_out_sample_exact_dim: int = 0 # Num exact dims for Wo input features (dim).
    rla_attn_out_projection_dim: int = 0 # Proj. dim for remaining input features of Wo.

    # RLA settings for Linear layers in Feed-Forward Network (FFN)
    # For SwiGLU FFN: w1, w_gate (both map hidden_dim x dim), w2 (maps dim x hidden_dim)
    # Common dimension for FFN input (w1, w_gate) is `dim`.
    # Common dimension for FFN output (w2) is `hidden_dim`.
    rla_ffn_in_sample_exact_dim: int = 0   # Num exact dims for FFN w1 & w_gate input features (dim).
    rla_ffn_in_projection_dim: int = 0   # Proj. dim for remaining input features of w1 & w_gate.

    rla_ffn_out_sample_exact_dim: int = 0  # Num exact dims for FFN w2 input features (hidden_dim).
    rla_ffn_out_projection_dim: int = 0  # Proj. dim for remaining input features of w2.

    # RLA settings for internal matmuls in Scaled Dot-Product Attention (SDPA)
    # These control `sample_and_project_mm` calls within `rla_scaled_dot_product_attention`.
    # For QK^T: common dimension is `E_q` (query feature dimension, query.size(-1)).
    rla_sdpa_qk_sample_exact_dim: int = 0 # Num exact dims for QK^T.
    rla_sdpa_qk_projection_dim: int = 0 # Proj. dim for remaining dims of QK^T. (Replaces old attention_score_sketch_size)

    # For Scores@V: common dimension is `S` (sequence length of key/value, key.size(-2)).
    rla_sdpa_sv_sample_exact_dim: int = 0 # Num exact dims for Scores@V.
    rla_sdpa_sv_projection_dim: int = 0 # Proj. dim for remaining dims of Scores@V. (Replaces old attention_weighed_sum_sketch_size)

    def __post_init__(self):
        if self.hidden_dim is None:
            # LLaMA3-like SwiGLU FFN: up_proj and gate_proj (dim -> multiple_of * dim), then down_proj (multiple_of * dim -> dim)
            # A common practice is multiple_of * hidden_dim = 2/3 * 4 * dim, so hidden_dim is often (2/3 * 4 * dim)
            # Or, simply 4 * dim if not SwiGLU, or a specific multiple for SwiGLU like Llama's 2.66x (rounded)
            self.hidden_dim = int(3 * self.dim / 2) # Example for SwiGLU based on some conventions

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads # Standard Multi-Head Attention

        # If deterministic is True, RLA parameters for sampling/projection dimensions are conceptually ignored
        # by the execution path (which will choose exact matmul).
        # However, keeping them allows easy switching between modes without losing settings.

# --- RoPE Helper Functions --- (Assuming these are defined as before)
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes Rotary Position Embedding frequencies."""
    freqs = theta ** (-torch.arange(0, dim, 2).float() / dim)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.stack([torch.sin(freqs), torch.cos(freqs)], dim=-1)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings."""
    xq_shape = xq.shape
    xk_shape = xk.shape
    xq = xq.view(*xq.shape[:-1], -1, 2)
    xk = xk.view(*xk.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(freqs_cis.shape[0], 1, -1, 2) # Adjust shape for broadcasting

    def last_dim_dot(a: torch.Tensor, b: torch.Tensor):
        return torch.einsum('...i,...i->...', a, b)

    xq_row_1 = last_dim_dot(xq, freqs_cis)
    xk_row_1 = last_dim_dot(xk, freqs_cis)

    freqs_cis = freqs_cis.flip(dims=[-1])
    freqs_cis[..., 1] = -freqs_cis[..., 1]

    xq_row_0 = last_dim_dot(xq, freqs_cis)
    xk_row_0 = last_dim_dot(xk, freqs_cis)

    xq = torch.stack([xq_row_0, xq_row_1], dim=-1).flatten(start_dim=-2).view(*xq_shape)
    xk = torch.stack([xk_row_0, xk_row_1], dim=-1).flatten(start_dim=-2).view(*xk_shape)
    return xq, xk
# --- End RoPE ---


class RLAAttention(nn.Module):
    """Multi-head attention module using rla_scaled_dot_product_attention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        if self.n_local_heads % self.n_local_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_local_heads}) must be divisible by n_kv_heads ({self.n_local_kv_heads})"
            )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Store RLA SDPA parameters from args for internal use in forward pass
        self.rla_sdpa_qk_sample_exact_dim = args.rla_sdpa_qk_sample_exact_dim
        self.rla_sdpa_qk_projection_dim = args.rla_sdpa_qk_projection_dim
        self.rla_sdpa_sv_sample_exact_dim = args.rla_sdpa_sv_sample_exact_dim
        self.rla_sdpa_sv_projection_dim = args.rla_sdpa_sv_projection_dim
        self.sketch_mode = args.sketch_mode
        self.deterministic = args.deterministic # Global flag for RLA

        # --- Linear projections ---
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        # For fused QKV, the output dimension is q_dim + k_dim + v_dim
        # k_dim = v_dim = self.n_kv_heads * self.head_dim
        qkv_dim = self.q_dim + self.kv_dim * 2 # q_dim for Q, kv_dim for K, kv_dim for V

        self.wqkv = RLALinear(
            args.dim,
            qkv_dim, # Output dim for concatenated Q, K, V
            bias=False, # Typically False for QKV projections in LLaMA-like models
            sample_exact_dim=args.rla_attn_qkv_sample_exact_dim,
            projection_dim=args.rla_attn_qkv_projection_dim,
            projection_mode=args.sketch_mode,
            deterministic=args.deterministic
        )
        self.wo = RLALinear(
            args.n_heads * self.head_dim, # Input dim is total head dimensions
            args.dim, # Output dim is model dimension
            bias=False, # Typically False for output projection
            sample_exact_dim=args.rla_attn_out_sample_exact_dim,
            projection_dim=args.rla_attn_out_projection_dim,
            projection_mode=args.sketch_mode,
            deterministic=args.deterministic
        )

    def deterministic_mode(self, enable: bool = True) -> "RLAAttention":
        """Sets the deterministic mode for the Wqkv and Wo linear layers and internal SDPA calls."""
        self.wqkv.deterministic_mode(enable)
        self.wo.deterministic_mode(enable)
        self.deterministic = enable # Update local state for SDPA
        return self

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        is_causal: bool = False,
        attn_mask: Optional[torch.Tensor] = None # Added attn_mask for rla_sdpa
    ):
        bsz, seqlen, _ = x.shape

        # 1. Project input to queries, keys, and values using RLALinear
        xqkv = self.wqkv(x) # Shape: (bsz, seqlen, q_dim + 2*kv_dim)
        
        # Split into Q, K, V
        xq, xk, xv = xqkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        # 2. Reshape for multi-head/multi-KV-head attention
        # Q: (bsz, seqlen, n_local_heads, head_dim)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # K: (bsz, seqlen, n_local_kv_heads, head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # V: (bsz, seqlen, n_local_kv_heads, head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 3. Apply rotary position embeddings
        # apply_rotary_emb expects (bsz, seqlen, n_heads, head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 4. Transpose for SDPA input format (bsz, n_heads, seqlen, head_dim)
        # Q: (bsz, n_local_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        # K: (bsz, n_local_kv_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        # V: (bsz, n_local_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)

        # 5. Compute attention output using the dedicated function
        # rla_scaled_dot_product_attention handles GQA via n_rep internally.
        # It also handles the self.deterministic flag.
        output = rla_scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            n_rep=self.n_rep,
            attn_mask=attn_mask, # Pass attn_mask
            is_causal=is_causal,
            # RLA Params for SDPA internal matmuls
            deterministic=self.deterministic, # This global flag overrides RLA in SDPA
            sample_exact_dim_qk=self.rla_sdpa_qk_sample_exact_dim,
            sketch_size_qk=self.rla_sdpa_qk_projection_dim, # Note: sketch_size_qk in SDPA is projection_dim
            sample_exact_dim_sv=self.rla_sdpa_sv_sample_exact_dim,
            sketch_size_sv=self.rla_sdpa_sv_projection_dim, # Note: sketch_size_sv in SDPA is projection_dim
            sketch_mode=self.sketch_mode
        )
        # output shape: (bsz, n_local_heads, seqlen, head_dim)

        # 6. Reshape output back
        # (bsz, seqlen, n_local_heads * head_dim) which is (bsz, seqlen, dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 7. Final output projection using RLALinear
        return self.wo(output)

class RLAFeedForward(nn.Module):
    """Feed-forward network with gated activation (e.g., SwiGLU variant),
    using RLALinear and configured by ModelArgs."""
    def __init__(self, args: ModelArgs, bias: bool = False): # Added bias as an arg, ModelArgs has most others
        super().__init__()

        self.deterministic = args.deterministic # Store global deterministic flag

        # For SwiGLU FFN:
        # w1 and w3 (or w_gate) project from dim to hidden_dim.
        # These can be combined into a single RLALinear layer (w13)
        # that projects from dim to 2 * hidden_dim.
        # The input to these layers is 'dim' (args.dim).
        self.w13 = RLALinear(
            in_features=args.dim,
            out_features=args.hidden_dim * 2, # For SwiGLU, two hidden projections
            bias=bias, # Usually False in LLaMA-like FFNs
            sample_exact_dim=args.rla_ffn_in_sample_exact_dim,
            projection_dim=args.rla_ffn_in_projection_dim,
            projection_mode=args.sketch_mode,
            deterministic=args.deterministic
        )

        # w2 projects from hidden_dim back to dim.
        # The input to this layer is 'hidden_dim' (args.hidden_dim).
        self.w2 = RLALinear(
            in_features=args.hidden_dim,
            out_features=args.dim,
            bias=bias, # Usually False in LLaMA-like FFNs
            sample_exact_dim=args.rla_ffn_out_sample_exact_dim,
            projection_dim=args.rla_ffn_out_projection_dim,
            projection_mode=args.sketch_mode,
            deterministic=args.deterministic
        )

    def deterministic_mode(self, enable: bool = True) -> "RLAFeedForward":
        """Sets the deterministic mode for the internal RLALinear layers."""
        self.w13.deterministic_mode(enable)
        self.w2.deterministic_mode(enable)
        self.deterministic = enable # Update local state if needed elsewhere, though RLALinear holds its own
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        
        # Project to 2 * hidden_dim
        # x13: (batch_size, seq_len, 2 * hidden_dim)
        x13 = self.w13(x)
        
        # Split into two parts for SwiGLU
        # x1: (batch_size, seq_len, hidden_dim)
        # x3: (batch_size, seq_len, hidden_dim)
        x1, x3 = x13.chunk(2, dim=-1)
        
        # Apply SwiGLU activation: SiLU(x1) * x3
        # activated: (batch_size, seq_len, hidden_dim)
        activated_x = F.silu(x1) * x3
        
        # Project back to dim
        # output: (batch_size, seq_len, dim)
        output = self.w2(activated_x)
        
        return output

class TransformerBlock(nn.Module):
    """Single Transformer block with RLA-enhanced attention and feed-forward layers."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = RLAAttention(args)
        # RLAFeedForward now takes ModelArgs directly for its configuration.
        # The 'bias' for FFN layers can be controlled here or via ModelArgs if added.
        self.feed_forward = RLAFeedForward(args=args, bias=False) # LLaMA-like FFNs often have bias=False

        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        # Store args if needed for other purposes, though submodules handle their own RLA config from args
        # self.args = args

    def deterministic_mode(self, enable: bool = True) -> "TransformerBlock":
        """Sets the deterministic mode for RLA components within this block."""
        self.attention.deterministic_mode(enable)
        self.feed_forward.deterministic_mode(enable)
        return self

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        # If a custom mask is needed (e.g., for padding), it should be passed to self.attention
        # For standard causal attention, is_causal=True is sufficient.
        attn_mask: Optional[torch.Tensor] = None # Allow passing an explicit attention mask
    ):
        # Pre-normalization for attention
        normed_x_attn = self.attention_norm(x)
        h = x + self.attention(normed_x_attn, freqs_cis, is_causal=(attn_mask is None), attn_mask=attn_mask)
        # Pre-normalization for FFN
        normed_h_ffn = self.ffn_norm(h)
        # Feed-forward with residual connection
        out = h + self.feed_forward(normed_h_ffn)
        return out

class Transformer(nn.Module):
    """Full Transformer model with multiple layers."""
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim, padding_idx=0)
        self.layers = nn.ModuleList([TransformerBlock(params) for _ in range(params.n_layers)])
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)
        
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = None
        if params.max_seq_len > 0: # Avoid computation if not needed or if RoPE is disabled
             self.update_freqs_cis(params.max_seq_len)

    def update_freqs_cis(self, max_seq_len: int) -> "Transformer":
        """Updates RoPE frequencies."""
        self.params.max_seq_len = max_seq_len
        # Ensure head_dim is correctly calculated based on current params
        head_dim = self.params.dim // self.params.n_heads
        if head_dim == 0: # Should not happen with valid params.dim and params.n_heads
            raise ValueError("Head dimension is zero. Check params.dim and params.n_heads.")
        if head_dim % 2 != 0: # RoPE requires even head dimension
            raise ValueError(
                f"Head dimension ({head_dim}) must be even for RoPE. "
                f"Got dim={self.params.dim}, n_heads={self.params.n_heads}"
            )
        
        # Assuming precompute_freqs_cis is available
        current_device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
        self.freqs_cis = precompute_freqs_cis(
            head_dim,
            max_seq_len,
            self.params.rope_theta,
        ).to(current_device) # Move to model's device during precomputation
        return self

    def deterministic_mode(self, enable: bool = True) -> "Transformer":
        """Sets deterministic mode for all RLA layers."""
        self.params.deterministic = enable # Update config state
        for layer in self.layers:
            if hasattr(layer, 'deterministic_mode'): # Ensure the layer has the method
                layer.deterministic_mode(enable)
        
        # If self.output were RLALinear, update it too.
        # This part remains commented as self.output is nn.Linear.
        # if isinstance(self.output, RLALinear):
        #     self.output.deterministic_mode(enable)
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if self.freqs_cis is None or seqlen > self.freqs_cis.shape[0]:
             # Handle case where freqs_cis needs update or wasn't precomputed for this length
             print(f"Warning: RoPE frequencies recomputed/updated for seqlen {seqlen}. Current max_seq_len: {self.params.max_seq_len}")
             self.update_freqs_cis(seqlen) # This will move freqs_cis to the model's device
        
        # Ensure freqs_cis is on the same device as input embeddings 'h'
        # update_freqs_cis should handle moving to device, but double check or ensure here.
        if self.freqs_cis.device != h.device:
            self.freqs_cis = self.freqs_cis.to(h.device)
            
        current_freqs_cis = self.freqs_cis[:seqlen] # Slice for current sequence length

        # Pass through transformer layers
        # TransformerBlock.forward expects (x, freqs_cis, attn_mask=None)
        # By not passing attn_mask, causal attention is implied within the block.
        for layer in self.layers:
            h = layer(h, current_freqs_cis) 
        
        h = self.norm(h)
        
        logits = self.output(h) # Standard nn.Linear for output
        
        return logits