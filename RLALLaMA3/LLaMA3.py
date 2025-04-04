import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming these are in a separate file or defined above
from .RLALinear import RLALinear # Import RLALinear
from .RLACore import rla_scaled_dot_product_attention # Import the SDPA function

# Modified ModelArgs without rlamha
@dataclass
class ModelArgs:
    """Configuration class for Transformer model hyperparameters."""
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    hidden_dim: int = 14436
    vocab_size: int = -1
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_seq_len: int = 2048

    sketch_mode: str = 'rademacher'
    attention_qkv_sketch_size: int = 0 # Sketch for Wq, Wk, Wv projections
    attention_out_sketch_size: int = 0 # Sketch for Wo projection
    feedforward_sketch_size_in: int = 0 # Sketch for FeedForward w1/w3
    feedforward_sketch_size_out: int = 0 # Sketch for FeedForward w2
    # Args for RLA SDPA
    attention_score_sketch_size: int = 0 # Sketch for QK^T. 0 means deterministic matmul.
    attention_weighed_sum_sketch_size: int = 0 # Sketch for Scores@V. 0 means deterministic matmul.
    # Global switch for RLALinear layers. If True, sketch sizes above are ignored.
    deterministic: bool = False


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
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Store RLA SDPA parameters from args
        self.attention_score_sketch_size = args.attention_score_sketch_size
        self.attention_weighed_sum_sketch_size = args.attention_weighed_sum_sketch_size
        self.sketch_mode = args.sketch_mode
        self.deterministic = args.deterministic # Global flag for RLA

        # --- Linear projections ---
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        qkv_dim = self.q_dim + self.kv_dim * 2

        self.wqkv = RLALinear(args.dim, qkv_dim, bias=False,
                              sketch_size=args.attention_qkv_sketch_size,
                              sketch_mode=args.sketch_mode,
                              deterministic=args.deterministic) # Use global flag
        self.wo = RLALinear(args.n_heads * self.head_dim, args.dim, bias=False,
                            sketch_size=args.attention_out_sketch_size,
                            sketch_mode=args.sketch_mode,
                            deterministic=args.deterministic) # Use global flag

    def deterministic_mode(self, enable: bool = True) -> "RLAAttention":
        """Sets the deterministic mode for the Wqkv and Wo linear layers."""
        self.wqkv.deterministic_mode(enable)
        self.wo.deterministic_mode(enable)
        self.deterministic = enable # Update local state
        return self

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        is_causal: bool = False
    ):
        bsz, seqlen, _ = x.shape

        # 1. Project input to queries, keys, and values using RLALinear
        # The deterministic flag of self.wqkv handles whether this is RLA or not
        xqkv = self.wqkv(x)
        xq, xk, xv = xqkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        # 2. Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 3. Apply rotary position embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 4. Transpose for SDPA input format (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 5. Compute attention output using the dedicated function
        # The rla_scaled_dot_product_attention function determines RLA use
        # based on the sketch size arguments being > 0.
        output = rla_scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            n_rep=self.n_rep,
            is_causal=is_causal,
            # RLA Params for SDPA - driven by sketch sizes
            deterministic=self.deterministic,
            sketch_size_qk=self.attention_score_sketch_size,
            sketch_size_sv=self.attention_weighed_sum_sketch_size,
            sketch_mode=self.sketch_mode
        )

        # 6. Reshape output back
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 7. Final output projection using RLALinear
        # The deterministic flag of self.wo handles whether this is RLA or not
        return self.wo(output)

class RLAFeedForward(nn.Module):
    """Feed-forward network with gated activation (e.g., SwiGLU variant)."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        bias: bool = False,
        sketch_size_in: int = 0,
        sketch_size_out: int = 0,
        sketch_mode: str = 'rademacher',
        deterministic: bool = False,
    ):
        super().__init__()
        self.w13 = RLALinear(in_dim, hidden_dim * 2, bias=bias,
                             sketch_size=sketch_size_in, sketch_mode=sketch_mode,
                             deterministic=deterministic)
        self.w2 = RLALinear(hidden_dim, out_dim, bias=bias,
                            sketch_size=sketch_size_out, sketch_mode=sketch_mode,
                            deterministic=deterministic)

    def deterministic_mode(self, enable: bool = True) -> "RLAFeedForward":
        # Set the deterministic mode for the feed-forward layer
        self.w13.deterministic_mode(enable)
        self.w2.deterministic_mode(enable)
        return self

    def forward(self, x):
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x3)

class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward layers."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = RLAAttention(args)
        self.feed_forward = RLAFeedForward(
            in_dim=args.dim,
            out_dim=args.dim,
            hidden_dim=args.hidden_dim,
            bias=False, # Often False in FFN
            sketch_size_in=args.feedforward_sketch_size_in,
            sketch_size_out=args.feedforward_sketch_size_out,
            sketch_mode=args.sketch_mode,
            deterministic=args.deterministic,
        )
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def deterministic_mode(self, enable: bool = True) -> "TransformerBlock":
        # Set the deterministic mode for the transformer block
        self.attention.deterministic_mode(enable)
        self.feed_forward.deterministic_mode(enable)
        return self

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        # Removed mask from here, pass is_causal=True to attention
    ):
        # Apply attention with residual connection and causality
        h = x + self.attention(self.attention_norm(x), freqs_cis, is_causal=True)
        # Apply feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
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
        # Keep output as standard Linear unless RLA is desired here too
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        # If RLA for output, replace above with:
        # self.output = RLALinear(params.dim, params.vocab_size, bias=False,
        #                         sketch_size=params.output_sketch_size, # Add this to ModelArgs
        #                         sketch_mode=params.sketch_mode,
        #                         deterministic=params.deterministic)

        self.freqs_cis = None
        if params.max_seq_len > 0: # Avoid computation if not needed
             self.update_freqs_cis(params.max_seq_len)

    def update_freqs_cis(self, max_seq_len: int):
        """Updates RoPE frequencies."""
        self.params.max_seq_len = max_seq_len
        # Ensure head_dim is correctly calculated
        head_dim = self.params.dim // self.params.n_heads
        if head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for RoPE")
        self.freqs_cis = precompute_freqs_cis(
            head_dim,
            max_seq_len,
            self.params.rope_theta,
        )
        return self

    def deterministic_mode(self, enable: bool = True) -> "Transformer":
        """Sets deterministic mode for all RLA layers."""
        self.params.deterministic = enable # Update config state
        for layer in self.layers:
            layer.deterministic_mode(enable)
        # If self.output is RLALinear, update it too
        # if isinstance(self.output, RLALinear):
        #     self.output.deterministic_mode(enable)
        return self

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        if self.freqs_cis is None or seqlen > self.freqs_cis.shape[0]:
             # Handle case where freqs_cis needs update or wasn't precomputed
             self.update_freqs_cis(seqlen)
             print(f"Warning: RoPE frequencies recomputed for seqlen {seqlen}")

        self.freqs_cis = self.freqs_cis.to(h.device) # Ensure device match
        freqs_cis = self.freqs_cis[:seqlen] # Slice

        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis) # Pass freqs_cis, is_causal=True is handled inside block
        h = self.norm(h)
        output = self.output(h).float()
        return output