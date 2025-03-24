import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    """Configuration class for Transformer model hyperparameters."""
    dim: int = 4096           # Dimensionality of the model (embedding size)
    n_layers: int = 32        # Number of transformer layers
    n_heads: int = 32         # Number of attention heads in each layer
    n_kv_heads: Optional[int] = None  # Number of key/value heads (defaults to n_heads if None)
    hidden_dim: int = 14436   # Size of the hidden layer in the feed-forward network
    vocab_size: int = -1      # Vocabulary size (must be set before use)
    norm_eps: float = 1e-5    # Epsilon value for numerical stability in RMS normalization
    rope_theta: float = 500000  # Base frequency for RoPE (Rotary Position Embedding)
    max_seq_len: int = 2048   # Maximum sequence length for precomputing RoPE frequencies

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads if n_kv_heads < n_heads.

    Args:
        x (torch.Tensor): Input tensor of shape (bs, slen, n_kv_heads, head_dim).
        n_rep (int): Number of times to repeat each key/value head.

    Returns:
        torch.Tensor: Repeated tensor of shape (bs, slen, n_kv_heads * n_rep, head_dim).
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # Add a dimension for repetition, expand, then reshape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute frequency values for Rotary Position Embeddings (RoPE).

    Args:
        dim (int): Head dimension (must be even).
        end (int): Maximum sequence length to precompute for.
        theta (float): Base frequency for RoPE. Defaults to 10000.0.

    Returns:
        torch.Tensor: Complex frequencies of shape (end, dim//2, 2) with sin and cos components.
    """
    # Compute frequencies based on position and dimension
    freqs = theta ** (-torch.arange(0, dim, 2).float() / dim)  # Shape: (dim//2,)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # Shape: (end,)
    freqs = torch.outer(t, freqs)  # Shape: (end, dim//2)
    # Stack sin and cos for complex representation
    freqs_cis = torch.stack([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # Shape: (end, dim//2, 2)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor of shape (bs, slen, n_heads, head_dim).
        xk (torch.Tensor): Key tensor of shape (bs, slen, n_kv_heads, head_dim).
        freqs_cis (torch.Tensor): Precomputed RoPE frequencies of shape (slen, head_dim//2, 2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
    """
    # Reshape to split head_dim into pairs for rotation
    xq = xq.view(*xq.shape[:-1], -1, 2)  # Shape: (bs, slen, n_heads, head_dim//2, 2)
    xk = xk.view(*xk.shape[:-1], -1, 2)  # Shape: (bs, slen, n_kv_heads, head_dim//2, 2)
    freqs_cis = freqs_cis.unsqueeze(1)   # Shape: (slen, 1, head_dim//2, 2)

    def last_dim_dot(a: torch.Tensor, b: torch.Tensor):
        """Dot product along the last dimension."""
        return torch.einsum('...i,...i->...', a, b)

    # Apply rotation using sin and cos components
    xq_row_1 = last_dim_dot(xq, freqs_cis)
    xk_row_1 = last_dim_dot(xk, freqs_cis)

    # Flip and negate for the second component of the rotation
    freqs_cis = freqs_cis.flip(dims=[-1])
    freqs_cis[..., 1] = -freqs_cis[..., 1]

    xq_row_0 = last_dim_dot(xq, freqs_cis)
    xk_row_0 = last_dim_dot(xk, freqs_cis)

    # Stack and flatten back to original shape
    xq = torch.stack([xq_row_0, xq_row_1], dim=-1).flatten(start_dim=-2)
    xk = torch.stack([xk_row_0, xk_row_1], dim=-1).flatten(start_dim=-2)
    return xq, xk

class Attention(nn.Module):
    """Multi-head attention module without KV caching."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        # Set key/value heads (defaults to n_heads if not specified)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads  # Number of query heads
        self.n_local_kv_heads = self.n_kv_heads  # Number of key/value heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # Repetition factor for grouped-query attention
        self.head_dim = args.dim // args.n_heads  # Dimension per head

        # Linear projections for query, key, value, and output
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for attention mechanism, processing the full sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seqlen, dim).
            freqs_cis (torch.Tensor): Precomputed RoPE frequencies for the sequence length.
            mask (Optional[torch.Tensor]): Attention mask (e.g., causal mask).

        Returns:
            torch.Tensor: Output tensor of shape (bsz, seqlen, dim).
        """
        bsz, seqlen, _ = x.shape
        # Project input to queries, keys, and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary position embeddings to queries and keys
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Use current keys and values directly (no caching)
        keys = xk
        values = xv

        # Repeat key/value heads if n_kv_heads < n_heads (for grouped-query attention)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention computation
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # Apply mask (e.g., causal mask)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # Softmax over last dim

        # Compute weighted values and project back to input dimension
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    """Feed-forward network with gated activation (e.g., SwiGLU variant)."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        """
        Initialize FeedForward module.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            hidden_dim (int): Hidden layer dimension.
            bias (bool): Whether to include bias in linear layers. Defaults to False.
        """
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias)  # First projection
        self.w2 = nn.Linear(hidden_dim, out_dim, bias)  # Output projection
        self.w3 = nn.Linear(in_dim, hidden_dim, bias)  # Gate projection

    def forward(self, x):
        """Apply gated feed-forward computation: w2(silu(w1(x)) * w3(x))."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward layers."""
    def __init__(self, args: ModelArgs):
        """
        Initialize TransformerBlock.

        Args:
            args (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        self.attention = Attention(args)  # Multi-head attention layer
        self.feed_forward = FeedForward(
            in_dim=args.dim,
            out_dim=args.dim,
            hidden_dim=args.hidden_dim,
        )  # Feed-forward layer
        # Use PyTorch's RMSNorm for normalization
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (bsz, seqlen, dim).
            freqs_cis (torch.Tensor): Precomputed RoPE frequencies.
            mask (Optional[torch.Tensor]): Attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (bsz, seqlen, dim).
        """
        # Apply attention with residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        # Apply feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    """Full Transformer model with multiple layers."""
    def __init__(self, params: ModelArgs):
        """
        Initialize the Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Token embedding layer
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim, padding_idx=0)
        # Stack of transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(params) for _ in range(params.n_layers)])
        # Final normalization using PyTorch's RMSNorm
        self.norm = nn.RMSNorm(params.dim, eps=params.norm_eps)
        # Output projection to vocabulary logits
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Precompute RoPE frequencies for the maximum sequence length
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Forward pass for the Transformer model, processing the full sequence.

        Args:
            tokens (torch.Tensor): Input token indices of shape (bsz, seqlen).

        Returns:
            torch.Tensor: Logits of shape (bsz, seqlen, vocab_size).
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # Convert tokens to embeddings
        self.freqs_cis = self.freqs_cis.to(h.device)  # Ensure freqs_cis is on the same device
        freqs_cis = self.freqs_cis[:seqlen]  # Slice frequencies to match sequence length

        # Create causal mask for autoregressive attention
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)  # Upper triangular mask for causality
            mask = mask.type_as(h)  # Match dtype of input tensor

        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)  # Apply final normalization
        output = self.output(h).float()  # Project to vocabulary logits
        return output