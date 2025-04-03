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

    sketch_mode: str = 'rademacher'  # Type of random projection ('rademacher' or 'gaussian')
    attention_qkv_sketch_size: int = 64 # Size of the sketch for query/key/value projections
    attention_score_sketch_size: int = 64 # Size of the sketch for attention score projections
    attention_weighed_sum_sketch_size: int = 64 # Size of the sketch for weighted sum projections
    attention_out_sketch_size: int = 64 # Size of the sketch for output projections
    feedforward_sketch_size_in: int = 64 # Size of the sketch for feed-forward projections
    feedforward_sketch_size_out: int = 64 # Size of the sketch for output projections
    rlamha: bool = False  # Flag to use randomized multi-head attention
    deterministic: bool = False  # Flag to switch between deterministic and randomized computation

def projection_sketch_mm(A, B, sketch_size, mode='rademacher'):
    """
    Approximates the matrix product A @ B using a projection sketch.

    Args:
        A: The first matrix (or batch of matrices).
        B: The second matrix (or batch of matrices).
        sketch_size: The size of the sketch (inner dimension of S).
        mode: Type of random projection ('rademacher' or 'gaussian').

    Returns:
        The approximated matrix product.
    """
    # Ensure inputs are tensors
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("Inputs A and B must be torch tensors.")
    if len(A.shape) != 3 or len(B.shape) != 2:
        raise ValueError("Not supported now: A should be 3D and B should be 2D.")

    # Determine shape for S: A(... m, n), B(... n, p) -> S(... n, c)
    # A @ S @ S.T @ B
    S_shape = (*A.shape[:-2], A.shape[-1], sketch_size)
    device = A.device

    if mode == 'rademacher':
        # +/- 1 entries
        S = (torch.randint(0, 2, S_shape, device=device, dtype=A.dtype) * 2 - 1)
    elif mode == 'gaussian':
        # N(0, 1) entries
        S = torch.randn(S_shape, device=device, dtype=A.dtype)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Scaling factor: E[S @ S.T] = sketch_size * I
    # We need A @ (S @ S.T / sketch_size) @ B
    # Compute as (A @ S / sketch_size) @ (S.T @ B)
    scaling_factor = sketch_size
    
    # --- Check for potential dimension mismatch before matmul ---
    if A.shape[-1] != S.shape[-2]:
         raise RuntimeError(f"Matrix A last dim {A.shape[-1]} doesn't match S second-to-last dim {S.shape[-2]}")
    if S.transpose(-1, -2).shape[-1] != B.shape[-2]:
         raise RuntimeError(f"Matrix S.T last dim {S.transpose(-1, -2).shape[-1]} doesn't match B second-to-last dim {B.shape[-2]}")
    # --- End Check ---

    AS = torch.matmul(A, S) / scaling_factor
    SB = torch.matmul(S.transpose(-1, -2), B)

    # --- Check for potential dimension mismatch before final matmul ---
    if AS.shape[-1] != SB.shape[-2]:
         raise RuntimeError(f"Matrix AS last dim {AS.shape[-1]} doesn't match SB second-to-last dim {SB.shape[-2]}")
    # --- End Check ---

    AB_bar = torch.matmul(AS, SB)
    return AB_bar

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

class RLAAttention(nn.Module):
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

        self.rlamha = args.rlamha  # Randomized Multi-Head Attention
        self.attention_score_sketch_size = args.attention_score_sketch_size
        self.attention_weighed_sum_sketch_size = args.attention_weighed_sum_sketch_size

        # Linear projections for query, key, value, and output
        '''
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        '''
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        qkv_dim = self.q_dim + self.kv_dim * 2
        self.wqkv = RLALinear(args.dim, qkv_dim, bias=False,
                              sketch_size=args.attention_qkv_sketch_size, sketch_mode=args.sketch_mode,
                              deterministic=args.deterministic)
        self.wo = RLALinear(args.n_heads * self.head_dim, args.dim, bias=False,
                            sketch_size=args.attention_out_sketch_size, sketch_mode=args.sketch_mode,
                            deterministic=args.deterministic)

    def deterministic_mode(self, enable: bool = True) -> "RLAAttention":
        # Set the deterministic mode for the attention layer
        self.wqkv.deterministic_mode(enable)
        self.wo.deterministic_mode(enable)
        self.rlamha = enable
        return self

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
        '''
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        '''
        xqkv = self.wqkv(x)  # Shape: (bsz, seqlen, q_dim + kv_dim * 2)
        xq, xk, xv = xqkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)  # Split into q, k, v

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
        """
        Initialize FeedForward module.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            hidden_dim (int): Hidden layer dimension.
            bias (bool): Whether to include bias in linear layers. Defaults to False.
        """
        super().__init__()
        '''
        self.w1 = nn.Linear(in_dim, hidden_dim, bias)  # First projection
        self.w2 = nn.Linear(hidden_dim, out_dim, bias)  # Output projection
        self.w3 = nn.Linear(in_dim, hidden_dim, bias)  # Gate projection
        '''
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
        """Apply gated feed-forward computation: w2(silu(w1(x)) * w3(x))."""
        x13 = self.w13(x)  # Shape: (bsz, seqlen, hidden_dim * 2)
        x1, x3 = x13.chunk(2, dim=-1)  # Split into two parts
        return self.w2(F.silu(x1) * x3)  # Apply gated activation and output projection

class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward layers."""
    def __init__(self, args: ModelArgs):
        """
        Initialize TransformerBlock.

        Args:
            args (ModelArgs): Model configuration parameters.
        """
        super().__init__()
        self.attention = RLAAttention(args)  # Multi-head attention layer
        self.feed_forward = RLAFeedForward(
            in_dim=args.dim,
            out_dim=args.dim,
            hidden_dim=args.hidden_dim,
            bias=False,
            sketch_size_in=args.feedforward_sketch_size_in,
            sketch_size_out=args.feedforward_sketch_size_out,
            sketch_mode=args.sketch_mode,
            deterministic=args.deterministic,
        )  # Feed-forward layer
        # Use PyTorch's RMSNorm for normalization
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
        self.freqs_cis = None
        self.update_freqs_cis(params.max_seq_len)

    def update_freqs_cis(self, max_seq_len: int):
        """
        Update the precomputed RoPE frequencies for a new maximum sequence length.

        Args:
            max_seq_len (int): New maximum sequence length.
        """
        self.params.max_seq_len = max_seq_len
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            max_seq_len,
            self.params.rope_theta,
        )
        return self
    
    def deterministic_mode(self, enable: bool = True) -> "Transformer":
        # Set the deterministic mode for all transformer layers
        for layer in self.layers:
            layer.deterministic_mode(enable)
        return self

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
    