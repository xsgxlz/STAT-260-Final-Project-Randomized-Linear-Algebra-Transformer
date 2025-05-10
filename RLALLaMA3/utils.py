import argparse
import math
import random
from dataclasses import dataclass, field

import torch
import functools

def linear_warmup_cosine_decay(current_step: int, warmup_steps: int, total_steps: int, min_lr: float):
    if current_step < warmup_steps:
        # Linear warm-up
        return float(current_step + 1) / float(warmup_steps + 1)
    # Cosine decay
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) + min_lr

def linear_warmup_cosine_decay_multiplicative(current_step: int, warmup_steps: int, total_steps: int, min_lr: float):
    if current_step != 1:
        pre = linear_warmup_cosine_decay(current_step - 1, warmup_steps, total_steps, min_lr)
        now = linear_warmup_cosine_decay(current_step, warmup_steps, total_steps, min_lr)
        return now / pre
    else:
        return linear_warmup_cosine_decay(1, warmup_steps, total_steps, min_lr)

def spearman_correlation(x, y):
    """
    Compute Spearman's rank correlation coefficient between two vectors.

    Args:
      x: Shape (N,)
      y: Shape (N,)

    Returns:
      Spearman's rank correlation coefficient (scalar)
    """

    # Compute ranks
    x_rank = torch.argsort(x).float() + 1  # Add 1 to get ranks starting from 1
    y_rank = torch.argsort(y).float() + 1

    # Compute difference in ranks
    d = x_rank - y_rank

    # Compute correlation coefficient
    n = x.size(0)
    rho = 1 - 6 * torch.sum(d ** 2) / (n * (n ** 2 - 1))
    return rho

def name_args(args, sep):
    dict_path_args = [args.task, args.max_level, args.random_seq_len, args.number_range]
    model_args = [args.model_type, args.dim, args.n_layers, args.n_heads, args.hidden_dim, args.batch_size]

    file_name_args = model_args

    dict_name = sep.join(map(str, dict_path_args))
    file_name = sep.join(map(str, file_name_args))

    return dict_name, file_name

def str_to_bool(v):
    """Convert string input to boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (e.g., True, False, yes, no, 1, 0).')

from dataclasses import dataclass, field
from typing import Optional, Tuple # Optional might be needed if n_kv_heads is re-introduced

@dataclass(frozen=True)
class Args:
    # Training parameters
    standard_lr: float = 2.5e-4
    standard_epoch: int = 20000
    standard_warmup_steps: int = 4000
    batch_size: int = 256
    min_lr: float = 1e-4
    grad_clip_max_norm: float = 5.0
    use_amp: bool = False
    use_compile: bool = False
    model_type: str = "node"

    # Data parameters
    task: str = "number_add"
    max_level: int = 40
    random_seq_len: bool = True
    number_range: tuple[int, int] = field(default_factory=lambda: (0, 99))

    # Model architecture parameters
    dim: int = 24
    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 84
    # n_kv_heads: Optional[int] = None # Kept commented as per original Args

    # --- Randomized Linear Algebra (RLA) Parameters ---
    deterministic: bool = False # Global switch for RLA. If True, all RLA approximations are disabled.
    sketch_mode: str = 'rademacher' # Default projection mode ('rademacher' or 'gaussian')

    # RLA for Attention Linear Layers (Wq, Wk, Wv, Wo)
    rla_attn_qkv_sample_exact_dim: int = 0
    rla_attn_qkv_projection_dim: int = 0
    rla_attn_out_sample_exact_dim: int = 0
    rla_attn_out_projection_dim: int = 0

    # RLA for Feed-Forward Network (FFN) Linear Layers
    rla_ffn_in_sample_exact_dim: int = 0
    rla_ffn_in_projection_dim: int = 0
    rla_ffn_out_sample_exact_dim: int = 0
    rla_ffn_out_projection_dim: int = 0

    # RLA for Scaled Dot-Product Attention (SDPA) internal matmuls
    rla_sdpa_qk_sample_exact_dim: int = 0
    rla_sdpa_qk_projection_dim: int = 0   # Was attention_score_sketch_size
    rla_sdpa_sv_sample_exact_dim: int = 0
    rla_sdpa_sv_projection_dim: int = 0   # Was attention_weighed_sum_sketch_size

    # Save path
    save_path: str = ""
    final_save_path: str = ""

    def __str__(self):
        """Return a formatted string representation of the Args object."""
        training_params = [
            f"model_type:         {self.model_type}",
            f"standard_lr:        {self.standard_lr:.1e}",
            f"standard_epoch:     {self.standard_epoch}",
            f"standard_warmup_steps: {self.standard_warmup_steps}",
            f"batch_size:         {self.batch_size}",
            f"min_lr:             {self.min_lr:.1e}",
            f"grad_clip_max_norm: {self.grad_clip_max_norm:.1f}",
            f"use_amp:            {self.use_amp}",
            f"use_compile:        {self.use_compile}",
        ]

        data_params = [
            f"task:               {self.task}",
            f"max_level:          {self.max_level}",
            f"random_seq_len:     {self.random_seq_len}",
            f"number_range:       {self.number_range}",
        ]

        model_params = [
            f"dim:                {self.dim}",
            f"n_layers:           {self.n_layers}",
            f"n_heads:            {self.n_heads}",
            f"hidden_dim:         {self.hidden_dim}",
        ]

        rla_params = [
            f"deterministic:      {self.deterministic}",
            f"sketch_mode:        {self.sketch_mode}",
            "",
            "  # For Attention Linear Layers (Wq, Wk, Wv, Wo):",
            f"  attn_qkv_sample_exact_dim: {self.rla_attn_qkv_sample_exact_dim}",
            f"  attn_qkv_projection_dim: {self.rla_attn_qkv_projection_dim}",
            f"  attn_out_sample_exact_dim: {self.rla_attn_out_sample_exact_dim}",
            f"  attn_out_projection_dim: {self.rla_attn_out_projection_dim}",
            "",
            "  # For Feed-Forward Network (FFN) Linear Layers:",
            f"  ffn_in_sample_exact_dim:   {self.rla_ffn_in_sample_exact_dim}",
            f"  ffn_in_projection_dim:   {self.rla_ffn_in_projection_dim}",
            f"  ffn_out_sample_exact_dim:  {self.rla_ffn_out_sample_exact_dim}",
            f"  ffn_out_projection_dim:  {self.rla_ffn_out_projection_dim}",
            "",
            "  # For Scaled Dot-Product Attention (SDPA) internal matmuls:",
            f"  sdpa_qk_sample_exact_dim: {self.rla_sdpa_qk_sample_exact_dim}",
            f"  sdpa_qk_projection_dim: {self.rla_sdpa_qk_projection_dim}",
            f"  sdpa_sv_sample_exact_dim: {self.rla_sdpa_sv_sample_exact_dim}",
            f"  sdpa_sv_projection_dim: {self.rla_sdpa_sv_projection_dim}",
        ]

        save_params = [
            f"save_path:          {self.save_path}",
            f"final_save_path:    {self.final_save_path}",
        ]

        sections = [
            ("Training Parameters", training_params),
            ("Data Parameters", data_params),
            ("Model Architecture Parameters", model_params),
            ("RLA Parameters", rla_params),
            ("Save Path Parameters", save_params),
        ]

        output_str = "Args Configuration:\n" # Renamed from 'output' to avoid conflict
        for header, params_list in sections: # Renamed params to params_list
            output_str += f"\n{header}:\n"
            # For RLA params, print them directly as they are already formatted with comments and spacing
            if header == "RLA Parameters":
                for param_line in params_list:
                    output_str += f"  {param_line}\n" # Add indent and newline
            else:
                output_str += "\n".join([f"  {param_line}" for param_line in params_list]) # param_line was param
                output_str += "\n"
        
        return output_str.rstrip()

def str_to_bool(v):
    """Convert string input to boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (e.g., True, False, yes, no, 1, 0).')

def parse_args():
    """Parse command-line arguments and return an Args object."""
    parser = argparse.ArgumentParser(description="Training script with configurable parameters.")

    # Grouping arguments
    training_group = parser.add_argument_group("Training Parameters")
    data_group = parser.add_argument_group("Data Parameters")
    model_group = parser.add_argument_group("Model Architecture Parameters")
    rla_group = parser.add_argument_group("RLA Parameters") # Added RLA group
    save_group = parser.add_argument_group("Save Path Parameters")

    # Training Parameters
    training_group.add_argument("--model_type", type=str, default="node",
                                help="Model type: 'transformer', 'node' or 'looped_transformer' (default: node)")
    training_group.add_argument("--standard_lr", type=float, default=2.5e-4,
                                help="Base learning rate (default: 2.5e-4)")
    training_group.add_argument("--standard_epoch", type=int, default=20000,
                                help="Number of epochs scaled by batch size (default: 20000)")
    training_group.add_argument("--standard_warmup_steps", type=int, default=4000,
                                help="Warmup steps scaled by batch size (default: 4000)")
    training_group.add_argument("--batch_size", type=int, default=256,
                                help="Batch size (default: 256)")
    training_group.add_argument("--min_lr", type=float, default=1e-4,
                                help="Minimum learning rate (default: 1e-4)")
    training_group.add_argument("--grad_clip_max_norm", type=float, default=5.0,
                                help="Maximum gradient norm for clipping (default: 5.0)")
    training_group.add_argument("--use_amp", type=str_to_bool, default=False,
                                help="Use automatic mixed precision (default: False)")
    training_group.add_argument("--use_compile", type=str_to_bool, default=False,
                                help="Use torch.compile (default: False)")

    # Data Parameters
    data_group.add_argument("--task", type=str, default="number_add",
                            help="Task type: 'number_add' or 'number_copy' (default: number_add)")
    data_group.add_argument("--max_level", type=int, default=40,
                            help="Maximum level for task complexity (default: 40)")
    data_group.add_argument("--random_seq_len", type=str_to_bool, default=True,
                            help="Use random sequence lengths (default: True)")
    data_group.add_argument("--number_range", type=int, nargs=2, default=[0, 99],
                            help="Range of numbers for the task (default: 0 99)")

    # Model Architecture Parameters
    model_group.add_argument("--dim", type=int, default=24,
                             help="Model dimension (default: 24)")
    model_group.add_argument("--n_layers", type=int, default=2,
                             help="Number of layers (default: 2)")
    model_group.add_argument("--n_heads", type=int, default=4,
                             help="Number of attention heads (default: 4)")
    model_group.add_argument("--hidden_dim", type=int, default=84,
                             help="Hidden dimension (default: 84)")

    # RLA Parameters - Added arguments
    rla_group.add_argument("--sketch_mode", type=str, default='rademacher', choices=['rademacher', 'gaussian'],
                           help="Type of random projection sketch (default: rademacher)")
    rla_group.add_argument("--deterministic", type=str_to_bool, default=False,
                           help="Globally disable RLA and use standard layers/ops (default: False)")
    rla_group.add_argument("--attn_qkv_sketch_size", type=int, default=0,
                           help="Sketch size for Wq, Wk, Wv projections in Attention (0 = deterministic, default: 0)")
    rla_group.add_argument("--attn_out_sketch_size", type=int, default=0,
                           help="Sketch size for Wo projection in Attention (0 = deterministic, default: 0)")
    rla_group.add_argument("--ffn_in_sketch_size", type=int, default=0,
                           help="Sketch size for first/third FFN projection (0 = deterministic, default: 0)")
    rla_group.add_argument("--ffn_out_sketch_size", type=int, default=0,
                           help="Sketch size for second FFN projection (0 = deterministic, default: 0)")
    rla_group.add_argument("--attn_score_sketch_size", type=int, default=0,
                           help="Sketch size for QK^T in SDPA (0 = deterministic, default: 0)")
    rla_group.add_argument("--attn_sum_sketch_size", type=int, default=0,
                           help="Sketch size for Scores@V in SDPA (0 = deterministic, default: 0)")


    # Save Path Parameters
    save_group.add_argument("--save_path", type=str, default="",
                            help="Path to save checkpoints (default: '')")
    save_group.add_argument("--final_save_path", type=str, default="",
                            help="Path to save final model (default: '')")

    # Parse arguments
    args = parser.parse_args()

    # Convert number_range to tuple
    number_range_tuple = tuple(args.number_range)

    # Create Args object with parsed values, including RLA parameters
    args_obj = Args(
        model_type=args.model_type,
        standard_lr=args.standard_lr,
        standard_epoch=args.standard_epoch,
        standard_warmup_steps=args.standard_warmup_steps,
        batch_size=args.batch_size,
        min_lr=args.min_lr,
        grad_clip_max_norm=args.grad_clip_max_norm,
        use_amp=args.use_amp,
        use_compile=args.use_compile,
        task=args.task,
        max_level=args.max_level,
        random_seq_len=args.random_seq_len,
        number_range=number_range_tuple,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        hidden_dim=args.hidden_dim,
        # RLA parameters added here
        sketch_mode=args.sketch_mode,
        deterministic=args.deterministic,
        attention_qkv_sketch_size=args.attn_qkv_sketch_size,
        attention_out_sketch_size=args.attn_out_sketch_size,
        feedforward_sketch_size_in=args.ffn_in_sketch_size,
        feedforward_sketch_size_out=args.ffn_out_sketch_size,
        attention_score_sketch_size=args.attn_score_sketch_size,
        attention_weighed_sum_sketch_size=args.attn_sum_sketch_size,
        # End RLA parameters
        save_path=args.save_path,
        final_save_path=args.final_save_path,
    )

    return args_obj