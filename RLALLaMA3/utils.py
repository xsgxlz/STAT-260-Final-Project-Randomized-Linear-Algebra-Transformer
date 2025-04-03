import argparse
import math
import random
from dataclasses import dataclass

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

def transformer_forward_pass(tokens, model):
    return model(tokens)

def name_args(args, sep):
    dict_path_args = [args.task, args.max_level, args.random_seq_len, args.number_range]
    model_args = [args.model_type, args.dim, args.n_layers, args.n_heads, args.hidden_dim, args.batch_size]

    if args.model_type == "transformer":
        file_name_args = model_args
    elif args.model_type == "node":
        node_args = [args.range_max - args.range_min, args.tol_base, args.tol_ratio, args.energy_penalty]
        file_name_args = model_args + node_args
    elif args.model_type == "looped_transformer":
        looped_transformer_args = [args.n_loops, args.n_layers_per_loop, args.loop_step_method]
        file_name_args = model_args + looped_transformer_args

    dict_name = sep.join(map(str, dict_path_args))
    file_name = sep.join(map(str, file_name_args))

    return dict_name, file_name

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

    # Data parameters
    task: str = "number_add"
    max_level: int = 40
    random_seq_len: bool = True
    number_range: tuple[int, int] = (0, 99)

    # Model architecture parameters
    dim: int = 24
    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 84

    # Save path
    save_path: str = ""
    final_save_path: str = ""

    def __str__(self):
        """Return a formatted string representation of the Args object."""
        # Define groups of parameters
        training_params = [
            f"standard_lr:        {self.standard_lr:.1e}",
            f"standard_epoch:     {self.standard_epoch}",
            f"standard_warmup_steps: {self.standard_warmup_steps}",
            f"batch_size:         {self.batch_size}",
            f"min_lr:             {self.min_lr:.1e}",
            f"grad_clip_max_norm: {self.grad_clip_max_norm:.1f}",
            f"use_amp:            {self.use_amp}",
            f"use_compile:       {self.use_compile}",
        ]
        
        data_params = [
            f"task:              {self.task}",
            f"max_level:         {self.max_level}",
            f"random_seq_len:    {self.random_seq_len}",
            f"number_range:      {self.number_range}",
        ]
        
        model_params = [
            f"dim:               {self.dim}",
            f"n_layers:          {self.n_layers}",
            f"n_heads:           {self.n_heads}",
            f"hidden_dim:        {self.hidden_dim}",
        ]
        
        save_params = [
            f"save_path:         {self.save_path}",
            f"final_save_path:   {self.final_save_path}",
        ]

        # Combine sections with headers
        sections = [
            ("Training Parameters", training_params),
            ("Data Parameters", data_params),
            ("Model Architecture Parameters", model_params),
            ("Save Path Parameters", save_params),
        ]
        
        # Build the formatted string
        output = "Args Configuration:\n"
        for header, params in sections:
            output += f"\n{header}:\n"
            output += "\n".join([f"  {param}" for param in params])
            output += "\n"
        
        return output.rstrip()  # Remove trailing newline

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

    # Save Path Parameters
    save_group.add_argument("--save_path", type=str, default="",
                            help="Path to save checkpoints (default: '')")
    save_group.add_argument("--final_save_path", type=str, default="",
                            help="Path to save final model (default: '')")

    # Parse arguments
    args = parser.parse_args()

    # Convert number_range to tuple
    number_range_tuple = tuple(args.number_range)

    # Create Args object with parsed values
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
        save_path=args.save_path,
        final_save_path=args.final_save_path,
    )

    return args_obj
