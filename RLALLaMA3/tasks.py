import sys
from typing import Callable, Any, Tuple
import functools
import math

import numpy as np
import torch
import torch.nn.functional as F

class CharacterTokenizer:
    """A character-level tokenizer for encoding and decoding text with special tokens."""
    def __init__(self, vocab: list):
        """
        Initialize the tokenizer with a vocabulary and special tokens.

        Args:
            vocab (list): List of characters to include in the vocabulary.
        """
        # Add special tokens for Start of Sequence (SOS), End of Sequence (EOS), and Padding (PAD)
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.vocab = [self.pad_token, self.sos_token, self.eos_token] + vocab
        
        self.vocab_size = len(self.vocab)

        # Create a direct mapping using a NumPy array for ASCII-based indexing
        ASCIIs = [ord(char) for char in vocab]
        self.char_to_index = np.full(128, -1, dtype=np.int32)  # Assuming standard ASCII range (0-127)
        self.char_to_index[ASCIIs] = np.arange(3, self.vocab_size)  # Offset for PAD, SOS, EOS

        # Define indices for special tokens
        self.pad_index = 0
        self.sos_index = 1
        self.eos_index = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Encode a string into a tensor of token indices.

        Args:
            text (str): Input text to encode.
            add_special_tokens (bool): Whether to add SOS and EOS tokens. Defaults to True.

        Returns:
            torch.Tensor: Tensor of token indices (long type).
        """
        indices = [self.char_to_index[ord(c)] for c in text]
        if -1 in indices:
            raise ValueError("Input text contains characters not in the vocabulary.")
        if add_special_tokens:
            indices = [self.sos_index] + indices + [self.eos_index]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tensor: torch.Tensor, remove_special_tokens: bool = True) -> str:
        """
        Decode a tensor of indices back into a string.

        Args:
            tensor (torch.Tensor): Tensor of token indices.
            remove_special_tokens (bool): Whether to remove special tokens. Defaults to True.

        Returns:
            str: Decoded string.
        """
        indices = tensor.tolist()
        if remove_special_tokens:
            indices = [i for i in indices if i not in {self.sos_index, self.eos_index, self.pad_index}]
        return ''.join([self.vocab[i] for i in indices])
    
class SyntheticDataset(torch.utils.data.Dataset):
    """A dataset class for generating synthetic data on-the-fly or pre-generating a fixed-size dataset."""
    
    def __init__(self, generate_fn: callable, generate_fn_args: dict, tokenizer: callable, length: int = sys.maxsize):
        """
        Initialize the synthetic dataset.

        Args:
            generate_fn (callable): Function to generate data samples.
            generate_fn_args (dict): Arguments to pass to the generate function.
            tokenizer (callable): Tokenizer instance to encode the generated strings.
            length (int): Number of samples in the dataset. Defaults to sys.maxsize (infinite).
        """
        self.generate_fn = generate_fn
        self.generate_fn_args = generate_fn_args
        self.tokenizer = tokenizer
        self.length = length
        
        # If a finite length is specified, pre-generate the dataset
        if length < sys.maxsize:
            self.data = [self.generate_fn(i, **self.generate_fn_args) for i in range(length)]
            # Tokenize the pre-generated data
            self.data = [(self.tokenizer.encode(data[0]), data[1], data[2], data[3]) for data in self.data]
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.length
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (encoded_tensor, length, ans_start, ans_length)
        """
        if self.length < sys.maxsize:
            return self.data[idx]
        else:
            # Generate on-the-fly for infinite datasets
            string, length, ans_start, ans_length = self.generate_fn(idx, **self.generate_fn_args)
            return self.tokenizer.encode(string), length + 1, ans_start + 1, ans_length
        
def number_copy_task(idx: int, number_range: tuple[int, int], sequence_length: int) -> tuple[str, int, int, int]:
    """
    Generate a number copy task sample where the input sequence is repeated after a separator.

    Args:
        idx (int): Index (unused, for compatibility with dataset).
        number_range (tuple[int, int]): Range of numbers to generate (min, max).
        sequence_length (int): Length of the number sequence.

    Returns:
        tuple: (string, total_length, answer_start, answer_length)
    """
    numbers = np.random.randint(number_range[0], number_range[1], sequence_length).tolist()
    numbers_str = ','.join(map(str, numbers))
    # Format: "numbers|numbers" (input|output)
    return f'{numbers_str}|{numbers_str}', len(numbers_str) * 2 + 1, len(numbers_str) + 1, len(numbers_str)

def number_add_task(idx: int, number_range: tuple[int, int], sequence_length: int) -> tuple[str, int, int, int]:
    """
    Generate a number addition task sample where numbers are summed, e.g., "11+22+33=66".

    Args:
        idx (int): Index (unused, for compatibility with dataset).
        number_range (tuple[int, int]): Range of numbers to generate (min, max).
        sequence_length (int): Number of terms to add in the sequence.

    Returns:
        tuple: (string, total_length, answer_start, answer_length)
            - string: The full task string (e.g., "11+22+33=66").
            - total_length: Length of the entire string.
            - answer_start: Index where the answer begins (after '=').
            - answer_length: Length of the answer portion.
    """
    # Generate random numbers within the specified range
    numbers = np.random.randint(number_range[0], number_range[1], sequence_length).tolist()
    
    # Calculate the sum of the numbers
    total = sum(numbers)
    
    # Create the input part (e.g., "11+22+33")
    input_str = '+'.join(map(str, numbers))
    
    # Create the full string with the answer (e.g., "11+22+33=66")
    full_str = f"{input_str}={total}"
    
    # Calculate lengths and positions
    total_length = len(full_str)
    answer_start = len(input_str) + 1  # Position after '='
    answer_length = len(str(total))    # Length of the answer digits
    
    return full_str, total_length, answer_start, answer_length

def make_random_length_task(
    task_fn: Callable[..., Tuple[Any, ...]], 
    sequence_length_dist: Any
) -> Callable[..., Tuple[Any, ...]]:
    """
    Create a wrapper function that randomizes the sequence length for a given task function.

    Args:
        task_fn (Callable): The original task function to wrap. It must accept a sequence_length
                           parameter and return a tuple.
        sequence_length_dist: Distribution object with a .sample() method (e.g., torch.distributions)
                             to sample sequence lengths.

    Returns:
        Callable: A wrapped function that randomizes sequence_length and calls task_fn with it.
    """
    def wrapped_task(*args, **kwargs) -> Tuple[Any, ...]:
        """
        Wrapped task function that injects a random sequence length.

        Args:
            *args: Positional arguments to pass to task_fn (e.g., idx, number_range).
            **kwargs: Keyword arguments to pass to task_fn, with sequence_length overridden.

        Returns:
            Tuple: The output of task_fn with a randomized sequence_length.
        """
        # Sample sequence length and ensure it’s an integer (add 1 to match original behavior)
        sequence_length = int(sequence_length_dist.sample()) + 1
        # Pass the random sequence_length to the task function, preserving other args
        return task_fn(*args, sequence_length=sequence_length, **kwargs)

    return wrapped_task

def padding_collate_fn(batch, nested_tensor=False, minimal_seq_len=None) -> tuple:
    """
    Collate function to pad sequences in a batch and convert metadata to tensors.

    Args:
        batch: List of (sequence, length, ans_start, ans_length) tuples.

    Returns:
        tuple: (padded_sequences, lengths_tensor, ans_starts_tensor, ans_lengths_tensor)
    """
    if nested_tensor and minimal_seq_len is not None:
        raise ValueError("minimal_seq_len is not supported for nested tensors.")
    sequences, lengths, ans_starts, ans_lengths = zip(*batch)
    if not nested_tensor:
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        seq_len = sequences.size(1)
        if minimal_seq_len is not None and seq_len < minimal_seq_len:
            sequences = F.pad(sequences, (0, minimal_seq_len - seq_len), value=0)
    else:
        sequences = torch.nested.nested_tensor(sequences, layout=torch.jagged)
    lengths = torch.tensor(lengths)
    ans_starts = torch.tensor(ans_starts)
    ans_lengths = torch.tensor(ans_lengths)
    return sequences, lengths, ans_starts, ans_lengths

def masked_acc_loss(
    pred: torch.Tensor,
    tokens: torch.Tensor,
    starts: torch.Tensor | int,
    lengths: torch.Tensor | int,
    require_loss: bool
) -> list[torch.Tensor]:
    """
    Compute accuracy and optionally loss over a masked region of predictions.

    Args:
        pred (torch.Tensor): Predictions tensor of shape (batch_size, seq_len, vocab_size).
        tokens (torch.Tensor): Ground truth tokens of shape (batch_size, seq_len).
        starts (torch.Tensor | int): Start indices of the region to evaluate.
        lengths (torch.Tensor | int): Lengths of the regions to evaluate.
        require_loss (bool): Whether to compute and return the loss.

    Returns:
        list[torch.Tensor]: [all_correct_acc, overall_acc, (loss if require_loss=True)]
    """
    bsz, max_seq_len, vocab_size = pred.size()
    mask = torch.arange(max_seq_len, device=pred.device).expand(bsz, max_seq_len)

    # Create masks for regions before and after the target area
    if isinstance(starts, int):
        mask_before = mask < starts - 1
        mask_after = mask > (starts + lengths - 2) if isinstance(lengths, int) else mask > (starts - 2 + lengths).unsqueeze(1)
    else:
        mask_before = mask < starts.unsqueeze(1) - 1
        mask_after = mask > (lengths - 2 + starts).unsqueeze(1)
    
    mask = mask_before | mask_after  # Mask out non-target regions

    # Compute overall accuracy (all tokens correct in sequence)
    result = pred.argmax(dim=-1)
    all_correct_acc = ((result == tokens) | mask).all(dim=-1).float().mean()

    # Compute accuracy only in the target region
    mask = ~mask
    pred_flat = pred.view(-1, vocab_size)
    tokens_flat = tokens.reshape(-1)
    mask_flat = mask.view(-1)

    pred = pred_flat[mask_flat]
    tokens = tokens_flat[mask_flat]
    correctness = pred.argmax(dim=-1) == tokens
    overall_acc = correctness.float().mean()

    ret = [all_correct_acc, overall_acc]

    # Compute loss if requested
    if require_loss:
        #loss = F.cross_entropy(pred, tokens, label_smoothing=0.1)
        loss = F.cross_entropy(pred, tokens, label_smoothing=0.1, reduction="sum")
        ret.append(loss)
    
    return ret

def single_answer_seq_loss(pred, tokens, lengths, ans_starts, ans_lengths):
    """
    Compute loss and accuracy metrics for a sequence with a single continuous answer region.

    Args:
        pred (torch.Tensor): Model predictions (batch_size, seq_len, vocab_size).
        tokens (torch.Tensor): Ground truth tokens (batch_size, seq_len), excluding SOS.
        lengths (torch.Tensor): Total sequence lengths.
        ans_starts (torch.Tensor): Start indices of the answer region.
        ans_lengths (torch.Tensor): Lengths of the answer region.

    Returns:
        tuple: (loss, full_seq_acc, ans_region_acc, ans_char_acc)
            - loss: Cross-entropy loss over the full sequence.
            - full_seq_acc: Accuracy over the entire sequence (all correct).
            - ans_region_acc: Accuracy over the answer region (all correct).
            - ans_char_acc: Per-character accuracy in the answer region.
    """
    tokens = tokens[:, 1:]  # Remove SOS token
    _, full_seq_acc, loss = masked_acc_loss(pred, tokens, 1, lengths, require_loss=True)
    with torch.inference_mode():
        ans_region_acc, ans_char_acc = masked_acc_loss(pred, tokens, ans_starts, ans_lengths + 1, require_loss=False)
    return loss, full_seq_acc, ans_region_acc, ans_char_acc

def get_dataset(task, max_level, random_seq_len, number_range, nested_tensor=False, pad_to_longest=False):
    if nested_tensor and pad_to_longest:
        raise ValueError("pad_to_longest is not supported for nested tensors.")
    vocab = vocabs_for_tasks[task]
    tokenizer = CharacterTokenizer(vocab)
    vocab_size = len(vocab) + 3 # pad, eos, sos

    if task == "number_add":
        task_fun = number_add_task
    elif task == "number_copy":
        task_fun = number_copy_task

    max_seq_len = max_len_for_tasks[task](number_range, max_level)

    if task in ["number_add", "number_copy"]:
        generate_fn_args = {"number_range": number_range}

        if random_seq_len:
            seq_len_dist_logits = torch.distributions.categorical.Categorical(logits=torch.linspace(0, math.log(1), max_level))
            task_fun = make_random_length_task(task_fun, seq_len_dist_logits)
        else:
            generate_fn_args["sequence_length"] = max_level
        
        dataset = SyntheticDataset(task_fun, generate_fn_args, tokenizer)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    collate_fn = functools.partial(padding_collate_fn,
                                   nested_tensor=nested_tensor,
                                   minimal_seq_len=max_seq_len if pad_to_longest else None)

    return dataset, collate_fn, vocab_size, max_seq_len

vocabs_for_tasks = {
    "number_copy": list("0123456789,|"),
    "number_add": list("0123456789+=")
}

max_len_for_tasks = {
    "number_copy": lambda number_range, max_level: (len(str(number_range[1])) + 1) * max_level * 2 + 1,
    "number_add": lambda number_range, max_level: (len(str(number_range[1])) + 1) * max_level + len(str(number_range[1] * max_level)) + 2
}
    