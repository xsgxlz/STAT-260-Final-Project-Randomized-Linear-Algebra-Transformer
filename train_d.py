# %%
import os
import time
import random
import copy
import sys
import math

import numpy as np
import torch

sys.path.append("/accounts/grad/zhangyunzhe2023/stat 260/STAT-260-Final-Project-Randomized-Linear-Algebra-Transformer")
from RLALLaMA3.LLaMA3 import ModelArgs, Transformer
from RLALLaMA3.utils import (
    linear_warmup_cosine_decay_multiplicative,
    name_args,
    Args,
)
from RLALLaMA3.tasks import single_answer_seq_loss, get_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

time_stamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
print(f"Time stamp: {time_stamp}")

# %%
# Define the arguments
##args = parse_args()
dim = 256
hidden_dim = 896
n_heads = 4
sample_ratio = 0.75
projection_ratio = 0.5
sv_sample_exact_dim = 48
sv_projection_dim = 8

head_dim = dim // n_heads

args = Args(
    # Training parameters
    standard_lr=1e-3,
    standard_epoch=100000,
    standard_warmup_steps=2000,
    batch_size=2048,
    min_lr=1e-4,
    grad_clip_max_norm=1.0,
    use_amp=True,
    use_compile=True,
    model_type="transformer",

    # Data parameters
    task="number_add",
    max_level=20,
    random_seq_len=True,
    number_range=(0, 99),

    # Model architecture parameters
    dim=dim,
    n_layers=4,
    n_heads=n_heads,
    hidden_dim=hidden_dim,

    # --- Randomized Linear Algebra (RLA) Parameters ---
    deterministic=True,
    sketch_mode='rademacher',

    # RLA for Attention Linear Layers (Wq, Wk, Wv, Wo)
    # dim
    rla_attn_qkv_sample_exact_dim=round(dim * sample_ratio),
    rla_attn_qkv_projection_dim=round((dim - round(dim * sample_ratio)) * projection_ratio),
    # dim
    rla_attn_out_sample_exact_dim=round(dim * sample_ratio),
    rla_attn_out_projection_dim=round((dim - round(dim * sample_ratio)) * projection_ratio),

    # RLA for Feed-Forward Network (FFN) Linear Layers
    # dim
    rla_ffn_in_sample_exact_dim=round(dim * sample_ratio),
    rla_ffn_in_projection_dim=round((dim - round(dim * sample_ratio)) * projection_ratio),
    # hidden_dim
    rla_ffn_out_sample_exact_dim=round(hidden_dim * sample_ratio),
    rla_ffn_out_projection_dim=round((hidden_dim - round(hidden_dim * sample_ratio)) * projection_ratio),

    # RLA for Scaled Dot-Product Attention (SDPA) internal matmuls
    # head_dim
    rla_sdpa_qk_sample_exact_dim=round(head_dim * sample_ratio),
    rla_sdpa_qk_projection_dim=round((head_dim - round(head_dim * sample_ratio)) * projection_ratio),
    # seq_len
    rla_sdpa_sv_sample_exact_dim=sv_sample_exact_dim,
    rla_sdpa_sv_projection_dim=sv_projection_dim,

    # Save path
    save_path="ckpt",
    final_save_path="ckpt_final",
)


print(args, end="\n\n")

# %%
# Prepare the data
dataset, collate_fn, vocab_size, max_seq_len = get_dataset(args.task,
                                                           args.max_level,
                                                           args.random_seq_len,
                                                           args.number_range,
                                                           nested_tensor=False,
                                                           pad_to_longest=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                         num_workers=torch.get_num_threads(), pin_memory=True)

# %%
def mean_seq_len(dataloader, num_samples=100):
    """
    Calculate the mean sequence length of the dataset.
    """
    total_len = 0
    num_samples = min(num_samples, len(dataloader.dataset))
    for i, x in enumerate(dataloader):
        if i >= num_samples:
            break
        total_len += x[1].float().mean().item()
    return total_len / num_samples

mean_len = mean_seq_len(dataloader)
print(f"Mean sequence length: {mean_len}")
print(f"Max sequence length: {max_seq_len}")

# %%
# Prepare the model

transformer_args = ModelArgs(
    # Standard model parameters from Args
    dim=args.dim,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    hidden_dim=args.hidden_dim, # ModelArgs.__post_init__ might adjust this if None
    vocab_size=vocab_size,      # Assumed to be defined in the scope
    max_seq_len=max_seq_len,    # Assumed to be defined in the scope

    # Parameters that might be hardcoded or could be added to Args if needed
    norm_eps=1e-5,              # Default in ModelArgs, can be overridden
    rope_theta=500000.0,        # Default in ModelArgs, can be overridden
    # n_kv_heads=args.n_kv_heads, # If you add n_kv_heads to Args and ModelArgs

    # RLA parameters from Args
    deterministic=args.deterministic,
    sketch_mode=args.sketch_mode,

    # RLA for Attention Linear Layers
    rla_attn_qkv_sample_exact_dim=args.rla_attn_qkv_sample_exact_dim,
    rla_attn_qkv_projection_dim=args.rla_attn_qkv_projection_dim,
    rla_attn_out_sample_exact_dim=args.rla_attn_out_sample_exact_dim,
    rla_attn_out_projection_dim=args.rla_attn_out_projection_dim,

    # RLA for Feed-Forward Network (FFN) Linear Layers
    rla_ffn_in_sample_exact_dim=args.rla_ffn_in_sample_exact_dim,
    rla_ffn_in_projection_dim=args.rla_ffn_in_projection_dim,
    rla_ffn_out_sample_exact_dim=args.rla_ffn_out_sample_exact_dim,
    rla_ffn_out_projection_dim=args.rla_ffn_out_projection_dim,

    # RLA for Scaled Dot-Product Attention (SDPA) internal matmuls
    rla_sdpa_qk_sample_exact_dim=args.rla_sdpa_qk_sample_exact_dim,
    rla_sdpa_qk_projection_dim=args.rla_sdpa_qk_projection_dim,
    rla_sdpa_sv_sample_exact_dim=args.rla_sdpa_sv_sample_exact_dim,
    rla_sdpa_sv_projection_dim=args.rla_sdpa_sv_projection_dim,
)

model = Transformer(params=transformer_args)

model = model.to(device).train()

# %%
model

# %%
# Training configuration
standard_lr = args.standard_lr / 512
standard_epoch = args.standard_epoch * 512
standard_warmup_steps = args.standard_warmup_steps * 512
batch_size = args.batch_size

lr = standard_lr * batch_size
warmup_steps = standard_warmup_steps // batch_size
epochs = standard_epoch // batch_size

print("Derived Parameters:")
print(f"lr: {lr}")
print(f"warmup_steps: {warmup_steps}")
print(f"epochs: {epochs}")
print(f"grad_clip_max_norm: {args.grad_clip_max_norm}", end="\n\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
            lr_lambda=lambda step: linear_warmup_cosine_decay_multiplicative(step, warmup_steps, epochs, args.min_lr))

scaler = torch.amp.GradScaler(device, enabled=args.use_amp)

# %%
# Save the model and arguments

def save_record(path, model, record, args, time_stamp, extra_info=None):
    dict_name, file_name = name_args(args, "_")

    os.makedirs(f"{path}/{dict_name}", exist_ok=True)
    file_name = file_name + f"_{time_stamp}"
    if extra_info is not None:
        file_name += f"_{extra_info}"
    
    record_dict = {
        "model": model,
        "record": record,
        "args": args,
        "time_stamp": time_stamp,
    }
        
    torch.save(record_dict, f"{path}/{dict_name}/{file_name}.pth")

# %%
# Backwards pass
def backward_pass(model, loss, optimizer, scaler, scheduler, grad_clip_max_norm):
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

# %%
def transformer_output_error(transformer_output, gt_transformer_output, pad_mask):
    """
    Calculate the error between the transformer output and the ground truth output.
    """
    # Apply the padding mask to both outputs
    pad_mask = pad_mask.unsqueeze(-1)
    transformer_output = transformer_output * pad_mask
    gt_transformer_output = gt_transformer_output * pad_mask

    # Calculate the mean squared error
    abs_error = (transformer_output - gt_transformer_output).square().sum()
    relative_error = abs_error / gt_transformer_output.square().sum()
    num_tokens = pad_mask.sum()
    abs_error = abs_error / num_tokens / transformer_output.size(-1)
    return abs_error, relative_error

# %%
@torch.compile(disable=not args.use_compile)
def train_step(model, train_data, mean_len, optimizer, scheduler, scaler, args, calculate_errors=False):
    device = train_data[0].device
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.use_amp):
        tokens, lengths, ans_starts, ans_lengths = train_data

        if calculate_errors:
            with torch.no_grad():
                pad_mask = tokens[:, 1:] == 0
                model.deterministic_mode(True)
                _, gt_transformer_output = model(tokens[:, :-1], return_transformer_output=True)
                model.deterministic_mode(args.deterministic)

        pred, transformer_output = model(tokens[:, :-1], return_transformer_output=True)
        transformer_output = transformer_output.detach()

        if calculate_errors:
            with torch.no_grad():
                abs_error, relative_error = transformer_output_error(transformer_output, gt_transformer_output, pad_mask)
                abs_error, relative_error = abs_error.detach(), relative_error.detach()
        
        result = single_answer_seq_loss(pred, tokens, lengths, ans_starts, ans_lengths)
        GPT_loss, full_seq_acc, ans_region_acc, ans_char_acc = result
        # Normalize the GPT loss by the batch size but not the sequence length
        GPT_loss = GPT_loss / args.batch_size
        total_loss = GPT_loss
        total_loss_for_backward = total_loss / mean_len
    
    if torch.isnan(total_loss) or torch.isinf(total_loss):# or (total_loss > smoothed_loss * 1.1):
        return [total_loss]
    
    with torch.no_grad():
        safe_params = [copy.deepcopy(i.state_dict()) for i in [model, optimizer, scheduler]]

    backward_pass(model, total_loss_for_backward, optimizer, scaler, scheduler, args.grad_clip_max_norm)
    
    data = [GPT_loss, 0, total_loss, full_seq_acc, ans_region_acc, ans_char_acc]

    if calculate_errors:
        data = data + [abs_error, relative_error]
    else:
        data = data + [0, 0]
        
    with torch.inference_mode():
        data = torch.tensor(data).cpu().numpy()

    return data, safe_params

# %%
record = np.zeros((epochs, 9))
num_NaNs = 0
smoothed_loss = None

safe_params = [copy.deepcopy(i.state_dict()) for i in [model, optimizer, scheduler]]

epoch = 0

for train_data in dataloader:
    if epoch >= epochs:
        break

    train_data = [x.to(device) for x in train_data]

    t0 = time.time()

    result = train_step(model, train_data, mean_len, optimizer, scheduler, scaler, args, calculate_errors=not args.deterministic)

    if len(result) == 1:
        data = result
        total_loss = data[0]
        num_NaNs += 1
        print(f"Epoch: {epoch}")
        print("Instability detected")
        print(f"Total Loss: {total_loss.item()}\n")
        model.load_state_dict(safe_params[0])
        optimizer.load_state_dict(safe_params[1])
        scheduler.load_state_dict(safe_params[2])
        optimizer.zero_grad(set_to_none=True)
        continue

    data, safe_params = result
    smoothed_loss = 0.99 * smoothed_loss + 0.01 * data[2].item() if smoothed_loss is not None else data[2].item()
    epoch = epoch + 1

    record[epoch - 1, :-1] = data
    record[epoch - 1, -1] = num_NaNs
        
    names = ["GPT loss", "Energy Reg", "Total_loss", "Full Seq Acc",
             "Ans Region Acc", "Ans Char Acc", "Abs Error", "Rel Error"]

    print(f"Epoch: {epoch}")
    for name, value in zip(names, data):
        print(f"{name}: {value}")
    print(f"Smoothed Loss: {smoothed_loss}")
    print(f"Time: {time.time() - t0}\n")

    if epoch % 10 == 0:
        save_record(args.save_path, model, record, args, time_stamp)
        pass


