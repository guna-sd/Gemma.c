''' This train code is written by reference of llama2.c train.py code'''

'''
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
'''

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
import json
import numpy as np
import wandb

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import Gemma, ModelConfig


out_dir : str = '/models/'
eval_interval : int = 2000
log_interval : int = 1
eval_iters : int= 100
eval_model : bool = False
always_save_checkpoint : bool = True
init_from : str = 'new'
device ='cuda' if torch.cuda.is_available() else 'cpu'
compile = True

#log
wandb_log : bool = False
wandb_file : str = 'gemmac'
wandb_run_name : str = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
wconfig = {k: globals()[k] for k in config_keys}

#data
batch_size : int = 128
max_seq_len : int = 1024
vocab_size : int = ModelConfig().vocab_size

#model
dim = 2048
n_layers = 18
n_heads = 16
n_kv_heads = 16
hidden_dim = 16384
head_dim: int = 256
norm_eps : float = 1e-6

#adam
gradient_accumulation_steps : int = 4
learning_rate : float = 5e-4
min_lr = 6e-5 
max_iters : int = 100000
beta1 : float = 0.9
beta2 : float = 0.95
grad_clip : float = 1.0
decay_lr : bool = True
warmup_iters : int = 1000
lr_decay_iters = 600000 
max_batch_size=48
block_size = 1024
best_val_loss = 1e9
running_mfu = -1.0
dtype : str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ddp = int(os.environ.get("RANK", -1)) != -1 


model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    head_dim = head_dim,
)

train_data = np.memmap('prepare/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('prepare/val.bin', dtype=np.uint16, mode='r')

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)


def get_batch(split : str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_lr(iter):
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    if iter > lr_decay_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def write_params(params: dict, path: str):
    with open(os.path.join(path,'params.json'), 'w') as file:
        json.dump(params, file)

def read_params(path: str) -> dict:
    try:
        with open(path, 'r') as file:
            params = json.load(file)
        return params
    except Exception as e:
        print(f"Error reading parameters from {path}: {e}")
        return None

def get_batch_size():
    pass 
  # not yet implemented ---> will be later...


class GemmaTrainer:
    def __init__(self):
        self.config, self.model = self.init_model()
        self.optimizer = torch.optim.AdamW(self.model.parameter(), lr=learning_rate, betas=(beta1, beta2) ,)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))        
 
    def init_model(self, init_from : str) ->  Gemma:
        if init_from == 'new':
            print("Initializing New Model...")
            write_params(model_args, out_dir)
            config = ModelConfig(**model_args)
            model = Gemma(config)

        elif init_from == 'resume':
            print(f"Resuming training from {out_dir}...")
            ckpt_path = os.path.join(out_dir, 'model.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            ckpt_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_kv_heads', 'head_dim', 'max_seq_len', 'vocab_size', 'dim']:
                model_args[k] = ckpt_model_args[k]
            config = ModelConfig(**model_args)
            model = Gemma(config)
            ckpt = checkpoint['model']
            prefix = "_orig_mod."
            ckpt = {key[len(prefix):]: value for key, value in ckpt.items() if key.startswith(prefix)}
            model.load_state_dict(ckpt)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            del ckpt
        return config, model.to(device)
    
    def save_ckpt(self, step:int = None) -> None:
        if step != None:
            ckpt_name = f'ckpt{step}.pt'
            ckpt = {
                    'model': self.model.state_dict(),
                    'model_args': model_args,
                    'iter_num': step,
                    'best_val_loss': best_val_loss,
                    'config': self.config,
            }
            torch.save(obj=ckpt,f = os.path.join(out_dir,ckpt_name))
            print(f'{ckpt_name} model saved at {out_dir}...')

        else:
            ckpt_name = f'ckpt.pt'
            torch.save(obj=self.model.state_dict(),f = os.path.join(out_dir,ckpt_name))
            print(f'{ckpt_name} model saved at {out_dir}...')
    
    def compile(self):
        if compile:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)
        
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def start_wandb():
        wandb.init(project=wandb_file, name=wandb_run_name, config=wconfig)
    
    def train(self):
        X, Y = get_batch('train')
        t0 = time.time()
        estimate_iter : int = 0
        iter_num : int = 0
        while True:
            learning_rate = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['learning_rate'] = learning_rate
            if iter_num % eval_iters == 0:
                loss = self.estimate_loss()
                print(f"step {iter_num}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")
                if loss['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = loss['val']
                    if iter_num > 0:
                        self.save_ckpt(step=iter_num)
            if iter_num == 0 and eval_model:
                break

            for step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = self.model(X,Y)
                    loss = loss / gradient_accumulation_steps
                X,Y = get_batch('train')
                self.scaler.scale(loss).backward()
            if grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0:
                lossf = loss.item() * gradient_accumulation_steps
                if estimate_iter >=5:
                    mfu = self.model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            estimate_iter += 1
            if iter_num > max_iters:
                print("training completed...")
                break
