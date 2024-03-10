"""
This file is written with the reference of convert.py from llama2.c
"""

import requests
from typing import List
import os
import argparse
from tqdm import tqdm
from model import Gemma, ModelConfig
import struct
import numpy as np
import torch


MODEL_2B_URL = "https://huggingface.co/GunA-SD/Hub/resolve/main/gemma/2b/gemma-2b.ckpt"
MODEL_7B_URL = "https://huggingface.co/GunA-SD/Hub/resolve/main/gemma/7b/gemma-7b.ckpt"

MODEL_PATH = os.path.dirname(__file__) + "/models"


def _download(url, download_folder='models', chunk_size=8192 * 4):

    filename = url.split('/')[-1]
    folder = os.path.join(os.path.dirname(__file__), download_folder)
    file_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    if os.path.exists(file_path):
        print(f"File '[{filename}]' already exists at {file_path}...😅")
        return file_path

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
              desc=f"Downloading: {filename}") as progress_bar:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    print("Download completed...😉")
    return file_path


def load_model(model : str = "2b"):
    if model == "2b":
        model_path = _download(MODEL_2B_URL)
    elif model == "7b":
        model_path = _download(MODEL_7B_URL)
        return Gemma(ModelConfig()).load_weights(model_path)

def _fp32(file, tensor):
    data = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    bindata = struct.pack(f"{len(data)}f",*data)
    file.write(bindata)


def _fp16(file, tensor):
    data = tensor.detach().cpu().view(-1).numpy().astype(np.float16)
    bindata = struct.pack(f"{len(data)}h",*data)
    file.write(bindata)

def _int8(file, tensor):
    data = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    bindata = struct.pack(f"{len(data)}b",*data)
    file.write(bindata)

def convert_to_bin(model, filepath, size = 'fp32'):
    file = open(filepath, 'wb')
    args = model.args
    headers = struct.pack('iiiiiii', args.dim, args.n_layers, args.n_heads, 
                          args.n_kv_heads, args.hidden_dim, args.head_dim,
                            args.vocab_size, args.max_seq_len, args.norm_eps)
    file.write(headers)
    if size == 'fp32':
        _fp32(file, model.embedder.weight)
        for layer in model.layers:
            _fp32(file, layer.input_layernorm.weight)
        for layer in model.layers:
            _fp32(file, layer.self_attn.qkv_proj.weight)
        for layer in model.layers:
            _fp32(file, layer.self_attn.o_proj.weight)
        for layer in model.layers:
            _fp32(file, layer.post_attention_layernorm.weight)
        for layer in model.layers:
            _fp32(file, layer.mlp.gate_proj.weight)
        for layer in model.layers:
            _fp32(file, layer.mlp.down_proj.weight)
        for layer in model.layers:
            _fp32(file, layer.mlp.up_proj.weight)
        _fp32(file, model.layernorm.weight)
        _fp32(file, model.freqs_cis)
        file.close()
        print(f"wrote model in .bin file {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelsize", type=str, default='2b', help="checkpoint size either 2b or 7b")
    parser.add_argument("--convert", type=str, default="bin" , help="convert binary use '--convert bin'")
    args = parser.parse_args()

    model = load_model(args.modelsize)
    if args.convert == "bin":
        convert_to_bin(os.path.join(os.path.dirname(__file__), "models"), model)
