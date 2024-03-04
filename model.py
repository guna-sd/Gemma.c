from dataclasses import dataclass
from typing import Tuple , List
from .tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from torch import nn
import argparse

@dataclass
class ModelConfig:
    def __init__(self, **kwargs):
    # default parameters for Gemma 7b model
        self.dim: int = kwargs.get('dim', 3072) # 2048 for 2b model
        self.n_layer: int = kwargs.get('n_layer', 28) # 18 for 2b model
        self.n_heads: int = kwargs.get('n_heads', 16) # 8 for 2b model
        self.n_kv_heads: int = kwargs.get('n_kv_heads', 16) # 1 for 2b model
        self.vocab_size: int = kwargs.get('vocab_size', 256000)
        self.max_seq_len : int = kwargs.get('max_seq_len', 8192)
        self.norm_eps : float = kwargs.get('norm_eps',1e-6)
        self.hidden_dim : int = kwargs.get('hidden_dim', 24576) # 16384 for 2b model
        self.head_dim : int = kwargs.get('head_dim', 256)

class Sampler(nn.Module):
    def __init__(self, vocab_size : int):
        super().__init__()
    @torch.no_grad()
    def forward(
        self,
        embedding : torch.Tensor,
        x : torch.Tensor,
        output_pos : torch.Tensor,
        temperature : torch.Tensor,
        topp : torch.Tensor,
        topk : torch.Tensor,
        embedding_bias : torch.Tensor = None
    ) -> torch.Tensor :
        x = x.index_select(
            1, output_pos).squeeze(dim=1)
        logits = torch.matmul(x, embedding.t())
        if embedding_bias is not None:
            logits = logits + embedding_bias
        if temperature is not None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)
        logits.div_(temperature.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort , probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        topp_mask = (probs_sum - probs_sort) > topp.unsqueeze(dim=1)
        probs_sort = torch.where(topp_mask, 0, probs_sort)
        topk_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        topk_mask = topk_mask.expand(probs_idx.shape[0], -1)
        topk_mask = topk_mask >= topk.unsqueeze(dim=1)
        probs_sort = torch.where(topk_mask, 0, probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_sort, dim=-1))

        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)
        return next_token_ids

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool = False):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output
    
class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True,):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out

class Attention(nn.Module):
    def __init__(self, args : ModelConfig):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.dim = args.dim
        self.scale = self.head_dim**-0.5

        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.dim, (self.n_heads + 2*self.n_kv_heads)*self.head_dim,bias=False)
        self.out_proj = nn.Linear(self.n_heads*self.head_dim, self.dim,bias=False)

    def forward(self,
                x : torch.Tensor,
                freqs_cis : torch.Tensor,
                kv : torch.Tensor,
                cache : Tuple[torch.Tensor, torch.Tensor],
                mask : torch.Tensor,
                ) -> torch.Tensor:
        """Applies the attention mechanism to the query and key tensors."""

        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis=freqs_cis)
        k = apply_rotary_emb(k, freqs_cis=freqs_cis)

        k_cache, v_cache = cache
        k_cache.index_copy(1, kv, k)
        v_cache.index_copy(1, kv, v)

        key = k_cache
        value = v_cache
        if self.n_kv_heads != self.n_heads:
            key = torch.repeat_interleave(key, self.repeats, dim=2)
            value = torch.repeat_interleave(value, self.repeats, dim=2)
        
        q = q.transpose(1,2)
        k = key.transpose(1,2)
        v = value.transpose(1,2)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        output = torch.matmul(scores, v)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        output = self.out_proj(output)
        return output

class MLP(nn.Module):
    def __init__(self, args : ModelConfig):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.hidden_dim = args.hidden_dim

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class GemmaBlock(nn.Module):
    def __init__(self, args : ModelConfig):
        super().__init__()
        self.self_attn = Attention(ModelConfig)
        self.mlp = MLP(ModelConfig)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_layernorm = RMSNorm(args.dim, eps=args)
    
    def forward(self,
                x : torch.Tensor,
                freqs_cis : torch.Tensor,
                kv : torch.Tensor,
                cache : Tuple[torch.Tensor, torch.Tensor],
                mask : torch.Tensor,
                ) -> torch.Tensor:
        res = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, freqs_cis, kv, cache, mask)
        x = x + res
        res = x
        x = self.post_layernorm(x)
        x = self.mlp(x)
        x = x + res
        return x

class Gemma(nn.Module):
    def __init__(self, args : ModelConfig):
        super().__init__()
        self.args = args
        assert args.dim % args.n_heads == 0

        self.head_dim = args.head_dim
        self.vocab_size = args.vocab_size
        self.num_layers = args.n_layers
        self.embedding = Embedding(self.vocab_size, args.dim)
        self.sample = Sampler(self.vocab_size)
        self.tokenizer = Tokenizer()

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(GemmaBlock(args))
        self.layernorm = RMSNorm(args.dim, eps=args.norm_eps)

        rope_theta = 10000.0
        freq_cis = precompute_freqs_cis(self.head_dim, args.max_seq_len * 2, theta=rope_theta)
        self.register_buffer("freq_cis", freq_cis)

    @torch.no_grad()
    def forward(
        self,
        inputs: torch.Tensor,
        input_pos: torch.Tensor,
        kv: torch.Tensor,
        caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_pos: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        freq_cis = self.freq_cis.index_select(0, input_pos)
        kv = input_pos

        x = self.embedding(inputs)
        x = x * (self.args.dim**0.5)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x, freq_cis, kv, caches[i], mask)
        x = self.layernorm(x)

        embedder_weight = self.embedding.weight
        next = self.sample(
            embedder_weight,
            x,
            output_pos,
            temperatures,
            top_ps,
            top_ks,
        )
        return next
    def generate(
            self,
            input : str,
            device : str = None,
            output_length : int = 128,
            temperature : float = 1.0,
            top_k : int = 70,
            top_p : float = 1.0,
    ) -> str :
        is_prompt = isinstance(input, str)
        if is_prompt:
            prompts = [input]
        if device == None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        batch_size = len(prompts)
        tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_size = min(len(token) for token in tokens)
        max_prompt_size = max(len(token) for token in tokens)
        max_sequence_size = output_length + max_prompt_size
        assert max_sequence_size <= self.args.max_seq_len

        cache = []
        for _ in range(self.num_layers):
            size = (batch_size , max_sequence_size, self.args.n_kv_heads, self.args.head_dim)
            k_cache = torch.zeros(size=size, dtype=torch.float32, device=device)
            v_cache = torch.zeros(size=size, dtype=torch.float32, device=device)
            cache.append((k_cache, v_cache))
        
        token_ids = torch.full((batch_size, max_sequence_size),self.tokenizer.pad_id, dtype=torch.int64)
        input_ids = torch.full((batch_size, min_prompt_size), self.tokenizer.pad_id, dtype=torch.int64)

        for i , p in enumerate(tokens):
            token_ids[i, : len(p)] = torch.tensor(p)
            input_ids[i, : min_prompt_size] = torch.tensor(p[:min_prompt_size])
        token_ids.to(device)
        input_ids.to(device)
        prompt_mask_tensor = token_ids != self.tokenizer.pad_id

        input_pos = torch.arange(0, min_prompt_size, dtype=torch.long, device=device)

        mask_tensor = torch.full((1,1,max_sequence_size, max_sequence_size),-2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_pos)
        output_pos = torch.LongTensor([min_prompt_size -1]).to(device)
        temperature_tensor = torch.FloatTensor([temperature] * batch_size).to(device)
        top_p_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_k_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_size, dtype=torch.int64).to(device)

        for i in range(max_sequence_size - min_prompt_size):
            next_token = self(
                input_ids,
                input_pos,
                None,
                curr_mask_tensor,
                output_pos,
                temperature_tensor,
                top_p_tensor,
                top_k_tensor,
            )
            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token = token_ids.index_select(1, output_index).squeeze(dim=1)
            output_token = torch.where(curr_prompt_mask, curr_token, next_token).unsqueeze(dim=1)
            token_ids.index_copy_(1,output_index, output_token)

            input_pos = output_token.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_pos)
            output_pos = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        token_ids = token_ids.tolist()
        result = []
        for i , token in enumerate(token_ids):
            trimmed = token[len(tokens[i]):len(tokens[i]) + output_length]
            if self.tokenizer.eos_id in trimmed:
                eos_index = trimmed.index(self.tokenizer.eos_id)
                trimmed = trimmed[:eos_index]
            result.append(self.tokenizer.decode(trimmed))
        return result

    def load_weights(self, model_path: str):
        self.load_state_dict(
            torch.load(
                model_path, mmap=True, weights_only=True,
            )['model_state_dict'],
            strict=False,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default="./model.pt", help='model path if any for inferencing')
    parser.add_argument('--prompt', type=str, required=True, help='prompt to generate')
    parser.add_argument('--device', type=str, required=False, help='device to run on')
    parser.add_argument('--output_length', type=int, required=False, default=128, help='output length')
    parser.add_argument('--temperature', type=float, required=False, default=1.0, help='temperature')
    parser.add_argument('--top_k', type=int, required=False, default=70, help='top k')
    parser.add_argument('--top_p', type=float, required=False, default=1.0, help='top p')
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    model = Gemma(config)
    model.load_weights(args.model)
    result = model.generate(args.prompt, device=device, output_length=args.output_length, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print([r for r in result])