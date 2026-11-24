"""EXAONE text backbone with a configurable split after layer 3 by default.

Uses the Transformers 5.20 Exaone4Model text decoder.
"""
from copy import copy

from .checkpoint import Checkpoint, PREFIX, load_module, make_empty, quantize_linears


def get_config(model_dir):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(model_dir))
    text = config.text_config
    if getattr(config, "model_type", None) != "exaone4_5":
        raise ValueError("Only EXAONE 4.5 checkpoints are supported")
    if text.num_hidden_layers != 64 or text.hidden_size != 5120:
        raise ValueError("Unexpected EXAONE 4.5 33B configuration")
    text._attn_implementation = "sdpa"
    return text


def build_masks(config, hidden_states, position_ids, cache):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    kw = dict(config=config, inputs_embeds=hidden_states, attention_mask=None,
              past_key_values=cache, position_ids=position_ids)
    builders = {
        "full_attention": create_causal_mask,
        "sliding_attention": create_sliding_window_causal_mask,
    }
    return {kind: builders[kind](**kw) for kind in dict.fromkeys(config.layer_types)}


class SplitShard:
    def __init__(self, model_dir, start, stop, device, quantize=False):
        import torch
        from transformers.models.exaone4.modeling_exaone4 import Exaone4DecoderLayer, Exaone4RotaryEmbedding

        self.torch = torch
        self.config = get_config(model_dir)
        self.device = torch.device(device)
        self.start, self.stop = start, stop
        if not 0 <= start < stop <= self.config.num_hidden_layers:
            raise ValueError("Invalid shard layer range")
        self.dtype = torch.bfloat16
        self.cache_config = copy(self.config)
        self.cache_config.num_hidden_layers = stop - start
        self.cache_config.layer_types = self.config.layer_types[start:stop]
        self.layers = []
        cp = Checkpoint(model_dir)
        self.rotary = Exaone4RotaryEmbedding(self.config).to(self.device).eval()
        for i in range(start, stop):
            layer = make_empty(Exaone4DecoderLayer, self.config, i)
            load_module(layer, cp, PREFIX + f"layers.{i}.", torch.bfloat16)
            if quantize:
                layer = quantize_linears(layer, self.dtype)
            layer.self_attn.layer_idx = i - start
            layer.to(self.device)
            self.layers.append(layer.eval())
            if self.device.type == "cuda":
                print(f"GPU loaded layer {i}, allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        self.reset()

    def reset(self):
        from transformers.cache_utils import DynamicCache
        self.cache = DynamicCache(config=self.cache_config)
        self.position = 0

    @property
    def last_is_final(self):
        return self.stop == self.config.num_hidden_layers

    @property
    def is_cpu(self):
        return self.device.type == "cpu"

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def forward(self, x, start_position):
        torch = self.torch
        if start_position != self.position or not x.shape[1]:
            raise ValueError(f"Expected nonempty input at position {self.position}")
        if start_position + x.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Input exceeds model context")
        x = x.to(device=self.device, dtype=self.dtype)
        positions = torch.arange(start_position, start_position + x.shape[1],
                                 device=self.device).unsqueeze(0)
        masks = build_masks(self.cache_config, x, positions, self.cache)
        pos_emb = self.rotary(x, positions)
        with torch.inference_mode():
            for i, layer in enumerate(self.layers, self.start):
                x = layer(x, attention_mask=masks[self.config.layer_types[i]],
                          position_ids=positions, position_embeddings=pos_emb,
                          past_key_values=self.cache, use_cache=True)
        self.position += x.shape[1]
        return x


class CpuFront:
    def __init__(self, model_dir, split=4):
        import torch
        from transformers.models.exaone4.modeling_exaone4 import Exaone4RMSNorm
        from transformers import AutoTokenizer

        self.torch = torch
        self.config = get_config(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.shard = SplitShard(model_dir, 0, split, "cpu")
        cp = Checkpoint(model_dir)
        self.embedding = make_empty(torch.nn.Embedding, self.config.vocab_size, self.config.hidden_size)
        load_module(self.embedding, cp, PREFIX + "embed_tokens.", torch.bfloat16)
        self.norm = make_empty(Exaone4RMSNorm, self.config.hidden_size, self.config.rms_norm_eps)
        load_module(self.norm, cp, PREFIX + "norm.", torch.bfloat16)
        self.lm_head = make_empty(torch.nn.Linear, self.config.hidden_size, self.config.vocab_size, False)
        if "lm_head.weight" in cp.key_map:
            load_module(self.lm_head, cp, "lm_head.", torch.bfloat16)
        else:
            raise KeyError("Official untied checkpoint requires lm_head.weight")
        self.shard.reset()

    def embed_and_run(self, ids, position):
        with self.torch.inference_mode():
            x = self.embedding(ids)
            return self.shard.forward(x, position)

    def logits(self, hidden):
        with self.torch.inference_mode():
            h = self.norm(hidden.to(device="cpu", dtype=self.torch.bfloat16))[:, -1]
            return self.lm_head(h).float()

    def next_token(self, hidden):
        return int(self.logits(hidden).argmax(dim=-1).item())
