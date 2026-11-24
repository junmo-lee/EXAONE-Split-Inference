"""Shard-aware EXAONE language-only checkpoint loader.

Avoids allocating the entire ~33B BF16 model in RAM/VRAM at once.
"""
import json
from pathlib import Path


MODEL_ID = "LGAI-EXAONE/EXAONE-4.5-33B"
PREFIX = "model.language_model."


def fetch_model(model_id=MODEL_ID, local_dir=None):
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=model_id, local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt", "*.jinja"],
    ))


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        indexes = list(self.path.glob("*.safetensors.index.json"))
        if len(indexes) != 1:
            raise FileNotFoundError("Expected one *.safetensors.index.json (official sharded BF16 checkpoint)")
        self.key_map = json.loads(indexes[0].read_text())["weight_map"]

    def load(self, prefix, dtype=None):
        from safetensors import safe_open

        matches = {key: name for key, name in self.key_map.items() if key.startswith(prefix)}
        if not matches:
            raise KeyError(f"No checkpoint tensors found under {prefix}")
        by_file = {}
        for key, name in matches.items():
            by_file.setdefault(name, []).append(key)
        state = {}
        for name, keys in by_file.items():
            with safe_open(str(self.path / name), framework="pt", device="cpu") as f:
                for key in keys:
                    value = f.get_tensor(key)
                    state[key[len(prefix):]] = value.to(dtype=dtype) if dtype else value
        return state


def load_module(module, checkpoint, prefix, dtype):
    state = checkpoint.load(prefix, dtype=dtype)
    module.load_state_dict(state, strict=True, assign=True)
    del state
    return module.eval()


def make_empty(constructor, *args):
    from accelerate import init_empty_weights
    with init_empty_weights():
        return constructor(*args)


def quantize_linears(module, compute_dtype):
    """Replace standard BF16 Linear layers with bitsandbytes NF4 before GPU upload."""
    import torch
    import bitsandbytes as bnb
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            converted = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                           bias=child.bias is not None,
                                           compress_statistics=True, quant_type="nf4",
                                           compute_dtype=compute_dtype, device="meta")
            converted.weight = bnb.nn.Params4bit(child.weight.detach().contiguous(),
                requires_grad=False, compress_statistics=True, quant_type="nf4", module=converted)
            if child.bias is not None:
                converted.bias = torch.nn.Parameter(child.bias.detach(), requires_grad=False)
            setattr(module, name, converted)
        else:
            quantize_linears(child, compute_dtype)
    return module
