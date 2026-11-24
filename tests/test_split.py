import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Exaone4Config, Exaone4Model
from transformers.models.exaone4.modeling_exaone4 import Exaone4RotaryEmbedding

from exaone_split.model import SplitShard
from exaone_split.protocol import pack_hidden, unpack_hidden


class TestSplit(unittest.TestCase):
    def test_cached_split_matches_official_across_sliding_window(self):
        config = Exaone4Config(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_attention_heads=4,
            num_key_value_heads=2, num_hidden_layers=8, sliding_window=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"]
            + ["sliding_attention"] * 3 + ["full_attention"],
        )
        config._attn_implementation = "sdpa"
        reference = Exaone4Model(config).to(torch.bfloat16).eval()
        reference.rotary_emb = Exaone4RotaryEmbedding(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = {f"model.language_model.{k}": v for k, v in reference.state_dict().items()}
            save_file(weights, str(path / "model.safetensors"))
            (path / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {k: "model.safetensors" for k in weights},
            }))
            for split in (2, 4, 7):
                with self.subTest(split=split), patch("exaone_split.model.get_config", return_value=config):
                    front = SplitShard(path, 0, split, "cpu")
                    back = SplitShard(path, split, 8, "cpu")
                    cache = None
                    position = 0
                    with torch.inference_mode():
                        for length in (3, 1, 1, 1, 2):
                            ids = torch.arange(position, position + length).unsqueeze(0)
                            official = reference(ids, use_cache=True, past_key_values=cache)
                            cache = official.past_key_values
                            h = front.forward(reference.embed_tokens(ids), position)
                            output = back.forward(unpack_hidden(pack_hidden(h)), position)
                            torch.testing.assert_close(reference.norm(output), official.last_hidden_state,
                                                       atol=0, rtol=0)
                            position += length
                    self.assertEqual(front.cache.get_seq_length(), position)
                    self.assertEqual(back.cache.get_seq_length(), position)
                    with self.assertRaises(ValueError):
                        back.forward(output, 0)
