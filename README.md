# EXAONE 4.5 Split Inference Lab

**Experimental prototype for public or synthetic prompts.** Raw activations and cache tensors may contain reconstructable text. Use the unauthenticated, unencrypted TCP protocol on loopback for this experiment.

## Scope and status

- Official checkpoint: `LGAI-EXAONE/EXAONE-4.5-33B` (text-only path, no vision encoder or MTP inference).
- CPU: tokenizer, input embeddings, first **4** Exaone4 decoder layers, final norm and LM head.
- GPU: remaining **60** decoder layers in **NF4** bitsandbytes quantization.
- CUDA state: KV cache stays on GPU for GPU layers; CPU shard maintains a separate KV cache for layers 0–3.
- Network: one-process-per-side TCP communication of lossless BF16 activations. A synthetic **1 ms RTT** is added once per token exchange (not actual kernel/network latency). No separate network namespaces are required.
- Decoding: greedy argmax, batch size 1, short text-only prompts. No speculative decoding, multimodal inputs, beam search, concurrent sessions, batching, encrypted transport, or authentication.
- For the RTX 4090 verification and measurements, see [the experiment report](REPORT.md). The protocol and split cache tests run on CPU.

## Verified model architecture (from upstream source)

The EXAONE 4.5 multimodal wrapper contains `model.language_model`, a standard `Exaone4Model` text backbone. Its config has 64 main layers, width 5120, and a mixture of global/sliding attention. The split path loads the `model.language_model.` tensors and the top-level LM head. The official config sets `tie_word_embeddings=false` and the checkpoint stores a separate `lm_head.weight`. Layer implementations are `Exaone4DecoderLayer` and use `Exaone4RotaryEmbedding`, `DynamicCache`, `create_causal_mask`, and `create_sliding_window_causal_mask`.

The GPU shard converts the official BF16 weights to NF4 at load time. The separately published AWQ checkpoint uses a different packing format and requires compatible modules.

## 1. Rent a Vast.ai instance (manual only)

Select one RTX 4090 (24 GiB), ideally >= 96 GiB system RAM and >= 150 GiB disk (more if cache persists). Use an Ubuntu + recent NVIDIA/PyTorch image that supports SSH. **This repository never provisions a rental, calls the Vast.ai API, or incurs charges.**

Do not open the experimental TCP server to the Internet. Keep client and server on the **same rental instance**, binding to `127.0.0.1`. Note that the VM/container still executes within one trust domain: it is an emulation of two hosts, not a security separation.

## 2. Install

On Vast.ai via SSH, after copying/extracting this repository:

```bash
cd exaone-split
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Use the CUDA-supported PyTorch wheel matching the rental image/driver.
# Install torch explicitly first if your image does not supply a suitable build.
python -m pip install -e .
python -c 'import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
python -m unittest discover -s tests -v
```

The project pins Transformers to 5.20.0. Verify that your PyTorch wheel exposes CUDA. Use official NVIDIA/PyTorch installation guidance for your driver and CUDA runtime.

## 3. Download and inspect the official checkpoint

```bash
exaone-split fetch --dir ./models/exaone45
exaone-split verify --dir ./models/exaone45
```

The download is large. The `fetch` command intentionally downloads the sharded official BF16 checkpoint, not the AWQ model. It skips nonessential files where practical but a source shard can still contain multimodal weights.

Expected architecture: `exaone4_5`, 64 text decoder layers, hidden width 5120.

## 4. Start the GPU process (Terminal A)

```bash
source .venv/bin/activate
exaone-split server --dir ./models/exaone45 --split 4 --host 127.0.0.1 --port 9876
```

It loads only layers 4–63 and quantizes each layer to NF4 on the GPU. Wait for `READY`. If a 24 GiB card runs out of VRAM, reduce context size or adapt the experimental design to CPU offload / a larger GPU; do not silently change the split configuration. Quantization accuracy differences are expected.

## 5. Start the CPU process (Terminal B)

```bash
source .venv/bin/activate
exaone-split client --dir ./models/exaone45 --split 4 \
  --host 127.0.0.1 --port 9876 --rtt-ms 1 \
  --max-tokens 16 --prompt '대한민국의 수도는 어디인가요?' \
  --output-json ./benchmark.json
```

The CPU loads 4 decoder layers, separate embedding and LM-head weights and final norm. The GPU receives BF16 hidden states and returns only the last hidden token (needed for the next-token head). Both sides retain their own per-layer KV caches. The benchmark JSON includes per-step CPU/GPU/network wall-clock timing, first-token latency, bytes transferred and total token throughput.

## 6. Repeat the benchmark

The server/client `--split` values must match. Change both from `4` to `2`, `8` or `16` to compare split depth. For network experiments, keep split=4 and vary `--rtt-ms` to `0.2`, `1`, `5`, and `20`; add `--bandwidth-mbps 1000` to simulate a nominal 1 Gbps serialization limit. Link delay is synthetic and added by the client *after* each exchange; it does not independently model queueing, jitter or TCP congestion.

## Implementation cautions

1. **Performance:** The CPU runs BF16 matrix operations and a large LM head, which can dominate decoding latency. Network simulation time and GPU processing time are both recorded separately. CPU initial layers can be substantially slower than a GPU-only baseline.
2. **Attention masks:** Hybrid global/sliding attention demands the same per-layer masking and RoPE behavior as upstream `Exaone4Model`. This prototype follows the Transformers 5.x implementation. API changes may break compatibility.
3. **Checkpoints:** The loader requires one `*.safetensors.index.json`; it loads each shard’s text backbone weights and the CPU LM head with `assign=True` on meta-allocated modules. Full 33B checkpoint download is still required.
4. **CPU/GPU numerics:** CPU BF16 and GPU NF4 do not produce identical outputs; use a smaller numerical-reference test before treating text differences as a distribution bug.
5. **Security:** Hidden states are not anonymization. Rootless users on another host are not a confidentiality guarantee. Running on Vast.ai specifically means the machine operator is part of your threat model.

## Diagnostic commands

```bash
nvidia-smi
python -m pip show torch transformers bitsandbytes safetensors
python -m unittest discover -s tests -v
```

If imports or cache keywords fail, verify that the active Python environment uses Transformers 5.20.0. If the official checkpoint lacks the expected key prefix, inspect the index before modifying `checkpoint.py`.

## Sources

- [Official model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B)
- [Transformers EXAONE 4.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/exaone4_5/modeling_exaone4_5.py)
- [Transformers EXAONE 4 text backbone](https://github.com/huggingface/transformers/blob/main/src/transformers/models/exaone4/modeling_exaone4.py)


## Ordered benchmark

```bash
OMP_NUM_THREADS=8 python -m exaone_split.cli benchmark --dir /root/exaone-model  \
  --output-json results/benchmark.json --repetitions 3 --threads 8
python -m unittest discover -s tests -v
```

The benchmark runs the official Transformers forward path, in-process split,
and a separate GPU server on loopback TCP. For the 4-layer split it measures
0, 0.2, 1, 5 and 20 ms and adds 1 and 10 Gbps conditions at 1 ms. Other splits
use 1 ms. JSON includes token IDs, per-step timing, latency and throughput
summaries, process peak RSS, CUDA allocator peaks, and device-wide NVIDIA samples.
The simulated delay is added once per forward exchange and includes an optional
sum of serialization times in both directions.

## CPU embedding-only split

Set `--split 0` on both endpoints to keep chat templating, tokenization,
embedding lookup, and text decoding on the client. The server runs all 64
decoder layers, final norm, LM head, and greedy token selection. It returns
one token ID per step. Run the long-prompt comparison from the project root.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m experiments.long_prompt \
  --dir /root/exaone-model --split 0 --output results/embedding-only-rerun.json
```

[4,096-token embedding-only analysis](docs/embedding-only-analysis.md)
