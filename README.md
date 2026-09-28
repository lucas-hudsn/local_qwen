# local-qwen — Qwen3.8-27B across 4 local engines (macOS)

Hands-on companion to [HF LLM course ch2.8](https://huggingface.co/learn/llm-course/chapter2/8)
(optimized inference: TGI / vLLM / llama.cpp), reproduced on Apple Silicon with
**ollama, llama.cpp, mlx-lm, and vLLM-metal**.

- Machine: MacBook Pro M4 Pro, 24 GB unified memory, macOS 26, arm64
- Model: `Qwen/Qwen3.8-27B` (hybrid-attention, multimodal) in 4-bit:
  - ollama / llama.cpp: `Q4_K_M` GGUF (ollama tag `qwen3.8:27b`, 27.3B params)
  - mlx-lm / vLLM-metal: 4-bit MLX (`dwsllc/Qwen3.8-27B-4bit`)
- Rule: only one 27B engine resident at a time (weights alone are ~16 GB).

## Method

Same prompt, greedy decoding, 200 new tokens:

> "Explain the difference between Flash Attention and PagedAttention in 150 words."
> `temperature=0, seed=0`

Per-engine capture: ollama `/api/generate` (`prompt_eval_*`, `eval_*` durations),
llama.cpp `/completion` (`timings`), mlx-lm CLI (reported tok/s + peak memory),
vLLM OpenAI-compatible server (streamed TTFT + `usage`). See `main.py` — it
automates exactly this loop over whichever backends are up.

## Results (measured 2026-09-28, single run each)

| Engine | Prefill | Decode | Peak mem | Notes |
|---|---|---|---|---|
| mlx-lm (`dwsllc` 4-bit) | 15.6 tok/s (70 tok) | **15.4 tok/s** (200 tok) | 15.53 GB | fastest decode by ~2–3x; applied chat template (70 prompt tok) |
| llama.cpp server (GGUF `Q4_K_M`, `-c 4096`, full offload) | 19.4 tok/s (18 tok) | 7.2 tok/s | n/a | best prefill; raw prompt, no template |
| ollama (`qwen3.8:27b`, default ctx 262144) | ~13.2 tok/s (30 tok) | 5.1 tok/s | n/a | +12.8 s model load; wrapper overhead + huge default ctx cost |
| vLLM-metal 0.30.0.dev | — | not measured | — | installed OK; server startup interrupted, rerun cmd below |

Caveats: single run, not a sweep; prompt-token counts differ (template/raw);
context sizes differ (ollama 262k vs llama.cpp 4k — KV cache size matters);
quality not scored (mlx raw-prompt output opened with meta-commentary, a
prompting artifact, not an engine defect).

## Setup that was actually run

```bash
# ollama: model already present (17 GB)
ollama show qwen3.8:27b   # arch qwen35, Q4_K_M, ctx 262144, MTP draft

# llama.cpp: no re-download — reuse ollama's GGUF blob (magic bytes = GGUF)
brew install llama.cpp
ln -s ~/.ollama/models/blobs/sha256-f5f1…57d ~/models/qwen3.8-27b-Q4_K_M.gguf
llama-server -m ~/models/qwen3.8-27b-Q4_K_M.gguf --port 8081 -c 4096 -ngl 99

# mlx-lm (needs Python ≤3.13; project pins 3.14, so an isolated 3.12 venv)
uv venv --python 3.12 /tmp/mlxbench/.venv
uv pip install --python /tmp/mlxbench/.venv/bin/python mlx-lm mlx-vlm huggingface_hub
/tmp/mlxbench/.venv/bin/mlx_lm.generate --model dwsllc/Qwen3.8-27B-4bit \
  --prompt "..." --max-tokens 200 --temp 0.0 --seed 0 --trust-remote-code

# vLLM-metal (the real vLLM path on Apple Silicon; needs Python 3.12 + Xcode CLT)
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
~/.venv-vllm-metal/bin/vllm serve dwsllc/Qwen3.8-27B-4bit \
  --gpu-memory-utilization 0.72 --max-model-len 2048
```

Upstream (CUDA) vLLM is Linux-only; on macOS use **vllm-metal** (MLX/Metal
backend, OpenAI-compatible `vllm serve`). It needs MLX-format weights
(`mlx-community/*-4bit`), not GGUF.

## Harness (`main.py`, stdlib-only)

```bash
python3 main.py --list            # show backends
python3 main.py                   # probe all, skip what's down
python3 main.py ollama llamacpp   # subset
python3 main.py --prompt "..." --max-tokens 100
```

Each `bench_*` returns `{prompt_tokens, gen_tokens, ttft_s, decode_tok_s,
total_s, sample}` or `{error}`; `main()` iterates and prints one table.

## Takeaways

1. On M4 Pro, **MLX wins decode** (~15 tok/s vs ~5–7 for the llama stack).
2. `llama-server` beats the ollama wrapper on the same weights — check default
   context size before concluding an engine is slow.
3. vLLM on a Mac is real now (vllm-metal + paged KV on unified memory), at the
   cost of a source-build install and MLX-only weights.
