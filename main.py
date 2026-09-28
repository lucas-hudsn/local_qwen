"""Multi-engine local LLM benchmark harness.

Iterates over ollama, llama.cpp (server), mlx-lm, and vLLM(-metal) with the
same prompt and prints one comparison table. Engines that are unreachable
(or not installed) are skipped, so run with whatever subset is serving.

Start only what you compare (one 27B at a time fits 24 GB RAM):

    ollama serve                      # then: ollama pull qwen3.8:27b
    llama-server -m ~/models/qwen3.8-27b-Q4_K_M.gguf --port 8081 -c 4096 -ngl 99
    vllm serve dwsllc/Qwen3.8-27B-4bit --gpu-memory-utilization 0.72 \\
        --max-model-len 2048 --host 127.0.0.1 --port 8000
    # mlx-lm needs no server (in-process); needs Python <=3.13, e.g. /tmp/mlxbench/.venv

Usage:
    python3 main.py [--list] [backend ...] [--prompt ...] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPT = (
    "Explain the difference between Flash Attention and PagedAttention "
    "in 150 words."
)
MAX_TOKENS = 200

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.8:27b"
LLAMACPP_URL = "http://localhost:8081"
VLLM_URL = "http://127.0.0.1:8000"
VLLM_MODEL = "dwsllc/Qwen3.8-27B-4bit"
MLX_MODEL = "dwsllc/Qwen3.8-27B-4bit"


def _post_json(url: str, payload: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def bench_ollama(prompt: str = PROMPT, max_tokens: int = MAX_TOKENS) -> dict:
    """Benchmark via ollama /api/generate (engine-reported durations)."""
    d = _post_json(
        f"{OLLAMA_URL}/api/generate",
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens, "seed": 0},
            "think": False,
        },
    )
    dec_s = d["eval_duration"] / 1e9
    return {
        "prompt_tokens": d.get("prompt_eval_count"),
        "gen_tokens": d.get("eval_count"),
        "ttft_s": round(d.get("prompt_eval_duration", 0) / 1e9, 2),
        "decode_tok_s": round(d["eval_count"] / dec_s, 2),
        "total_s": round(d.get("total_duration", 0) / 1e9, 2),
        "sample": d.get("response", "")[:200],
    }


def bench_llamacpp(prompt: str = PROMPT, max_tokens: int = MAX_TOKENS) -> dict:
    """Benchmark via llama-server /completion (engine-reported timings)."""
    d = _post_json(
        f"{LLAMACPP_URL}/completion",
        {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": False,
            "cache_prompt": False,
        },
    )
    t = d.get("timings", {})
    pre_s = t.get("prompt_ms", 0) / 1000
    dec_s = t.get("predicted_ms", 0) / 1000
    return {
        "prompt_tokens": t.get("prompt_n"),
        "gen_tokens": t.get("predicted_n"),
        "ttft_s": round(pre_s, 2),
        "decode_tok_s": round(t["predicted_n"] / dec_s, 2),
        "total_s": round(pre_s + dec_s, 2),
        "sample": d.get("content", "")[:200],
    }


def bench_mlx(prompt: str = PROMPT, max_tokens: int = MAX_TOKENS) -> dict:
    """Benchmark in-process via mlx_lm (no server needed)."""
    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        return {"error": f"mlx_lm not importable ({exc}); use a Python<=3.13 MLX env"}
    model, tokenizer = load(MLX_MODEL)
    prompt_tokens = len(tokenizer.encode(prompt))
    t0 = time.perf_counter()
    text = generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=0.0, seed=0
    )
    total_s = time.perf_counter() - t0
    gen_tokens = len(tokenizer.encode(text))
    return {
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "ttft_s": None,
        "decode_tok_s": round(gen_tokens / total_s, 2),
        "total_s": round(total_s, 2),
        "sample": text[:200],
    }


def bench_vllm(prompt: str = PROMPT, max_tokens: int = MAX_TOKENS) -> dict:
    """Benchmark via vLLM OpenAI API with streaming (measures true TTFT)."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    usage: dict = {}
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            obj = json.loads(chunk)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(delta)
    total_s = time.perf_counter() - t0
    gen_tokens = usage.get("completion_tokens") or 0
    dec_s = total_s - (ttft or 0)
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "gen_tokens": gen_tokens,
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "decode_tok_s": round(gen_tokens / dec_s, 2) if gen_tokens and dec_s > 0 else 0,
        "total_s": round(total_s, 2),
        "sample": "".join(parts)[:200],
    }


BACKENDS = {
    "ollama": bench_ollama,
    "llamacpp": bench_llamacpp,
    "mlx": bench_mlx,
    "vllm": bench_vllm,
}


def run_all(prompt: str, max_tokens: int, which: list[str]) -> list[tuple[str, dict]]:
    """Iterate over backends; a failure in one never stops the others."""
    results = []
    for name in which:
        fn = BACKENDS[name]
        try:
            results.append((name, fn(prompt, max_tokens)))
        except Exception as exc:  # unreachable server, timeout, ...
            results.append((name, {"error": f"{type(exc).__name__}: {exc}"}))
    return results


def print_table(results: list[tuple[str, dict]]) -> None:
    header = f"{'backend':<10}{'prompt_tok':>11}{'gen_tok':>9}{'ttft_s':>9}{'tok/s':>9}{'total_s':>9}  sample / error"
    print(header)
    print("-" * len(header))
    for name, r in results:
        if "error" in r:
            print(f"{name:<10}{'—':>11}{'—':>9}{'—':>9}{'—':>9}{'—':>9}  SKIP: {r['error']}")
            continue
        ttft = f"{r['ttft_s']:.2f}" if r["ttft_s"] is not None else "n/a"
        sample = (r.get("sample") or "").replace("\n", " ")
        print(
            f"{name:<10}{r['prompt_tokens']:>11}{r['gen_tokens']:>9}"
            f"{ttft:>9}{r['decode_tok_s']:>9.2f}{r['total_s']:>9.2f}  {sample}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backends", nargs="*", help="subset to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list backends and exit")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    if args.list:
        print("\n".join(BACKENDS))
        return

    which = args.backends or list(BACKENDS)
    unknown = [w for w in which if w not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backends: {unknown} (choose from {list(BACKENDS)})")

    print(f"prompt: {args.prompt!r}  max_tokens={args.max_tokens}\n")
    print_table(run_all(args.prompt, args.max_tokens, which))


if __name__ == "__main__":
    main()
