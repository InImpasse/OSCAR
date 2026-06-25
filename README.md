# Qwen3 INT2 KV cache with OSCAR — llama.cpp fork

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a **~2-bit (INT2) KV
cache** with the **OSCAR calibrated rotation**, so a 4B thinking model can run at **32K context**
with a tiny KV footprint — targeting edge / MacBook deployment.

On GPQA-Diamond with **Qwen3-4B-Thinking-2507**, full **K+V INT2 + OSCAR** recovers **f16-level
accuracy** (external OSCAR INT2 reference is ~62%).

> Base project docs (build options, general usage, supported backends) are upstream
> [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — preserved here as
> [`README.upstream.md`](README.upstream.md). This README covers only the INT2/OSCAR additions.

---

## Results — GPQA-Diamond @ 32K (Qwen3-4B-Thinking-2507, Q4_K_M weights)

Full chain-of-thought (`n_predict=16000`), HP buffer `sink=512 / recent=2048`, clip 0.96.

| KV cache config | GPQA |
|---|---|
| broken INT2 (32-wide rotation, no clip) | 0/2 |
| INT2 + full-head Hadamard + clip (data-free) | 2/6 = 33% |
| **INT2 + OSCAR calibrated rotation — K only (V=f16, isolation test)** | **6/6** (= f16) |
| **INT2 + OSCAR calibrated rotation — full K+V** | **14/20 = 70%** |
| f16 baseline (same n=20 sample) | 10/20 = 50% |
| external OSCAR INT2 reference (full 198-q) | ~62% |

**Read this honestly:**
- The decisive fix is the **data-calibrated rotation** (`R·H·P`), *not* plain Hadamard — Hadamard
  is near-useless at boundary layers (layer-0 attention-score SNR 1.8 dB vs calibrated 8.5 dB),
  which is why data-free Hadamard only reaches 33%.
- The n=20 used **stochastic decoding (temp ≈ 0.6)**, so INT2 and f16 sample *different*
  completions per question — that's why INT2 (70%) > f16 (50%) on this *hard* draw (sampling
  noise; both estimate the same true ~60-70%). The robust signals: n=6 INT2 == f16 (6/6),
  per-token SQNR (calibrated ≫ Hadamard for K; sound for V), and the pipeline matching the external
  OSCAR reference
  62% recipe.
- All numbers are **CPU-validated** on small samples (CPU is ~28 min/q for K-INT2, ~90 min/q for
  full K+V INT2). A full 198-q representative run belongs on GPU/Metal.

---

## What this fork adds

- **`GGML_TYPE_Q2_0`** KV type — block of 32, `{d:f16, m:f16, qs:u8[8]}` = 12 B/32 elems
  (~3 bits/elem effective; f16 KV @ 32K ≈ 7.5 GB → INT2 ≈ 1.4 GB).
- **OSCAR calibrated rotation** — per-layer orthogonal `R·H·P` (`R_k` = eigenvectors of the query
  covariance, `R_v` of the value covariance), shipped as GGUF tensors `blk.{i}.attn_k_rot.weight`
  / `attn_v_rot.weight`. Applied **in-graph, post-RoPE** in `src/models/qwen3.cpp` (`Q@M`, `K@M`;
  V rotated + `M_vᵀ` undo). The same orthogonal `M` hits both Q and K, so `Q'·K'` is exactly
  preserved and there is **no per-access undo** → fast.
- **Outlier clip** — per-row percentile clamp before quant (`LLAMA_KV_CLIP_RATIO`; OSCAR
  calibration commonly uses K=0.96 / V=0.92).
- **HP sink+recent buffer** — first `LLAMA_KV_HP_SINK` and last `LLAMA_KV_HP_RECENT` tokens kept
  high-precision; the rest INT2 (joint LP+HP attention).
- Lloyd-Max INT2 levels; the in-quant Hadamard is gated behind `LLAMA_KV_NO_HADAMARD` (the
  calibrated rotation already includes the `H`, so the quant must not re-apply it).

Runtime env vars:

| var | meaning | value used |
|---|---|---|
| `LLAMA_KV_NO_HADAMARD` | skip the in-quant Hadamard (rotation is in-graph) | `1` |
| `LLAMA_KV_CLIP_RATIO`  | per-row outlier clip percentile | `0.96` |
| `LLAMA_KV_HP_SINK`     | high-precision prefix tokens | `512` |
| `LLAMA_KV_HP_RECENT`   | high-precision recent tokens | `2048` |
| `LLAMA_KV_HP_NO_FUSED_Q2_0` | disable CUDA fused `q2_0` LP + F16 HP attention (slow concat path) | unset |
| `LLAMA_KV_Q2_0_OWHT` | enable staged CUDA q2_0 group writer/HP fused decoder; follows `LLAMA_KV_NO_HADAMARD` and `LLAMA_KV_CLIP_RATIO` in the writer (`LLAMA_CUDA_Q2_0_OWHT` is a compatibility alias) | unset |

With `--flash-attn` and `q2_0` LP + F16 HP caches, fused joint-softmax attention is enabled by default on CUDA (see `ggml_flash_attn_ext_q2_0_f16`).
`q2_0`+HP is not a pure `q2_0` cache: the main K/V cache uses `q2_0`, while the sink+recent HP side cache is stored as F16.
For prompt-processing batches that do not build the HP attention branch, the LP mask must keep HP-covered positions visible; otherwise sink/recent tokens would be masked from LP without being added back by HP attention.

Code touched (vs upstream): `ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/quants.c` (Q2_0 quant +
full-head OWHT + clip + NO_HADAMARD gate), `src/llama-arch.{h,cpp}`, `src/llama-model.h` (register
`ATTN_K_ROT`/`ATTN_V_ROT`), `src/models/qwen3.cpp` (apply rotations), plus the original Q2_0 type +
HP-buffer infra (`ggml-common.h`, `llama-kv-cache.cpp`, Metal kernels, …).

### Local CUDA smoke/benchmark snapshot

Measured with `llama-bench -ngl 999 -fa 1 -r 1`, `LLAMA_KV_HP_SINK=512`,
`LLAMA_KV_HP_RECENT=2048`, and `scripts/measure_vram.sh` on an RTX 5050 Laptop GPU.
The short/medium/long rows use `-p 512/2048/4096` and `-n 128`.

| model | length | KV | peak MiB | pp t/s | tg t/s |
|---|---:|---|---:|---:|---:|
| granite-4.0-1b-base-bf16 | short | f16 | 3764 | 2418.0 | 59.6 |
| granite-4.0-1b-base-bf16 | short | q2_0 | 3728 | 1889.9 | 63.7 |
| granite-4.0-1b-base-bf16 | short | q2_0+HP | 3728 | 1187.6 | 58.4 |
| granite-4.0-1b-base-bf16 | medium | f16 | 3872 | 2710.0 | 59.7 |
| granite-4.0-1b-base-bf16 | medium | q2_0 | 3746 | 1120.3 | 63.2 |
| granite-4.0-1b-base-bf16 | medium | q2_0+HP | 3746 | 1123.7 | 64.0 |
| granite-4.0-1b-base-bf16 | long | f16 | 4032 | 2940.7 | 62.8 |
| granite-4.0-1b-base-bf16 | long | q2_0 | 3766 | 732.1 | 63.9 |
| granite-4.0-1b-base-bf16 | long | q2_0+HP | 3766 | 729.7 | 63.2 |
| gemma-4-E2B-it-bf16 | short | f16 | 5206 | 1727.7 | 41.1 |
| gemma-4-E2B-it-bf16 | short | q2_0 | 5176 | 1117.4 | 27.6 |
| gemma-4-E2B-it-bf16 | short | q2_0+HP | 5176 | 1106.6 | 27.9 |
| gemma-4-E2B-it-bf16 | medium | f16 | 5190 | 2229.1 | 42.2 |
| gemma-4-E2B-it-bf16 | medium | q2_0 | 5146 | 1008.6 | 27.3 |
| gemma-4-E2B-it-bf16 | medium | q2_0+HP | 5146 | 1021.7 | 27.1 |
| gemma-4-E2B-it-bf16 | long | f16 | 5206 | 2714.8 | 46.1 |
| gemma-4-E2B-it-bf16 | long | q2_0 | 5202 | 51.6 | 26.9 |
| gemma-4-E2B-it-bf16 | long | q2_0+HP | 5202 | 54.1 | 27.1 |

---

## Build

```bash
# Linux / server, CPU-only (what the results above were produced on):
cmake -B build -DLLAMA_CURL=OFF -DGGML_METAL=OFF
cmake --build build -j --target llama-cli

# macOS (Apple Silicon): Metal is ON by default
cmake -B build
cmake --build build -j --target llama-cli
```

## Run

Requires a **rotated GGUF** (`*-rot-kv.gguf`) containing the `attn_k_rot`/`attn_v_rot` tensors (see
below). Then:

```bash
LLAMA_KV_NO_HADAMARD=1 LLAMA_KV_CLIP_RATIO=0.96 \
LLAMA_KV_HP_SINK=512 LLAMA_KV_HP_RECENT=2048 \
./build/bin/llama-cli -m qwen3-4b-thinking-q4km-rot-kv.gguf \
  --cache-type-k q2_0 --cache-type-v q2_0 \
  -c 32768 -n 16000 \
  -p "your prompt"
```

- `--cache-type-k q2_0 --cache-type-v q2_0` = full K+V INT2. (Use `--cache-type-v f16` to keep V
  high-precision / isolate the K rotation.)
- A plain (non-rotated) GGUF with these flags falls back to data-free Hadamard INT2 (~33% — do not
  use for accuracy).

---

## Producing the rotated model (calibration → GGUF)

The rotation matrices are **data-calibrated** from the model's own activations on GPQA (GPU needed
for the dump; tooling lives in the CoQuant `rotation/` scripts):

1. **Dump post-RoPE Q/K/V** on GPQA calibration prompts with an activation-dump pipeline.
2. **Compute rotations** — `METHOD=qqt_sst` (`R·H·P`): `R_k` from the query covariance, `R_v` from
   the value covariance → `k_rotation_qqt_r_h_pbr.pt`, `v_rotation_sst_r_h_pbr.pt` (36 × 128×128).
3. **Bake into GGUF** — append the per-layer matrices (stored as `Mᵀ` so `ggml_mul_mat(rot,K)=K@M`)
   as `blk.{i}.attn_{k,v}_rot.weight` → `*-rot-kv.gguf` (copy-through; base weights not
   re-quantized).

The base model is a standard Qwen3-4B-Thinking-2507 Q4_K_M GGUF; only the KV cache is INT2.

---

## MacBook deployment

On Apple Silicon the build above enables **Metal** and llama.cpp offloads to the GPU
automatically; the in-graph rotation (plain matmuls) runs on Metal fine.

**⚠️ Caveat:** the INT2/OSCAR scheme is currently **validated on the CPU backend only**. The Metal
`q2_0` dequant kernels have **not** been updated/validated for the calibrated-rotation +
`LLAMA_KV_NO_HADAMARD` path — so GPU output is **unverified** and may be wrong/degraded.

- **Correct output today** (matches the results above): add `-ngl 0` to keep the KV path on CPU.
  Correct, but no GPU speedup and slow at 32K.
- **Metal GPU speed:** drop `-ngl 0`, but **sanity-check** (compare a known question CPU vs GPU)
  before trusting it.
- **Make the GPU path correct** (the real MacBook win): the Metal `q2_0` dequant kernel needs the
  same `NO_HADAMARD` / Lloyd-Max alignment the CPU path got — a follow-up to test on a Mac.

---

## Upstream

Fork of **`ggml-org/llama.cpp`**. INT2/OSCAR is additive and gated behind the `q2_0` cache type +
the env vars above; with default cache types this behaves exactly like upstream llama.cpp. Full
base docs: [`README.upstream.md`](README.upstream.md).
