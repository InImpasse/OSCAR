# OSCAR INT4 KV cache — llama.cpp fork

This fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) carries the
OSCAR KV-cache work used by the `OSCAR-KV-Quant` validation harness.

The current delivery path is **OSCAR INT4 KV cache**:

- rotated GGUF model with baked OSCAR K/V rotations
- `q4_0/q4_0` KV cache through llama.cpp `--cache-type-k/--cache-type-v`
- CUDA flash-attention validation on Granite 4.0 1B
- long-context memory reduction with prefill speed close to BF16

The earlier INT2 / OSCAR2 work remains in the branch as an experimental research
path, but it is not the current delivery target.

> Base llama.cpp documentation is preserved in
> [`README.upstream.md`](README.upstream.md). This README covers only the OSCAR
> KV-cache additions and the validated INT4 path.

---

## Current Status

Validated target:

| variant | model | KV cache | status |
|---|---|---|---|
| `baseline_bf16` | base GGUF | `bf16/bf16` | accuracy and memory baseline |
| `plain_int4` | base GGUF | `q4_0/q4_0` | healthy INT4 control |
| `oscar_int4` | rotated GGUF | `q4_0/q4_0` | current delivery path |

Experimental / non-delivery paths:

| variant | KV cache | status |
|---|---|---|
| `plain_int2` | `q2_0/q2_0` | memory win, too slow at long prefill |
| `oscar_int2` | `q2_0/q2_0` | experimental; 32K path is a known no-go |
| `oscar2` KV | `oscar2/oscar2` | research path kept for comparison |

The key practical result is that INT4 gets most of the KV memory reduction needed
for the target hardware while avoiding the long-prefill slowdown seen with INT2.

---

## 32K Result Snapshot

Granite 4.0 1B BF16, RTX 5050 Laptop GPU, llama.cpp CUDA build, `-ngl 999`,
flash attention enabled.

| variant | prompt | KV | KV pool MiB | peak MiB | pp tok/s | tg tok/s |
|---|---:|---|---:|---:|---:|---:|
| `baseline_bf16` | 32768 | `bf16/bf16` | 2560.0 | 6142 | 2586.6 | 49.4 |
| `oscar_int4` | 32768 | `q4_0/q4_0` | 720.0 | 4306 | 2576.8 | 36.8 |

INT4 reduced peak memory by about **1836 MiB**, essentially matching the
theoretical KV-pool reduction of **1840 MiB**, while keeping prefill throughput
within the BF16 band in this run.

Small-sample quality smoke:

| variant | GPQA | GSM8K |
|---|---:|---:|
| `baseline_bf16` | 3/10 | 4/10 |
| `oscar_int4` | 4/10 | 4/10 |

These quality numbers are a small smoke test, not a full benchmark claim, but
they are enough to keep `oscar_int4` in the same band as the BF16 baseline for
the current delivery gate.

---

## What This Fork Adds

- CUDA KV-cache support for the quantized cache paths used by the harness.
- OSCAR rotated GGUF support through baked per-layer K/V rotation tensors.
- llama.cpp command-line support for the relevant KV cache types.
- Accuracy and benchmark tooling compatibility with `llama-bench`,
  `llama-server`, and `llama-eval`.

The INT4 delivery path uses the existing llama.cpp `q4_0` KV cache type. The
rotated model contains OSCAR K/V rotation tensors, while the KV cache itself is
selected at runtime:

```bash
--cache-type-k q4_0 --cache-type-v q4_0
```

The model weights are not quantized by these flags; they only control KV cache
storage.

---

## Build

CUDA build:

```bash
cmake -S . -B build-cuda \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON

cmake --build build-cuda -j 4 --target llama-cli llama-bench llama-server
```

Check GPU visibility:

```bash
build-cuda/bin/llama-bench --list-devices
```

---

## Run INT4 KV

Use a rotated GGUF that contains the OSCAR K/V rotation tensors.

```bash
LLAMA_KV_NO_HADAMARD=1 \
LLAMA_KV_CLIP_RATIO=0.96 \
build-cuda/bin/llama-cli \
  -m checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf \
  -c 32768 \
  -ngl 999 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -p "What is 2+2?"
```

For benchmarking:

```bash
build-cuda/bin/llama-bench \
  -m checkpoints/gguf/granite-4.0-1b-base-bf16-rot-kv.gguf \
  -p 32768 \
  -n 64 \
  -r 1 \
  -ngl 999 \
  -fa 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

---

## Rotated GGUF

The rotated model is a normal GGUF with additional per-layer K/V rotation
tensors:

```text
blk.{i}.attn_k_rot.weight
blk.{i}.attn_v_rot.weight
```

The runtime applies the rotations in the llama.cpp graph. The cache type is still
chosen independently with `--cache-type-k` and `--cache-type-v`.

---

## Notes On INT2

The branch still contains INT2 and OSCAR2 code paths because they were part of
the exploration history. They are useful for research comparison, but they are
not the current product-facing target:

- exact `q2_0/q2_0` is substantially slower at long prefill
- the 32K INT2 run is guarded as a known no-go in the harness
- OSCAR2 remains experimental and should not be presented as the delivery path

Use `q4_0/q4_0` for the current OSCAR 4bit path.

---

## Upstream

Fork of **`ggml-org/llama.cpp`**. With default cache settings this fork behaves
like upstream llama.cpp. The OSCAR path is selected by using a rotated GGUF and a
quantized KV cache type such as `q4_0/q4_0`.
