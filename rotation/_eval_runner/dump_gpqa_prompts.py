#!/usr/bin/env python3
"""Send GPQA-diamond prompts to a sglang server at max_tokens=1 to trigger
the DUMP_KVCACHE hook on every prefill. This is the calibration-data
producer for the OSCAR rotation phase.

The server is expected to be configured with:
    DUMP_KVCACHE=true
    DUMP_KVCACHE_TOKENS=<budget>
so the dump hook auto-stops once the token budget is reached; this script
just keeps sending prompts until the server has captured enough.

Usage:
  python dump_gpqa_prompts.py \
    --model Qwen/Qwen3-8B \
    --base-url http://127.0.0.1:31050/v1 \
    --num-threads 32 \
    --num-prompts 198 \
    --temperature 0.6 \
    --variant diamond
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SE_DIR = REPO / "third_party" / "simple_evals"
assert SE_DIR.is_dir(), f"missing {SE_DIR}"
sys.path.insert(0, str(SE_DIR.parent))

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--variant", default="diamond")
    p.add_argument("--num-prompts", type=int, default=198,
                   help="how many GPQA prompts to send (sequence_lens budget on "
                   "the server side will auto-stop the dump hook before this).")
    p.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help=(
            "Number of concurrent requests. QKV dumping synchronizes CUDA and "
            "writes tensors for every layer, so the stable default is 1."
        ),
    )
    p.add_argument("--temperature", type=float, default=0.6,
                   help="Sampling temperature (unused because max_tokens=1, "
                        "but kept for API compatibility).")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=1,
                   help="1 is enough — we only need the prefill pass to fire.")
    p.add_argument("--api", choices=("chat", "completions"), default="chat",
                   help="Use completions for base models without a chat_template.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dump-dir",
        default=os.environ.get("DUMP_KVCACHE_DIR"),
        help="QKV dump directory. Defaults to DUMP_KVCACHE_DIR.",
    )
    p.add_argument(
        "--dump-token-budget",
        type=int,
        default=int(os.environ.get("DUMP_KVCACHE_TOKENS", "0") or "0"),
        help="Stop submitting prompts once layer_0 seq_lens reaches this token budget.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Sleep between sequential dump requests.",
    )
    return p


def _build_prompts(num_prompts, variant, seed):
    """Load GPQA via simple-evals' loader to get the exact same prompt
    distribution as the eval."""
    import pandas
    url = f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
    df = pandas.read_csv(url)
    rows = [r.to_dict() for _, r in df.iterrows()]
    rng = random.Random(seed)
    if num_prompts < len(rows):
        rows = rng.sample(rows, num_prompts)
    out = []
    for r in rows:
        perm = rng.sample(range(4), 4)
        choices = [
            r["Correct Answer"], r["Incorrect Answer 1"],
            r["Incorrect Answer 2"], r["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in perm]
        body = {
            "A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3],
            "Question": r["Question"],
        }
        out.append(QUERY_TEMPLATE_MULTICHOICE.format(**body))
    return out


def _send_one(client, model, prompt, temperature, top_p, top_k, max_tokens, api):
    try:
        if api == "completions":
            client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"top_k": top_k},
            )
        else:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"top_k": top_k},
            )
        return "ok"
    except Exception as e:
        return f"err: {e!r}"


def _dumped_tokens(dump_dir, layer_id=0):
    if not dump_dir:
        return 0
    seq_dir = Path(dump_dir) / f"layer_{layer_id}" / "seq_lens"
    if not seq_dir.is_dir():
        return 0
    try:
        import torch
    except Exception:
        return 0

    total = 0
    for path in sorted(seq_dir.glob("*.pt"), key=lambda p: int(p.stem)):
        try:
            seq_lens = torch.load(path, weights_only=True, map_location="cpu")
            total += int(seq_lens.sum().item())
        except Exception:
            continue
    return total


def _budget_reached(args):
    if args.dump_token_budget <= 0:
        return False
    return _dumped_tokens(args.dump_dir) >= args.dump_token_budget


def main():
    args = _build_argparser().parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)

    prompts = _build_prompts(args.num_prompts, args.variant, args.seed)
    print(f"[dump] sending {len(prompts)} GPQA-{args.variant} prompts at "
          f"max_tokens={args.max_tokens} (server-side DUMP_KVCACHE_TOKENS "
          "controls when the dump hook stops)", flush=True)

    t0 = time.time()
    submitted = 0
    if args.num_threads <= 1:
        n_ok = n_err = 0
        for i, prompt in enumerate(prompts):
            if _budget_reached(args):
                print(
                    f"[dump] token budget reached before prompt {i}; "
                    f"dumped_tokens={_dumped_tokens(args.dump_dir)}",
                    flush=True,
                )
                break
            submitted += 1
            r = _send_one(
                client,
                args.model,
                prompt,
                args.temperature,
                args.top_p,
                args.top_k,
                args.max_tokens,
                args.api,
            )
            if r == "ok":
                n_ok += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f"  prompt {i}: {r}", flush=True)
            if args.poll_interval > 0:
                time.sleep(args.poll_interval)
    else:
        n_ok = n_err = 0
        prompt_iter = iter(enumerate(prompts))
        pending = {}
        with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
            while True:
                while len(pending) < args.num_threads and not _budget_reached(args):
                    try:
                        i, prompt = next(prompt_iter)
                    except StopIteration:
                        break
                    fut = ex.submit(
                        _send_one,
                        client,
                        args.model,
                        prompt,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        args.max_tokens,
                        args.api,
                    )
                    pending[fut] = i
                    submitted += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    i = pending.pop(fut)
                    r = fut.result()
                    if r == "ok":
                        n_ok += 1
                    else:
                        n_err += 1
                        if n_err <= 5:
                            print(f"  prompt {i}: {r}", flush=True)
                if _budget_reached(args):
                    print(
                        f"[dump] token budget reached; "
                        f"dumped_tokens={_dumped_tokens(args.dump_dir)}",
                        flush=True,
                    )
                    break
    dumped = _dumped_tokens(args.dump_dir)
    print(
        f"[dump] done in {time.time()-t0:.1f}s  submitted={submitted} "
        f"ok={n_ok}  err={n_err}  dumped_tokens={dumped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
