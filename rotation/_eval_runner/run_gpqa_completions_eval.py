#!/usr/bin/env python3
"""Small GPQA multiple-choice eval for base models served via /v1/completions."""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path


QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# Granite base models are FIM-trained; stop before FIM continuation garbage.
DEFAULT_STOP_SEQUENCES = [
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
]


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--variant", default="diamond")
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def load_examples(variant, num_examples, seed):
    import pandas

    url = f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
    df = pandas.read_csv(url)
    rows = [row.to_dict() for _, row in df.iterrows()]
    rng = random.Random(seed)
    if num_examples < len(rows):
        rows = rng.sample(rows, num_examples)

    examples = []
    letters = "ABCD"
    for row in rows:
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        perm = rng.sample(range(4), 4)
        shuffled = [choices[i] for i in perm]
        gold = letters[perm.index(0)]
        prompt = QUERY_TEMPLATE_MULTICHOICE.format(
            Question=row["Question"],
            A=shuffled[0],
            B=shuffled[1],
            C=shuffled[2],
            D=shuffled[3],
        )
        examples.append({"prompt": prompt, "gold": gold, "question": row["Question"]})
    return examples


def extract_answer(text):
    matches = re.findall(r"(?i)answer\s*:\s*([A-D])\b", text or "")
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([A-D])\b", text or "")
    return matches[-1].upper() if matches else None


def main():
    args = build_argparser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Local SGLang evals should not be routed through the user's global proxy.
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    from openai import OpenAI

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )
    examples = load_examples(args.variant, args.num_examples, args.seed)

    correct = 0
    answered = 0
    records = []
    t0 = time.time()
    for idx, ex in enumerate(examples):
        response = None
        for retry in range(args.max_retries + 1):
            try:
                response = client.completions.create(
                    model=args.model,
                    prompt=ex["prompt"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    stop=DEFAULT_STOP_SEQUENCES,
                    extra_body={
                        "top_k": args.top_k,
                        "min_tokens": args.min_tokens,
                    },
                )
                break
            except Exception as exc:
                if retry >= args.max_retries:
                    if args.continue_on_error:
                        print(
                            f"[{idx + 1:03d}/{len(examples):03d}] "
                            f"failed after {args.max_retries + 1} attempts: {exc}",
                            flush=True,
                        )
                        break
                    raise
                backoff = min(2 ** retry, 30)
                print(
                    f"[{idx + 1:03d}/{len(examples):03d}] retry {retry} "
                    f"in {backoff}s: {exc}",
                    flush=True,
                )
                time.sleep(backoff)
        if response is None:
            if not args.continue_on_error:
                raise RuntimeError(f"example {idx} failed without a response")
            record = {
                "index": idx,
                "gold": ex["gold"],
                "pred": None,
                "correct": False,
                "prompt": ex["prompt"],
                "response": None,
                "usage": None,
                "error": "no response after retries",
            }
            records.append(record)
            print(
                f"[{idx + 1:03d}/{len(examples):03d}] pred=None "
                f"gold={ex['gold']} ok=False error=True",
                flush=True,
            )
            if args.sleep_between:
                time.sleep(args.sleep_between)
            continue
        text = response.choices[0].text or ""
        pred = extract_answer(text)
        ok = pred == ex["gold"]
        correct += int(ok)
        answered += int(pred is not None)
        record = {
            "index": idx,
            "gold": ex["gold"],
            "pred": pred,
            "correct": ok,
            "prompt": ex["prompt"],
            "response": text,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        records.append(record)
        print(
            f"[{idx + 1:03d}/{len(examples):03d}] pred={pred} gold={ex['gold']} ok={ok}",
            flush=True,
        )
        if args.sleep_between:
            time.sleep(args.sleep_between)

    elapsed = time.time() - t0
    score = correct / len(examples) if examples else 0.0
    answer_rate = answered / len(examples) if examples else 0.0
    metrics = {
        "score": score,
        "correct": correct,
        "num_examples": len(examples),
        "answered": answered,
        "answer_rate": answer_rate,
        "elapsed_sec": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "io_log.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    lines = [
        f"Evaluation results for GPQA-{args.variant} on {args.model}",
        f"score={score:.6f}",
        f"correct={correct}/{len(examples)}",
        f"answer_rate={answer_rate:.6f}",
        f"elapsed_sec={elapsed:.1f}",
    ]
    log_text = "\n".join(lines) + "\n"
    (out / "eval.log").write_text(log_text)
    print(log_text, flush=True)


if __name__ == "__main__":
    main()
