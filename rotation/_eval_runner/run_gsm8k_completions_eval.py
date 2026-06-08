#!/usr/bin/env python3
"""Small GSM8K eval for base models served via /v1/completions."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--data-path", type=Path, default=Path("/tmp/gsm8k_test.jsonl"))
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    return parser


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"[gsm8k] downloading {GSM8K_URL} -> {path}", flush=True)
    with urllib.request.urlopen(GSM8K_URL, timeout=30) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        _download(path)
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def extract_answer(text: str | None) -> int | None:
    if not text:
        return None
    matches = re.findall(r"-?\d[\d,]*", text)
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def gold_answer(answer: str) -> int:
    marker = "####"
    if marker in answer:
        return int(answer.split(marker)[-1].strip().replace(",", ""))
    parsed = extract_answer(answer)
    if parsed is None:
        raise ValueError(f"cannot parse gold answer: {answer}")
    return parsed


def example_prompt(row: dict, include_answer: bool) -> str:
    text = f"Question: {row['question']}\nAnswer:"
    if include_answer:
        text += f" {row['answer']}"
    return text


def build_prompt(rows: list[dict], idx: int, num_shots: int) -> str:
    shots = "\n\n".join(example_prompt(rows[i], include_answer=True) for i in range(num_shots))
    return shots + "\n\n" + example_prompt(rows[idx], include_answer=False)


def main() -> int:
    args = build_argparser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    from openai import OpenAI

    rows = load_rows(args.data_path)
    eval_indices = list(range(args.num_shots, min(args.num_shots + args.num_examples, len(rows))))
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )

    records = []
    correct = 0
    answered = 0
    t0 = time.time()
    for pos, idx in enumerate(eval_indices, start=1):
        prompt = build_prompt(rows, idx, args.num_shots)
        response = None
        for retry in range(args.max_retries + 1):
            try:
                response = client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    stop=["Question:"],
                    extra_body={"top_k": args.top_k},
                )
                break
            except Exception as exc:
                if retry >= args.max_retries:
                    raise
                backoff = min(2**retry, 30)
                print(f"[{pos:03d}/{len(eval_indices):03d}] retry in {backoff}s: {exc}", flush=True)
                time.sleep(backoff)
        text = response.choices[0].text if response else ""
        pred = extract_answer(text)
        gold = gold_answer(rows[idx]["answer"])
        ok = pred == gold
        correct += int(ok)
        answered += int(pred is not None)
        records.append(
            {
                "index": idx,
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "prompt": prompt,
                "response": text,
                "usage": response.usage.model_dump() if response and response.usage else None,
            }
        )
        print(f"[{pos:03d}/{len(eval_indices):03d}] pred={pred} gold={gold} ok={ok}", flush=True)
        if args.sleep_between:
            time.sleep(args.sleep_between)

    elapsed = time.time() - t0
    score = correct / len(eval_indices) if eval_indices else 0.0
    answer_rate = answered / len(eval_indices) if eval_indices else 0.0
    metrics = {
        "task": "gsm8k",
        "score": score,
        "correct": correct,
        "num_examples": len(eval_indices),
        "answered": answered,
        "answer_rate": answer_rate,
        "elapsed_sec": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out / "io_log.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    log_text = (
        f"Evaluation results for GSM8K on {args.model}\n"
        f"score={score:.6f}\n"
        f"correct={correct}/{len(eval_indices)}\n"
        f"answer_rate={answer_rate:.6f}\n"
        f"elapsed_sec={elapsed:.1f}\n"
    )
    (out / "eval.log").write_text(log_text)
    print(log_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
