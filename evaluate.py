#!/usr/bin/env python3
"""
Simple reasoning evaluation harness for math datasets.

The script reads JSONL datasets, queries a local model server for each problem,
and reports accuracy along with pass@k metrics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


DEFAULT_DATASETS = ("math500", "aime-24", "aime-25")


def parse_k_values(raw_values: Sequence[int]) -> List[int]:
    if not raw_values:
        raise ValueError("At least one value must be provided to --pass-k.")
    seen = set()
    result: List[int] = []
    for value in sorted(raw_values):
        if value in seen:
            continue
        if value <= 0:
            raise ValueError("pass@k values must be positive integers.")
        seen.add(value)
        result.append(value)
    return result


def parse_extra_fields(pairs: Sequence[str]) -> Dict[str, str]:
    extra: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Could not parse '{pair}'. Expected key=value entries for --extra-field."
            )
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid key in --extra-field entry '{pair}'.")
        extra[key] = value
    return extra


def load_jsonl(path: pathlib.Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file {path} does not exist.")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
            if "problem" not in record or "answer" not in record:
                raise ValueError(
                    f"Dataset entries must include 'problem' and 'answer' keys "
                    f"(file={path}, line={line_no})."
                )
            records.append(record)
    return records


def normalize_answer(answer: object) -> str:
    """Keep a lightweight normalization to compare math answers."""
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""

    # Prefer the last \boxed{} expression if present.
    import re

    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]

    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace(" ", "").replace("\n", "")
    text = text.strip()
    return text.lower()


def extract_prompt(record: Dict[str, object]) -> str:
    prompt = record.get("question") or record.get("problem")
    if not prompt:
        raise KeyError("Dataset entry is missing 'problem' or 'question'.")
    return str(prompt)


def extract_problem_id(record: Dict[str, object]) -> Optional[str]:
    value = record.get("unique_id") or record.get("id")
    if value is None:
        return None
    return str(value)


@dataclass
class SampleOutcome:
    samples: List[str]
    normalized: List[str]
    gold: str

    def top1_correct(self) -> bool:
        return bool(self.normalized) and self.normalized[0] == self.gold

    def pass_at_k(self, k: int) -> bool:
        return any(pred == self.gold for pred in self.normalized[:k])


class LocalServerClient:
    def __init__(
        self,
        url: str,
        timeout: float,
        max_retries: int,
        retry_delay: float,
        extra_fields: Dict[str, str],
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_fields = extra_fields

    def generate(
        self,
        dataset: str,
        prompt: str,
        sample_id: int,
    ) -> Dict[str, str]:
        payload = {
            "dataset": dataset,
            "prompt": prompt,
            "sample_id": sample_id,
        }
        payload.update(self.extra_fields)
        data = json.dumps(payload).encode("utf-8")

        attempt = 0
        while True:
            try:
                req = urllib.request.Request(
                    self.url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    content_type = response.headers.get("Content-Type", "")
                    body = response.read()
                if "application/json" not in content_type:
                    raise ValueError(
                        f"Expected JSON response but received Content-Type '{content_type}'."
                    )
                return json.loads(body.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - we want to catch network errors.
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"Local server request failed after {self.max_retries} retries."
                    ) from exc
                time.sleep(self.retry_delay)


def evaluate_problem(
    dataset_name: str,
    total: int,
    idx: int,
    item: Dict[str, object],
    client: LocalServerClient,
    response_field: str,
    num_samples: int,
    pass_k_values: Sequence[int],
    request_interval: float,
) -> Dict[str, object]:
    prompt = extract_prompt(item)
    gold = normalize_answer(item["answer"])
    problem_id = extract_problem_id(item)
    samples: List[str] = []

    for sample_id in range(num_samples):
        response = client.generate(dataset_name, prompt, sample_id)
        if response_field not in response:
            raise KeyError(f"Response is missing '{response_field}' field: {response}")
        prediction = response[response_field]
        samples.append(prediction)
        if request_interval > 0 and sample_id != num_samples - 1:
            time.sleep(request_interval)

    normalized_predictions = [normalize_answer(pred) for pred in samples]
    outcome = SampleOutcome(samples=samples, normalized=normalized_predictions, gold=gold)

    top1_flag = outcome.top1_correct()
    pass_hits = {k: int(outcome.pass_at_k(k)) for k in pass_k_values}
    id_suffix = f"#{problem_id}" if problem_id else ""
    logline = (
        f"[{dataset_name}{id_suffix}] Problem {idx}/{total}: "
        f"top1={'correct' if top1_flag else 'incorrect'}"
    )

    return {"top1": top1_flag, "pass_hits": pass_hits, "logline": logline}


def evaluate_dataset(
    dataset_name: str,
    records: Sequence[Dict[str, object]],
    client: LocalServerClient,
    response_field: str,
    num_samples: int,
    pass_k_values: Sequence[int],
    request_interval: float,
    concurrency: int,
) -> Dict[str, float]:
    total = len(records)
    if not total:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")

    top1_correct = 0
    pass_k_hits = {k: 0 for k in pass_k_values}

    def run(idx: int, item: Dict[str, object]) -> Dict[str, object]:
        return evaluate_problem(
            dataset_name=dataset_name,
            total=total,
            idx=idx,
            item=item,
            client=client,
            response_field=response_field,
            num_samples=num_samples,
            pass_k_values=pass_k_values,
            request_interval=request_interval,
        )

    if concurrency == 1:
        for idx, item in enumerate(records, 1):
            result = run(idx, item)
            if result["top1"]:
                top1_correct += 1
            for k, hit in result["pass_hits"].items():
                pass_k_hits[k] += hit
            print(result["logline"])
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(run, idx, item): idx
                for idx, item in enumerate(records, 1)
            }
            for future in as_completed(futures):
                result = future.result()
                if result["top1"]:
                    top1_correct += 1
                for k, hit in result["pass_hits"].items():
                    pass_k_hits[k] += hit
                print(result["logline"])

    metrics = {"accuracy": top1_correct / total}
    for k, hits in pass_k_hits.items():
        metrics[f"pass@{k}"] = hits / total
    return metrics


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Math reasoning evaluation harness.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to evaluate. Defaults to math500, aime-24, aime-25.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("datasets"),
        help="Directory containing <dataset>.jsonl files.",
    )
    parser.add_argument(
        "--server-url",
        required=True,
        help="Local server URL that accepts POST requests with JSON payloads.",
    )
    parser.add_argument(
        "--response-field",
        default="answer",
        help="JSON field in the server response that holds the model answer.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to request per problem.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        nargs="+",
        default=[1],
        help="List of k values for pass@k computation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of times to retry a failed server request.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between retries.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout (seconds) for each server request.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Optional delay between back-to-back sample requests.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent problems to evaluate.",
    )
    parser.add_argument(
        "--extra-field",
        action="append",
        default=[],
        help="Additional key=value pairs to include in every request payload.",
    )

    args = parser.parse_args(argv)

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be a positive integer.")

    pass_k_values = parse_k_values(args.pass_k)
    if pass_k_values[-1] > args.num_samples:
        raise ValueError("max(pass@k) cannot exceed --num-samples.")

    extra_fields = parse_extra_fields(args.extra_field)

    client = LocalServerClient(
        url=args.server_url,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        extra_fields=extra_fields,
    )

    overall_metrics: Dict[str, List[float]] = {}
    for dataset in args.datasets:
        data_path = args.data_dir / f"{dataset}.jsonl"
        records = load_jsonl(data_path)
        metrics = evaluate_dataset(
            dataset_name=dataset,
            records=records,
            client=client,
            response_field=args.response_field,
            num_samples=args.num_samples,
            pass_k_values=pass_k_values,
            request_interval=args.request_interval,
            concurrency=args.concurrency,
        )
        print(f"\nDataset={dataset}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        for key, value in metrics.items():
            overall_metrics.setdefault(key, []).append(value)

    print("\n=== Macro Average Across Datasets ===")
    for key, values in overall_metrics.items():
        avg = sum(values) / len(values)
        print(f"{key}: {avg:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
