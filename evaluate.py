#!/usr/bin/env python3
"""
Simple reasoning evaluation harness for math datasets.

The script reads JSONL datasets, queries a local model server for each problem,
and reports accuracy along with pass@k metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def coerce_extra_value(raw: str) -> object:
    """Convert CLI-provided values into JSON-compatible Python objects."""
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


def coerce_extra_fields(extra: Dict[str, str]) -> Dict[str, object]:
    return {key: coerce_extra_value(value) for key, value in extra.items()}


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


def build_openai_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ValueError("OpenAI backend requires a non-empty --server-url.")
    suffix = "chat/completions"
    if base.endswith(suffix):
        return base
    if base.endswith("v1"):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"


def build_sglang_generate_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base:
        raise ValueError("SGLang backend requires a non-empty --server-url.")
    if base.endswith("generate"):
        return base
    return f"{base}/generate"


def post_json_with_retries(
    url: str,
    payload: Dict[str, object],
    timeout: float,
    max_retries: int,
    retry_delay: float,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    attempt = 0
    final_headers = {"Content-Type": "application/json"}
    if headers:
        final_headers.update(headers)

    while True:
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers=final_headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
            if "application/json" not in content_type:
                raise ValueError(
                    f"Expected JSON response but received Content-Type '{content_type}'."
                )
            return json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - propagate network issues after retries.
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"Request to {url} failed after {max_retries} retries."
                ) from exc
            time.sleep(retry_delay)


@dataclass
class SampleOutcome:
    samples: List[str]
    normalized: List[str]
    gold: str

    def top1_correct(self) -> bool:
        return bool(self.normalized) and self.normalized[0] == self.gold

    def pass_at_k(self, k: int) -> bool:
        return any(pred == self.gold for pred in self.normalized[:k])


class BackendClient:
    def generate(self, dataset: str, prompt: str, sample_id: int) -> str:
        raise NotImplementedError


class CustomServerBackend(BackendClient):
    def __init__(
        self,
        url: str,
        timeout: float,
        max_retries: int,
        retry_delay: float,
        extra_fields: Dict[str, str],
        response_field: str,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_fields = extra_fields
        self.response_field = response_field

    def generate(self, dataset: str, prompt: str, sample_id: int) -> str:
        payload: Dict[str, object] = {
            "dataset": dataset,
            "prompt": prompt,
            "sample_id": sample_id,
        }
        payload.update(self.extra_fields)
        response = post_json_with_retries(
            url=self.url,
            payload=payload,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        if self.response_field not in response:
            raise KeyError(f"Response is missing '{self.response_field}' field: {response}")
        return str(response[self.response_field])


class OpenAIBackend(BackendClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        retry_delay: float,
        extra_fields: Dict[str, object],
        system_prompt: Optional[str],
    ) -> None:
        self.url = build_openai_chat_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_fields = extra_fields
        self.system_prompt = system_prompt

    def generate(self, dataset: str, prompt: str, sample_id: int) -> str:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        payload.update(self.extra_fields)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = post_json_with_retries(
            url=self.url,
            payload=payload,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            headers=headers,
        )
        choices = response.get("choices")
        if not choices:
            raise KeyError(f"OpenAI response missing 'choices': {response}")
        choice0 = choices[0]
        message = choice0.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        if "text" in choice0:
            return str(choice0["text"])
        raise KeyError(f"OpenAI choice missing content: {choice0}")


class SGLangBackend(BackendClient):
    def __init__(
        self,
        base_url: str,
        timeout: float,
        max_retries: int,
        retry_delay: float,
        sampling_params: Dict[str, object],
        api_key: Optional[str],
    ) -> None:
        self.url = build_sglang_generate_url(base_url)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sampling_params = sampling_params
        self.api_key = api_key

    def generate(self, dataset: str, prompt: str, sample_id: int) -> str:
        payload = {
            "text": prompt,
            "sampling_params": dict(self.sampling_params),
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = post_json_with_retries(
            url=self.url,
            payload=payload,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            headers=headers,
        )
        if "text" not in response:
            raise KeyError(f"SGLang response missing 'text': {response}")
        return str(response["text"])


def create_backend_client(
    backend: str,
    server_url: str,
    timeout: float,
    max_retries: int,
    retry_delay: float,
    extra_fields: Dict[str, str],
    response_field: str,
    model: Optional[str],
    api_key: Optional[str],
    system_prompt: Optional[str],
) -> BackendClient:
    if backend == "custom":
        return CustomServerBackend(
            url=server_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            extra_fields=extra_fields,
            response_field=response_field,
        )

    if backend == "openai":
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI backend requires an API key via --api-key or OPENAI_API_KEY."
            )
        if not model:
            raise ValueError("OpenAI backend requires --model to be specified.")
        typed_extra = coerce_extra_fields(extra_fields)
        return OpenAIBackend(
            base_url=server_url,
            api_key=resolved_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            extra_fields=typed_extra,
            system_prompt=system_prompt,
        )

    if backend == "sglang":
        typed_sampling = coerce_extra_fields(extra_fields)
        resolved_key = api_key or os.environ.get("SGLANG_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        return SGLangBackend(
            base_url=server_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            sampling_params=typed_sampling,
            api_key=resolved_key,
        )

    raise ValueError(f"Unsupported backend '{backend}'.")


def evaluate_problem(
    dataset_name: str,
    total: int,
    idx: int,
    item: Dict[str, object],
    client: BackendClient,
    num_samples: int,
    pass_k_values: Sequence[int],
    request_interval: float,
) -> Dict[str, object]:
    prompt = extract_prompt(item)
    gold = normalize_answer(item["answer"])
    problem_id = extract_problem_id(item)
    samples: List[str] = []

    for sample_id in range(num_samples):
        prediction = client.generate(dataset_name, prompt, sample_id)
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

    record = {
        "index": idx,
        "problem_id": problem_id,
        "input": prompt,
        "outputs": samples,
        "normalized_outputs": normalized_predictions,
        "gold_answer": item["answer"],
        "normalized_gold": gold,
        "top1_correct": top1_flag,
        "pass_hits": pass_hits,
    }

    return {"top1": top1_flag, "pass_hits": pass_hits, "logline": logline, "record": record}


def evaluate_dataset(
    dataset_name: str,
    records: Sequence[Dict[str, object]],
    client: BackendClient,
    num_samples: int,
    pass_k_values: Sequence[int],
    request_interval: float,
    concurrency: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    total = len(records)
    if not total:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")

    top1_correct = 0
    pass_k_hits = {k: 0 for k in pass_k_values}
    problem_records: List[Optional[Dict[str, object]]] = [None] * total

    def run(idx: int, item: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
        return idx, evaluate_problem(
            dataset_name=dataset_name,
            total=total,
            idx=idx,
            item=item,
            client=client,
            num_samples=num_samples,
            pass_k_values=pass_k_values,
            request_interval=request_interval,
        )

    if concurrency == 1:
        for idx, item in enumerate(records, 1):
            current_idx, result = run(idx, item)
            if result["top1"]:
                top1_correct += 1
            for k, hit in result["pass_hits"].items():
                pass_k_hits[k] += hit
            print(result["logline"])
            problem_records[current_idx - 1] = result["record"]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(run, idx, item): idx
                for idx, item in enumerate(records, 1)
            }
            for future in as_completed(futures):
                current_idx, result = future.result()
                if result["top1"]:
                    top1_correct += 1
                for k, hit in result["pass_hits"].items():
                    pass_k_hits[k] += hit
                print(result["logline"])
                problem_records[current_idx - 1] = result["record"]

    metrics = {"accuracy": top1_correct / total}
    for k, hits in pass_k_hits.items():
        metrics[f"pass@{k}"] = hits / total
    finalized_records = [record for record in problem_records if record is not None]
    return metrics, finalized_records


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
        "--backend",
        choices=["custom", "openai", "sglang"],
        default="custom",
        help="Backend type to query: a custom local server, an OpenAI-compatible server, or the native SGLang /generate endpoint.",
    )
    parser.add_argument(
        "--model",
        help="Model name for OpenAI-compatible backends.",
    )
    parser.add_argument(
        "--response-field",
        default="answer",
        help="JSON field in the custom backend response that holds the model answer.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for OpenAI/SGLang servers. Falls back to the OPENAI_API_KEY or SGLANG_API_KEY environment variables.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Optional system prompt inserted before every user query for OpenAI-compatible backends.",
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
        default=3000.0,
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
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs"),
        help="Directory where per-dataset JSON outputs (records and summary) are stored.",
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

    client = create_backend_client(
        backend=args.backend,
        server_url=args.server_url,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        extra_fields=extra_fields,
        response_field=args.response_field,
        model=args.model,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
    )

    overall_metrics: Dict[str, List[float]] = {}
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        data_path = args.data_dir / f"{dataset}.jsonl"
        records = load_jsonl(data_path)
        metrics, problem_records = evaluate_dataset(
            dataset_name=dataset,
            records=records,
            client=client,
            num_samples=args.num_samples,
            pass_k_values=pass_k_values,
            request_interval=args.request_interval,
            concurrency=args.concurrency,
        )
        dataset_output_dir = args.output_dir / f"{dataset}-{run_timestamp}"
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        records_path = dataset_output_dir / "records.json"
        summary_path = dataset_output_dir / "summary.json"

        with records_path.open("w", encoding="utf-8") as f:
            json.dump(problem_records, f, ensure_ascii=False, indent=2)

        summary_payload = {
            "dataset": dataset,
            "timestamp": run_timestamp,
            "num_problems": len(records),
            "num_samples": args.num_samples,
            "backend": args.backend,
            "server_url": args.server_url,
            "pass_k": pass_k_values,
            "metrics": metrics,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

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
