# Math Evaluation Harness

This repository provides a lightweight framework to evaluate reasoning models on the **math500**, **aime-24**, and **aime-25** datasets. The harness communicates with a local model server, collects responses, and reports accuracy and pass@k metrics.

## Dataset Format

Datasets live in `datasets/` as `<dataset>.jsonl`. Each line is a JSON object with at least:

```json
{"problem": "Problem statement...", "answer": "Gold answer"}
```

Sample placeholders for every dataset are already included. Replace them with the official problems/answers when you are ready to run an evaluation.

## Local Server Contract

`evaluate.py` expects a local HTTP endpoint capable of returning model answers.

* Method: `POST`
* Payload: JSON object containing `dataset`, `prompt`, `sample_id`, along with any `--extra-field key=value` flags you pass to the script.
* Response: JSON object including an `answer` field (customizable via `--response-field`).

You can wrap any inference stack with a small Flask/FastAPI server that adheres to the contract above.

## Running an Evaluation

```bash
python3 evaluate.py \
  --server-url http://localhost:8000/generate \
  --num-samples 5 \
  --pass-k 1 3 5 \
  --extra-field temperature=0.2
```

Key flags:

* `--datasets`: subset of datasets to evaluate (defaults to all three).
* `--num-samples`: number of responses to request per problem.
* `--pass-k`: list of k values for pass@k (must be ≤ `--num-samples`).
* `--concurrency`: number of problems evaluated in parallel (speeds up local servers that support multiple simultaneous requests).
* `--request-interval`: optional delay between sample requests (useful when rate limiting your server).
* `--max-retries` / `--retry-delay`: retry behavior for flaky local endpoints.

The script prints per-dataset accuracy and pass@k, plus a macro average across the selected datasets.

## Answer Normalization

Before comparing predictions to gold answers, the harness:

1. Extracts the last `\boxed{...}` expression if present.
2. Removes common LaTeX markers and whitespace.
3. Lowercases the result.

You can adjust `normalize_answer` in `evaluate.py` if your dataset requires a different canonicalization.
