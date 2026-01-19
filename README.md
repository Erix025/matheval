# Math Evaluation Harness

This repository provides a lightweight framework to evaluate reasoning models on the **math500**, **aime-24**, and **aime-25** datasets. The harness communicates with a local model server, collects responses, and reports accuracy and pass@k metrics.

## Dataset Format

Datasets live in `datasets/` as `<dataset>.jsonl`. Each line is a JSON object with at least:

```json
{"problem": "Problem statement...", "answer": "Gold answer"}
```

Sample placeholders for every dataset are already included. Replace them with the official problems/answers when you are ready to run an evaluation.

* `answer` may be a string or a number—the harness normalizes both (along with predictions) before scoring.
* Entries may optionally expose metadata such as `id`, `unique_id`, `subject`, etc. `evaluate.py` logs the IDs and **prefers** a `question` field when present (falling back to `problem`). This allows datasets like AIME-24 (where `problem` stores extra commentary) to expose the true prompt separately.

## Local Server Contract

`evaluate.py` expects a local HTTP endpoint capable of returning model answers.

* Method: `POST`
* Payload: JSON object containing `dataset`, `prompt`, `sample_id`, along with any `--extra-field key=value` flags you pass to the script.
* Response: JSON object including an `answer` field (customizable via `--response-field`).

You can wrap any inference stack with a small Flask/FastAPI server that adheres to the contract above.

## Supported Backends

`evaluate.py` now understands three request styles:

1. **Custom server** (`--backend custom`, default): identical to the original contract above.
2. **OpenAI-compatible chat completions** (`--backend openai`): sends prompts via `/v1/chat/completions` as described in the [OpenAI-compatible completions guide](https://docs.sglang.io/basic_usage/openai_api_completions.html). Provide `--model`, `--api-key`/`OPENAI_API_KEY`, and optional `--system-prompt`/`--extra-field temperature=0.2 max_tokens=512`.
3. **SGLang native `/generate` endpoint** (`--backend sglang`): posts `{"text": prompt, "sampling_params": {...}}` following the [send_request reference](https://docs.sglang.io/basic_usage/send_request.html). Attach sampling options with `--extra-field temperature=0.1 max_new_tokens=256`.

Example invocations:

```bash
# OpenAI server
python3 evaluate.py --backend openai --server-url https://api.openai.com \
  --model gpt-4o-mini --api-key "$OPENAI_API_KEY" --datasets math500 --num-samples 2

# Local SGLang runtime
python3 evaluate.py --backend sglang --server-url http://localhost:30000 \
  --extra-field temperature=0 --extra-field max_new_tokens=256
```

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
