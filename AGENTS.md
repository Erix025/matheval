# Repository Guidelines

This guide explains how to contribute effectively to the math evaluation harness while preserving consistent behavior across datasets and backends.

## Project Structure & Module Organization
- `evaluate.py` is the single entry point; it handles dataset loading, request dispatch, normalization, and scoring.
- `datasets/` hosts JSONL inputs (`math500.jsonl`, `aime-24.jsonl`, `aime-25.jsonl`). Replace the placeholder rows in-place rather than renaming files, because CLI defaults assume these names.
- `scripts/` is the staging area for ad hoc helpers such as dataset translators or log parsers; keep experimental utilities there.
- Add future tests beside their targets (e.g., `tests/test_evaluate.py`) so imports remain relative to the project root.

## Build, Test, and Development Commands
- `python3 evaluate.py --help` prints every CLI flag and backend option.
- `python3 evaluate.py --backend custom --server-url http://localhost:8000 --datasets math500 --num-samples 2` performs a fast sanity check against a local server.
- `python3 evaluate.py --backend openai --server-url https://api.openai.com --model gpt-4o-mini --api-key "$OPENAI_API_KEY"` runs the hosted-chat pathway; pair it with `--extra-field temperature=0`.
- Keep a dry-run dataset handy (two or three rows) so you can iterate without exhausting API quotas.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation and type hints (`List[str]`, `Optional[int]`).
- Match the functional style already in `evaluate.py`: pure helpers, descriptive verbs (`load_jsonl`, `normalize_answer`), and dataclasses for structured results.
- Prefer `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for module constants (e.g., `DEFAULT_DATASETS`), and docstrings for non-trivial helpers.
- Run `python3 -m black evaluate.py` or an equivalent formatter before sending a PR; the file currently follows Black’s defaults.

## Testing Guidelines
- There is no automated suite yet; add `pytest`-style cases under `tests/` when modifying answer normalization, retry logic, or CLI parsing.
- Include regression tests for any bug fix that affects comparison logic or HTTP payload formatting.
- Always run at least one end-to-end invocation on each dataset you touched; log accuracy and pass@k so reviewers can compare before/after numbers.

## Commit & Pull Request Guidelines
- Existing history (`git log --oneline`) shows short, imperative subjects (“Update datasets and prompt handling”). Follow that format and keep bodies for context or links.
- Pull requests should state the motivation, summarize behavioral changes, list datasets/backends validated, and mention any required secrets (e.g., `OPENAI_API_KEY`).
- Include terminal snippets or tables for accuracy changes plus screenshots only when working on visualization helpers.

## Security & Configuration Tips
- Never commit real competition data or API keys; load secrets from environment variables such as `OPENAI_API_KEY`.
- Throttle remote calls with `--request-interval` when pointing at rate-limited services, and keep a small `--num-samples` during validation to avoid accidental quota burns.
