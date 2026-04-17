# Golden Task Harness

`src/evals/golden_tasks.py` is the first typed scaffold for buyer-grade workflow evals.

## What it does

- defines a durable `GoldenTaskCase` schema with explicit inputs and expectations
- scores each response against contract density and response-quality metrics
- aggregates suite-level pass/fail counts for regression reporting

## What belongs in a case

- a real scientific prompt or workflow request
- any attached file IDs, resource URIs, dataset URIs, or tool allowlists
- expectations that matter for product diligence:
  - required terminology
  - forbidden unsupported claims
  - minimum evidence and measurement density
  - maximum acceptable meta narration
  - minimum answer completeness

## Recommended next step

Create a private operator dataset with 5-10 canonical workflows, run them through `/v3`, and save both the raw responses and the resulting suite report. That is the beginning of a diligence artifact a frontier lab can actually trust.
