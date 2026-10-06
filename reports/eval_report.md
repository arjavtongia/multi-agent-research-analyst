# Evaluation report

Generated 2026-10-06 16:33 UTC by `research-agents eval`.

## Configuration

- Provider: `nvidia`, main model `nvidia/nemotron-3-super-120b-a12b`, fast model `openai/gpt-oss-20b`
- Search: `duckduckgo`, 3 sources per sub-question
- Max revisions: 2, token budget: 300,000
- Human review: auto-approved

## Summary

| Metric | Value |
|---|---|
| Questions run | 10 (0 errors) |
| Mean citation coverage | 90.7% |
| Mean supported-claim rate | 91.8% |
| Mean unique sources cited | 9.7 |
| Mean revisions used | 1.2 |
| Median latency (s) | 686.8 |
| Mean tokens per report | 104,659 |

## Per question

| id | topic | status | coverage | supported | sources cited | revisions | latency (s) | tokens |
|---|---|---|---|---|---|---|---|---|
| tech-llm-energy | technology | completed | 100.0% | 97.5% | 9 | 2 | 900.3 | 154,280 |
| tech-passkeys | technology | completed | 100.0% | 100.0% | 18 | 1 | 677.7 | 125,645 |
| tech-solid-state-batteries | technology | completed | 100.0% | 100.0% | 12 | 0 | 350.1 | 70,793 |
| econ-minimum-wage | economics | completed | 100.0% | 100.0% | 9 | 0 | 277.1 | 52,674 |
| econ-quantitative-tightening | economics | completed | 100.0% | 100.0% | 11 | 1 | 695.8 | 119,998 |
| econ-four-day-week | economics | completed | 6.8% | 33.3% | 3 | 2 | 562.3 | 88,070 |
| sci-microplastics | science | completed | 100.0% | 100.0% | 12 | 1 | 664.5 | 87,098 |
| sci-base-editing | science | completed | 100.0% | 87.5% | 8 | 2 | 736.2 | 111,441 |
| sci-fusion | science | completed | 100.0% | 100.0% | 7 | 1 | 708.6 | 103,114 |
| sci-direct-air-capture | science | completed | 100.0% | 100.0% | 8 | 2 | 774.7 | 133,479 |

## Metric definitions

- **Citation coverage**: factual sentences with at least one `[S#]` citation divided by all factual sentences. A sentence is factual if it has a citation, or has six or more words and is not a question, signpost or declared evidence gap.
- **Supported-claim rate**: cited factual sentences judged `supported` by the fact-checker on the final draft, divided by all cited factual sentences.
- **Unique sources cited**: distinct source ids cited in the final report.
- **Revisions used**: automatic rewrites triggered by the fact-checker (capped at 2).
- **Latency and tokens**: wall-clock seconds and total input+output tokens per run.

The fact-checker is itself an LLM, so the supported-claim rate is a model judgement, not ground truth. Spot-check a few reports under `runs/` before trusting the numbers.
