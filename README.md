# genai-image-bench

A benchmark harness for comparing text-to-image models on the criteria that matter
for production creative work, rather than on which output looks nicer.

Most public image-model comparisons are a handful of cherry-picked prompts rendered
side by side and scored by whoever ran them, knowing which model produced what. That
design yields numbers, but the numbers carry the evaluator's expectations as much as
the models' behaviour. This harness is built to avoid that, and the first version of
it failed to — see *Blinding, and how the first implementation broke it* below.

## What it measures

Four axes, each chosen because a studio pipeline breaks on it:

| Axis | Why it matters in production |
|---|---|
| **Prompt fidelity** | Can the model satisfy countable and spatial constraints ("three characters, the tallest on the left")? Art direction is mostly constraints. |
| **Text rendering** | Titles, signage and UI inside generated frames. Still the most common reason a generated asset can't ship without manual repair. |
| **Character consistency** | The same character across multiple shots. The hardest requirement for any narrative or storyboard use. |
| **Style adherence** | Holding a specified visual style across a set, rather than drifting toward the model's own default aesthetic. |

## Design decisions

**Blind scoring.** Outputs are presented shuffled, identified only by an opaque id
assigned at generation time. The id is the image's filename on disk, so the provider
name appears nowhere the scorer can see it. `blind_map.json` is the only artifact
that knows which provider produced which output, and it is read in exactly one place:
`report.py`, after scoring is complete.

**The rubric is fixed before the run.** Per-prompt pass criteria live in
`config/prompts.yaml` and are written before any image is generated. Deciding what
counts as success after seeing the outputs is how a comparison turns into a
justification. The same rule forbids swapping prompts to get cleaner numbers.

**Failures are data.** Timeouts, rate limits, quota rejections, refusals and
malformed responses are recorded with their reason rather than retried into
invisibility. A model that produces excellent images 60% of the time is a different
engineering proposition from one that is merely good every time, and an aggregate
quality score hides that.

**Cost and latency are recorded per request.** A model that is 8% better at 20x the
cost is not better for most pipeline work.

## Blinding, and how the first implementation broke it

The first version stripped the provider field out of the data structure handed to the
scorer, and wrote every image to disk as `{provider}__{prompt_id}__r{repeat}.png`.
The scoring CLI then printed that path to the scorer before asking for a score, and
opened the file in a browser, where the same string appeared in the title bar.

The careful-looking part of the code protected a channel that wasn't leaking. The
filename was.

This is worth recording rather than quietly fixing, because it is the same failure
mode the benchmark is designed to detect: a measurement that looks sound, produces
plausible numbers, and is compromised by a detail outside the part of the system
anyone was examining. Blinding that hasn't been verified end to end isn't blinding.

The fix assigns the opaque id at generation time and uses it as the filename, so no
provider-identifying string exists anywhere on the scorer's side of the boundary.

A related defect found in the same review: image bytes were written with a hardcoded
`.png` extension regardless of the actual response format. Every file was JPEG.
Browsers content-sniff, so the report rendered correctly and the bug stayed invisible
— while JPEG compression artifacts sat directly on top of the style-adherence
criteria, which ask the scorer to judge visible brush texture.

## What this does not claim

- Scores come from a single human scorer on a small prompt set. They indicate
  direction, not statistical significance. Inter-rater reliability would need
  multiple scorers; that hasn't been done.
- Blinding removes provider bias, not aesthetic preference. Judgement is still
  subjective.
- Pollinations serves cached results for previously-seen prompts with a one-year
  immutable cache header. Re-running the same suite against it inflates measured
  success rate and deflates measured latency, because cache hits bypass the
  provider's rate gate entirely. Only first-generation-on-fresh-prompts figures
  are measurements of the provider; `x-cache` is recorded per request so hits and
  misses can be separated.
- Free-tier endpoints may serve lower resolution or different quotas than paid
  tiers, so latency and quality figures here are not representative of paid-tier
  performance.
- Cost figures are estimates from published per-image pricing, held in
  `COST_PER_IMAGE` in `bench/providers.py`. They are not measured billing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in whichever keys you have
```

Providers without credentials are skipped with a notice and the run continues with
the rest. Pollinations needs no key, so the harness does something before anything
is configured — but it now gates the legacy image endpoint behind an x402
micropayment challenge, so expect a high failure rate on concurrent requests. See
the results for what that looks like in practice.

## Running

```bash
python run.py generate              # all prompts, all available providers
python run.py generate --axis text_rendering --repeats 3
python run.py score  <run_id>       # blind scoring pass
python run.py report <run_id>       # unblind, build report.html
python run.py runs                  # list runs
```

`generate` writes `runs/<run_id>/` — images, `results.json`, `blind_map.json`.
`score` writes `scores.json`. `report` joins them into `report.html` and
`summary.json`.

`runs/` is ignored by default; committing a run is a deliberate act. One curated,
scored and reported run lives in `runs/reference/` as evidence that the blind
procedure was executed rather than merely described — generations are unseeded and
not reproducible, so that run is a record, not build output.

## Results

<!-- TODO after group E. Leave empty until a real scored run exists. No placeholder
     numbers. Cover:
       - which provider won on which axis, and by how much
       - the Pollinations x402 failure rate as a finding about free-tier
         reliability, not as noise
       - Pollinations latency split by cache hit and miss, with the caveat that
         suite prompts were already warm by the time the reference run was made
       - anything that contradicted expectation
       - what the numbers do not support -->

## Repo layout

```
bench/providers.py   provider adapters behind one interface
bench/runner.py      async execution, concurrency limits, backoff, failure capture
bench/score.py       blind scoring CLI
bench/report.py      unblinding and HTML comparison grid
config/prompts.yaml  prompt suite and per-prompt pass criteria
run.py               entry point
```

## Adapting to API drift

Image APIs change shape often. Each adapter isolates one provider's request and
response handling behind `generate()`, so a breaking change touches one class. If a
provider starts failing with parse errors, that adapter is where to look.
