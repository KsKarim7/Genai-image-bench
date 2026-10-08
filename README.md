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

The scorer's input is a purpose-built `scoring_manifest.json` holding only unit ids,
axis, scale, image paths, prompt text and pass criteria. It carries no provider name
and no field that joins to one: not response size, not latency, not content type, not
prompt id. `score.py` reads that file and nothing else.

**Sets are scored as sets.** Prompts declaring a `consistency_group` or `style_group`
form one scoring unit per provider, presented together and given a single score.
Asking whether three images show the same individual cannot be answered one shuffled
image at a time, which is what the first implementation asked. A unit never spans
providers, so showing the set together keeps it blind.

Members are presented in the order the suite declares them, so a consistency group's
character sheet is always the reference the other criteria point at. Style groups have
no privileged member, so their criteria are order-independent.

**The rubric is fixed before the run.** Per-prompt pass criteria live in
`config/prompts.yaml` and are written before any image is generated. Deciding what
counts as success after seeing the outputs is how a comparison turns into a
justification. The same rule forbids swapping prompts to get cleaner numbers.

One wording revision is on record. Six criteria named another prompt by id ("same
individual as cc_01"). Under set scoring the scorer never sees a prompt id, so those
ids pointed at something invisible. They now refer to the set, and for consistency
groups to the character-sheet reference that is always presented first. The substance
is unchanged and the revision predates any scored run. What the rule guards against
is a rubric that changes without a record, not one that changes.

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

Two further instances of the same pattern surfaced while fixing the first.

**Second: `results.json` was an unblinding oracle.** `score.py` read it for prompt
text, and every row in it carries `provider` beside `meta.bytes`. A file's size on
disk is exactly `meta.bytes`, so the provider behind any image was recoverable by
arithmetic, with `blind_map.json` never opened. The data structure handed to the
scorer had been carefully stripped; the file it was read from had not. `latency_s`
and `content_type` were two further join keys in the same row.

**Third: the fix for the `.png` bug created a new leak.** Recording the real format
made the extension track the response type, and response type tracks provider almost
perfectly, so sorting the image folder by extension separated the providers without
reading anything. Fixing one confound opened a channel. Filenames now carry no suffix
at all, format lives in the metadata, and images reach the scorer through an `<img>`
tag, which sniffs the bytes and does not need one.

Three occurrences, two of them created or overlooked while examining the first, is
the argument for `tests/test_blinding.py`. The property is one assertion to state and
demonstrably easy to break by accident; a careful read had already missed it twice.

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
- Set-scored axes produce one score per provider per group, so character
  consistency and style adherence each rest on a single judgement by a single
  scorer. Where some generations in a group failed, the set is scored on the
  images that exist, which is a weaker test of consistency than a full set.
- Prompt ids appear inside some pass criteria ("same individual as cc_01"), so the
  rubric text reaches the scorer. That is a weaker join back to `results.json` than
  the ones closed above, and it is left in place because pass criteria are fixed
  before the run and are not reworded mid-flight.
- Hugging Face returns HTTP 503 while a model cold-starts. That is classified as an
  integration error rather than a transient one, so it is not retried even though
  the message says to retry. Cold starts need their own retry budget; not done.
- Blind ids are 40 bits of UUID4 and run ids have one-second resolution. A collision
  in either would silently overwrite data rather than fail. Negligible at this scale
  and not guarded.
- Free-tier endpoints may serve lower resolution or different quotas than paid
  tiers, so latency and quality figures here are not representative of paid-tier
  performance.
- Cost figures are list prices, not measured billing. `LIST_PRICE_USD_PER_IMAGE` in
  `bench/providers.py` holds what each provider publishes for the tier this harness
  uses. Every configured provider runs on a free tier, so the results table reports
  "free tier" rather than a dollar figure; what a run would cost at paid-tier rates
  is a different claim and does not share that column.

## Reference: published pricing

List prices for the providers this benchmark actually runs, on the tier the harness
uses. These are published rates, not measured billing: what a run would cost, not what
the results table reports. The results table reports this run. API pricing moves, so
every figure below is dated and shows its derivation.

Batch and Flex tiers are cheaper and are deliberately not used. Batch roughly halves
the Gemini price, and destroys the latency measurement that is one of the four things
being compared.

| Provider | Tier used | List price / image | Derivation | Source |
|---|---|---|---|---|
| `gemini-3.1-flash-lite-image` | Standard, paid | **$0.0336** | Published per-image rate. Cross-checks against $30.00 per 1M output tokens at roughly 1,120 tokens per image. | [ai.google.dev pricing](https://ai.google.dev/gemini-api/docs/pricing), page last updated 2026-10-07, read 2026-10-09 |
| `pollinations` | Anonymous, keyless | **$0**, gated rather than billed | Unpaid requests are refused with HTTP 402, not charged. The x402 challenge asks 10000 base units of USDC (6 decimals) = 0.01 USDC, so ~$0.01 is the price of not being gated. | Measured from the `payment-required` response header, 2026-10-08 |

No Gemini image model has a free tier: Google lists "Free Tier: Not available" for all
of them as of 2026-10-09. `gemini-2.5-flash-image`, which this harness targeted first,
is deprecated, and Google's own documentation contradicts itself on when it goes away
-- the pricing page says it shut down on 2026-10-02, the deprecations table says
2027-03-15. That is left unresolved here because the harness moved to the named
replacement rather than depend on either date.

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
python -m unittest discover -s tests -t .
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
tests/               blinding, unit-grouping and report-join checks
run.py               entry point
```

## Adapting to API drift

Image APIs change shape often. Each adapter isolates one provider's request and
response handling behind `generate()`, so a breaking change touches one class. If a
provider starts failing with parse errors, that adapter is where to look.
