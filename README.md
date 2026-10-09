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

**Failures are classified on what they are, not on what the HTTP status suggests.**
Two cases from this build pull in opposite directions. Google answers a model you
are not entitled to with HTTP 429 and `limit: 0`, which looks like a rate limit and
is a permanent entitlement failure, so it is `not_entitled` and never retried.
Cloudflare answers a failed inference with HTTP 409, which looks like a client error
and is a transient backend failure on a well-formed request, so it is
`backend_failure` and is retried. The second reclassification also recovered scoring
coverage that had been lost to it, but that is a consequence of fixing a
misclassification rather than the reason for it: a failed prediction carrying a
request id is transient whether or not any coverage depends on it.

**Both success rates are reported**, first attempt and after retries. Either alone
misleads. The post-retry figure hides how much work success took; the first-attempt
figure hides that the work is often cheap. It is also the distinction a pipeline
decision actually turns on, which is whether retrying is affordable. Keeping both
visible means a reclassification that enables retries cannot quietly flatter the
result, because the pre-retry number stays on the page.

**Cost and latency are recorded per request.** A model that is 8% better at 20x the
cost is not better for most pipeline work.

**The per-axis scores are the result.** The `overall` column is an unweighted mean
across the four axes, present as a summary and nothing more. It deliberately does not
weight by prompt count: prompt fidelity and text rendering have three prompts each
while the grouped axes resolve to one scored set, and that ratio is a fact about how
the suite was written rather than about the models. An aggregate also hides the
per-axis differences this comparison exists to surface, so a reader scanning the one
number is reading the wrong column.

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

- This benchmark was built and run entirely on free tiers, with nothing spent. That
  constrained which providers are in it, so the comparison covers what can be
  measured for nothing rather than the strongest models available. Published list
  prices are in the reference table below so the cost argument can still be made,
  but no figure reported here was paid for.
- Gemini was excluded on evidence rather than on the pricing page. Both 3.1 image
  models answered every request with HTTP 429 and `limit: 0 requests per day on
  Free Tier`, so the free tier exists as a tier and grants no quota at all. A zero
  ceiling is an entitlement failure wearing a rate-limit code, and the harness
  records it as its own `not_entitled` kind rather than retrying something that
  waiting cannot change. The pricing page agreed, but the run is what settled it.
- Output resolution and compression are not held constant, and cannot be on these
  endpoints. Pollinations returns 768x768 JPEG at 34-61 KiB; both Workers AI models
  return 1024x1024 JPEG at 443-772 KiB, roughly ten times the bytes for under twice
  the pixels. Equalising was attempted and does not work: the documented Pollinations
  API takes `width` and `height` with a default of 1024, but the anonymous endpoint
  ignores them and serves `sana` at 768x768 regardless, measured twice; and
  flux-1-schnell exposes no size parameter at all. More pixels and lighter
  compression plausibly both help on text rendering and on judging brush texture, so
  part of any difference on those two axes belongs to the encoder rather than the
  model. The scoring view shows every image at the same display width, which blunts
  the resolution half of this and not the compression half.
- Latency on a free tier is not a stable property of a model, and is not averaged
  into one figure here. flux-1-schnell was observed in four windows on the same
  prompts: 27.4-58.6s (n=3), 1.92-7.22s (n=11), 5.18-38.34s (n=8), 2.10-4.67s (n=12).
  That reads as regime-switching rather than variance around a mean, so the windows
  are reported separately with their n and no pooled median is given for it. Maximum is
  reported beside median for the same reason: flux-2-klein-4b at a 15.71s median with
  a 118s maximum is a different proposition from one reliably at 15s, and the median
  alone hides it.
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
- Pollinations caches aggressively, so its cold-generation latency can only be
  measured on a prompt it has not served before. The figures gathered so far
  straddle a change in per-provider concurrency from 2 to 1, which against a
  rate-gated endpoint are two different conditions: median 5.62s over 5 samples at
  concurrency 2, and 5.23s over 6 at concurrency 1. The ranges overlap almost
  entirely, so the conditions are not distinguishable at these sample sizes, but
  they are reported separately rather than pooled.
- Success rate is not stable either, and one run does not establish it. Pollinations
  returned 41.7% and then 25.0% across two fully cold runs of the same suite, both
  n=12, and 91.7% on a run where ten of twelve responses came from its cache. Only
  the cold figures measure the provider, and even those disagree by a factor of 1.7.
- Set-scored axes produce one score per provider per group, so character
  consistency and style adherence each rest on a single judgement by a single
  scorer. Where some generations in a group failed, the set is scored on the
  images that exist, which is a weaker test of consistency than a full set.
- Prompt ids appear inside some pass criteria ("same individual as cc_01"), so the
  rubric text reaches the scorer. That is a weaker join back to `results.json` than
  the ones closed above, and it is left in place because pass criteria are fixed
  before the run and are not reworded mid-flight.
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

Both Workers AI models draw on one account and so one 10,000 Neuron daily allowance,
which is why they share a rate gate rather than getting a concurrency cap each. A full
12-prompt run across both spends about 1,942 Neurons, under a fifth of one day.

| Provider | Tier used | List price / image | Derivation | Source |
|---|---|---|---|---|
| `cloudflare-flux-1-schnell` | Workers free allocation | **$0** in use | 10,000 Neurons/day at no charge, no payment method. 4.80 Neurons per 512x512 tile plus 9.60 per step; measured output is 1024x1024, so 4 tiles at 4 steps is 57.6 Neurons per image and 691 for a 12-prompt run. List rate is $0.0000528 per tile and $0.0001056 per step, so about $0.00063 per image if it were billed. | [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/), read 2026-10-09 |
| `cloudflare-flux-2-klein-4b` | Workers free allocation | **$0** in use | Same allowance. 26.05 Neurons per output 512x512 tile with no step multiplier, so 104.2 per 1024x1024 image and 1,250 for a 12-prompt run. Cheapest of the six priced Workers AI image models after schnell; the rest cost 1,300 to 2,600 Neurons per image and would exhaust the daily allowance inside a single run. | [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/), read 2026-10-09 |
| `pollinations` | Anonymous, keyless | **$0**, gated rather than billed | Unpaid requests are refused with HTTP 402, not charged. The x402 challenge asks 10000 base units of USDC (6 decimals) = 0.01 USDC, so ~$0.01 is the price of not being gated. | Measured from the `payment-required` response header, 2026-10-08 |

Gemini is absent from this table because it is absent from the benchmark. See the
limitation above for how that was established.

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
procedure was executed rather than merely described.

Reproducibility is split. Pollinations requests carry a random seed, recorded per
image in `results.json`, and the seed is honoured: two independent generations from
one seed returned byte-identical images, so those requests can be re-issued. The
Workers AI models expose no seed, so their outputs cannot be regenerated at all. The
case for committing a run therefore rests on the Workers AI images, which are a
record rather than build output.

## Results

<!-- TODO after group E. Leave empty until a real scored run exists. No placeholder
     numbers. Cover:
       - which provider won on which axis, and by how much
       - the Pollinations x402 failure rate as a finding about free-tier
         reliability, not as noise
       - Pollinations latency split by cache hit and miss, with the caveat that
         suite prompts were already warm by the time the reference run was made
       - anything that contradicted expectation
       - what the numbers do not support
       - lead the Pollinations finding with fast, free and unreliable rather than
         with the failure rate alone: 3.28s median against 15.71 and 32.50 is a real
         advantage for anything that can absorb the failures
       - report flux-1-schnell latency as three windows with their n, not pooled -->

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

That was tested during this build rather than assumed. Gemini 3.1 moved image
generation off `models/{id}:generateContent` onto `/v1beta/interactions`, with a
different request body and a different place for the image bytes, and the change was
contained to one class.

A Hugging Face adapter was also written and then removed. It had never executed
against the real API, and untested code in a project about measurement validity is
the wrong thing to ship -- the same reasoning that removed `--repeats` rather than
finishing it. Adding a provider back is one class.
