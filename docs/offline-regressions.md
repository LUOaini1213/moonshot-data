# Scoring and cache regression contracts

## Problems reproduced

- GSM8K's suffix comparison could accept `112` for target `12`; numeric
  normalization could also round two different answers to the same value.
  Malformed answers could crash scoring or match an invalid target.
- Exact, relaxed, reverse-exact and SQuAD scoring used truncating `zip` with a
  prediction-count denominator. Missing samples could disappear from a score.
- Reusing a runner database after changing the system prompt could reuse an old
  response. Cache hits also restored the old dataset label and row metadata,
  including coercing string labels such as `"True"` into other types.
- Relaxed matching removed newlines and tabs from predictions but retained them
  in scalar targets, unlike its treatment of alternative-target lists.

## Current contracts

The five deterministic metrics exercised here require nonempty, equally sized
batches of prompts, predictions and targets. Scalar strings/bytes and unsized
containers are rejected with `ValueError`; valid wrong predictions still receive
normal scores. Aggregate scores and per-row evidence cover the same samples.

GSM8K follows the template's case-sensitive `The answer is` marker. The last
marker wins; its first complete whitespace-delimited token must be a numeric
answer. Supported forms are signed ASCII integers or decimals, leading decimal
points, correctly grouped thousands, scientific notation, an optional `$`, `£`
or `€` prefix and one trailing sentence period. Complete values are compared
using `Decimal`, without float conversion or significant-digit rounding.
Attached units, fractions, malformed separators and nonfinite values are
unmatched. A malformed final answer cannot fall back to an earlier valid one.
Whitespace-separated explanatory prose may follow the numeric token.

Benchmark cache identity now includes both rendered user prompt and system
prompt, alongside the existing endpoint/recipe/dataset/template IDs. An
identical request can reuse its response and duration. The current dataset
target, target type, row index, selection seed and attack provenance remain
current; a corrected label does not require another model request. Old rows
with SQL `NULL` system prompts are compatible with the empty system prompt.
The schema is unchanged.

The sample-selection seed is not a model-generation setting. Endpoint ID is
still the connector identity: if the model, generation parameters or external
service behind an existing endpoint ID changes, use a new runner/database or
endpoint ID. This patch does not provide a fingerprint of provider settings or
detect external model updates.

## Existing contribution and evidence boundaries

Commit `e87f9ee92eb4d763bae75d72afcd4bac1345fdf1` retains the Singapore data
corrections submitted in [upstream PR #223](https://github.com/aiverify-foundation/moonshot-data/pull/223).
The three option/key regressions use that issue's **zero-based** example indices:
transport 15 and 26, and public housing 2. Name corrections are also protected.
These checks validate those corrections, not the factual accuracy of every
question in the asset collection. No examples are added, removed or reordered.

`datasets/cache.json` is derived metadata. Moonshot 0.7.6 checks the raw dataset
file's xxh64 hash and refreshes changed entries during normal listing; dataset
reading uses current examples. It is distinct from the persistent runner
prediction cache. The stored metadata cache is left unchanged here.

The original `llm-eval-experiments` checkout and historical results remain
pinned to their original data version. These changes do not rewrite or rerun
those experiments, and no new model-quality result is claimed.

## Verification

The suite uses actual Moonshot 0.7.6 types, public `Runner.run_recipes`, its
`Run` pipeline, the SQLite asset and result formatter. Tests replace asset
discovery, metric configuration IO, model transport and output capture so that
they do not require API keys or paid inference. Runner tests check both
prediction call counts and the targets passed to scoring/exported reports.

CI installs `requirements-review.txt`, runs `tests/`, and uploads JUnit, the
checkout commit and resolved dependencies. A pytest error or empty suite fails
the job. Upstream provider/GPU integration tests remain separate; the upstream
Black Duck service is restricted to its owning repository, where its URL and
credentials are configured. The offline workflow runs independently on forks.
