# evalgate

**A regression gate for LLM applications.** Score your feature against a golden
dataset, diff it against a committed baseline, and fail the build when it gets
worse.

[![CI](https://github.com/Waddles1729/evalgate/actions/workflows/ci.yml/badge.svg)](https://github.com/Waddles1729/evalgate/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The problem

Most teams shipping an LLM feature can tell you their latest prompt "feels
better." Very few can tell you whether last Thursday's change made the refund
answers worse, because nothing measured it. The PoC works, the demo lands, and
then quality becomes a matter of opinion — until a customer finds the case where
the assistant gives medical advice it was supposed to refuse.

The missing piece is not a dashboard. It is **a test that fails**.

evalgate is that test. It runs in CI, it compares against a number you
committed, and it blocks the merge when the number gets worse.

```
score 0.442  (-0.527 vs baseline 0.969)

gate: FAIL
  ✗ must-pass cases failed: cancel-billing-period, medical-refusal, legal-refusal
  ✗ overall score fell by 0.527, over the allowed 0.020 (0.969 → 0.442)
  ✗ cases that were passing now fail: refund-window, password-reset, … (+2 more)
```

## Try it — no API key, no network

The bundled example ships with a deterministic offline provider, so the numbers
below are the numbers you will get.

```bash
git clone https://github.com/Waddles1729/evalgate
cd evalgate
pip install -e .

cd examples/support-bot
evalgate run -s evalgate.yaml
```

```
support-bot — fake:fake-1
score 0.969  (-0.000 vs baseline 0.969)
cases 7/7 passed  ·  0.0s  ·  18 cached

  brevity              ████████████████████ 1.000
  closeness            ████████████████████ 1.000
  correctness          ████████████████████ 1.000
  grounded             ███████████████····· 0.750
  policy               ████████████████████ 1.000

gate: PASS
```

Now break it. Point the suite at the older, vaguer prompt and run again:

```bash
sed -i 's|prompts/v2.txt|prompts/v1.txt|' evalgate.yaml
evalgate run -s evalgate.yaml --gate ; echo "exit=$?"
```

```
score 0.442  (-0.527 vs baseline 0.969)
cases 0/7 passed

  closeness            ████················ 0.208  -0.792
  correctness          ████················ 0.179  -0.821
  policy               ██████████████······ 0.714  -0.286

failing cases (7)
  medical-refusal  [must-pass]  0.105
    policy: contains forbidden text: i'd guess, probably nothing serious
    correctness: token overlap with the reference is 0.00
  …

gate: FAIL
exit=1
```

That exit code is the whole point.

## How it fits your project

The unit of evaluation is **your application**, not a bare prompt. Point
evalgate at a function and it will measure whatever that function does —
retrieval, tools, a multi-step agent graph, all of it:

```python
# myapp/evals.py
from evalgate import Case
from myapp.agent import answer_question

def answer(case: Case) -> str:
    return answer_question(case.input, user_id="eval-harness")
```

```bash
evalgate gate --answer-fn myapp.evals:answer
```

## The suite file

One YAML file, reviewable in a pull request like any other test:

```yaml
name: support-bot
dataset: dataset.jsonl
prompt: prompts/v2.txt
baseline: baseline.json

provider:
  name: openai          # or anthropic, or fake
  model: gpt-4o-mini

judge:
  name: anthropic       # judging with a different family than you generate with
  model: claude-sonnet-4-20250514   # avoids a model grading its own habits

scorers:
  # Cheap, deterministic, and carries most of the signal.
  - name: policy
    type: contains
    weight: 2.0
    threshold: 1.0
    none_of: ["I'd guess", "you would win"]

  - name: brevity
    type: length
    weight: 0.5
    max_words: 60

  # The judge handles only what the cheap scorers cannot.
  - name: correctness
    type: judge
    weight: 2.0
    threshold: 0.6
    criteria: factual correctness against the reference answer

  - name: grounded
    type: faithfulness
    weight: 1.5
    threshold: 0.6
    skip_tags: [refusal, fallback]

gate:
  min_score: 0.60            # absolute floor
  max_regression: 0.02       # tolerated drop vs baseline
  fail_on_must_pass: true    # a must-pass case failing always blocks
  fail_on_new_case_failure: true
```

The dataset is JSONL, one case per line, so adding a case is a one-line diff:

```jsonl
{"id": "cancel-billing-period", "input": "If I cancel today does my plan stop immediately?", "expected": "You can cancel at any time from Settings > Billing > Cancel plan. Your plan stays active until the end of the current billing period.", "tags": ["billing"], "must_pass": true, "context": ["Cancelling is done in Settings > Billing > Cancel plan.", "A cancelled plan remains active until the end of the current billing period."]}
```

## Scorers

| Type | Needs a model? | What it answers |
| --- | --- | --- |
| `exact_match` | no | Is the answer exactly the expected string? |
| `contains` | no | Does required text appear — and forbidden text not? |
| `regex` | no | Does the answer match (or avoid) a pattern? |
| `json_valid` | no | Is it parsable JSON with the required keys? |
| `length` | no | Is it inside the word budget? |
| `similarity` | embeddings | How close is it to the reference? |
| `judge` | yes | How correct is it, on a stated rubric? |
| `faithfulness` | yes | Is every claim supported by the retrieved context? |
| `answer_relevancy` | yes | Does it answer the question that was asked? |

Add your own in three lines:

```python
from evalgate.scorers import Scorer, ScoreResult, register

class NoPIIScorer(Scorer):
    type = "no_pii"
    def score(self, case, answer, *, judge):
        found = my_pii_detector(answer)
        return ScoreResult(0.0 if found else 1.0, f"found {found}" if found else "clean")

register(NoPIIScorer)
```

## Four decisions worth explaining

**Judges vote on a rubric, not a float.** Asking a model for "a score from 0 to
1" produces 0.8 for everything. Asking it to choose between five labelled
levels produces something stable enough to diff week over week. The mapping
back to 0–1 stays in our code, where it is testable.

**Every model call is cached on a hash of the request.** Editing one case and
re-running should cost one call, not the whole dataset. This is the difference
between evals that run on every pull request and evals that run once a quarter
because someone noticed the bill.

**Scorers can be scoped to tags.** Faithfulness against retrieved context is
meaningless for a policy refusal — the refusal is correct *because* it ignores
the context. Scoring it anyway drags the average down for a reason nobody can
act on, which is how teams learn to ignore their own dashboard. `skip_tags`
exists so the number stays worth reading.

**Must-pass cases are gated individually.** Averages hide the cases you cannot
ship without: the refusal that has to stay a refusal, the regulated disclaimer,
the customer bug you already promised was fixed. Those fail the build on their
own, no matter what the mean did.

## CI

Drop in the bundled action:

```yaml
- uses: Waddles1729/evalgate@v0
  with:
    suite: evals/evalgate.yaml
    comment: true          # post the report on the PR
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

It writes `report.md` (a PR comment) and `report.json` (for anything
downstream), and exits non-zero when the gate fails.

## Commands

```bash
evalgate run        # run and report
evalgate gate       # run, report, exit non-zero on a regression
evalgate baseline   # record the current scores as the baseline
evalgate show       # describe the suite without running it
evalgate clear-cache

# useful while iterating
evalgate run --tag refusal        # only the cases you are working on
evalgate run --case medical-refusal
```

## Install

```bash
pip install -e .        # PyYAML is the only runtime dependency
```

Provider calls go through `urllib`, so adding evalgate to a project does not
pull two vendor SDKs into its dependency tree.

## License

MIT.
