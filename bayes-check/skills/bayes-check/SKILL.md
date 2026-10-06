---
name: bayes-check
description: Verify the factual and quantitative claims of an answer with deterministic Bayesian scoring (bayes_score tool) before finalizing, then rewrite to honor each verdict.
version: 0.1.0
---

# bayes-check — Bayesian claim verification

This skill is the procedure. Enforcement lives in the bayes-check plugin: on an opted-in session
(`/bayes on`) every final answer is annotated with the computed verdicts, or marked
`⚠ bayes-check: unscored` when you skipped the tool. The plugin makes no model calls; the math
runs in Python, never in your head.

## When

Any turn where bayes-check says it is active, and any research, advisory, statistical, historical,
or "is it true that…" answer where a confident-sounding wrong claim would mislead.

## Procedure

1. **Draft** the answer internally. Do not send it yet.
2. **Extract claims.** Every sentence that asserts a checkable fact: numbers, dates, names,
   rankings, causal statements, attributions, API/code behavior. Skip opinions, plans, and
   restatements of the user's own words. Give each a short id (`c1`, `c2`, …), the exact text,
   and a `claim_type` (statistic, date, causal, definition, attribution, code_behavior, …).
3. **Set an honest prior** (0–1): how likely the claim is true *before* the evidence below.
   Omit it to use the learned base rate for that claim type. A claim you merely "remember" is
   not evidence — list it as `model_knowledge`, which barely moves the posterior.
4. **List only evidence you actually observed this session**, typed by `source_type`:

   | source_type | LR (supports) | use for |
   |---|---|---|
   | primary_source | 4.0 | the original document, dataset, spec, statute |
   | peer_reviewed | 3.5 | published peer-reviewed work |
   | official_docs | 3.0 | vendor/project documentation |
   | code_or_data | 3.0 | code you read, data you computed |
   | tool_output | 2.5 | output of a tool you ran this turn |
   | reputable_secondary | 2.0 | established press, textbooks |
   | web_search | 1.5 | a search snippet |
   | user_provided | 1.5 | the user said so |
   | model_knowledge | 1.2 | your own recall |
   | anecdotal | 1.1 | forum posts, hearsay |

   Set `stance: contradicts` when a source disagrees — that divides by the LR. Several sources
   of the same type are discounted (they are usually correlated), so ten search snippets do not
   equal one primary source.
5. **Call `bayes_score` once** with all claims.
6. **Rewrite** so every claim follows its `action`:
   - `state_plainly` (posterior ≥ 0.9 by default): keep as stated.
   - `qualify` (between): hedge explicitly — "likely", "reported by X", "I could not confirm".
   - `remove` (< 0.5 by default): cut it, or say plainly that it is unverified.
7. Finish. Do not write the footer yourself; the plugin appends it.

## Honesty rules

- Never invent or upgrade evidence to clear a threshold. A low posterior is the useful output.
- If you change a claim's wording in the rewrite, the footer still shows the scored wording;
  keep them aligned.
- This separates the checkable from the uncheckable. It does not make prose true; it makes it
  auditable.

## User feedback

`/bayes correct <hash> true|false` records ground truth for a footer line; `/bayes refit`
recalibrates the likelihood ratios from those labels. `/bayes status` shows what is on.
