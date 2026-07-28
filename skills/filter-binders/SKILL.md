---
name: filter-binders
description: >-
  Triage a pool of candidate binders into keep/cull priority tiers by comparing
  each binder's metric value against user-supplied cutoffs, with a survivor guard
  that rejects any filter that would over-cull the pool. Use after computing a
  metric (folding confidence, binding energy, a peptide descriptor, contact
  count) across a binder pool, when you need to bucket candidates into priority
  levels without wiping the pool out. Pure logic — no dependencies.
---

# Filter binders (bucket-by-cutoffs + survivor guard)

A reusable prioritization algorithm: given a pool of binders each carrying a
chosen metric value, assign every binder a priority tier by comparing its metric
to a set of thresholds, and refuse the whole filter if applying it would leave
too few survivors. Pure Python logic — no library, no model, no network.

This is a **self-contained, instruction-only skill**. There is no wrapper script
and no project dependency — you (the agent) supply the metric values and run the
bucketing function below. It has **no dependencies**.

## When to use it

- You have computed a per-binder metric (folding confidence, binding energy, a
  `peptide-descriptors` scalar, a `contact-eval` count) across a pool and want to
  triage them into keep/cull priority tiers.
- You want a principled cull that will not accidentally empty the pool.

## Procedure

Each binder gets the **highest** priority tier whose threshold it clears.

- **Cutoffs** are a `{priority_int: threshold}` map, e.g.
  `{3: 0.9, 2: 0.7, 1: 0.5}`. A binder clearing none of the thresholds gets
  **priority 0** (culled).
- **direction** decides what "clears" means:
  - `"decreasing"` — higher metric is better; a value passes a threshold when
    `value >= threshold`.
  - `"increasing"` — lower metric is better; a value passes when
    `value <= threshold`.
- **Metric value resolution** — prefer a per-binder `objectives` dict keyed by
  `metric_name`; otherwise fall back to a scalar column named `metric_name`.
- **Survivor guard** — before committing, count how many binders would remain at
  `priority >= 1`. If that count is fewer than
  `max(1, ceil(min_survivor_fraction * pool_size))`, **reject the entire filter**
  and return the rejection reason instead of over-culling.

Inputs: `metric_name`, `cutoffs`, `direction`, `min_survivor_fraction`.
Output: `decisions` — a list of `{sequence, new_priority, value}` — or, if the
guard trips, `{rejected: True, rejection_reason}`.

```python
import math

def filter_binders(binders, metric_name, cutoffs, direction="decreasing",
                   min_survivor_fraction=0.1):
    """
    binders: list of dicts, each ideally with 'sequence', and either an
        'objectives' dict keyed by metric_name or a scalar column metric_name.
    cutoffs: {priority_int: threshold}, e.g. {3: 0.9, 2: 0.7, 1: 0.5}.
    direction: 'decreasing' (higher is better, value >= threshold passes) or
        'increasing' (lower is better, value <= threshold passes).
    """
    def resolve(b):
        obj = b.get("objectives")
        if isinstance(obj, dict) and metric_name in obj:
            return obj[metric_name]
        return b.get(metric_name)

    def passes(value, threshold):
        if value is None:
            return False
        return value >= threshold if direction == "decreasing" else value <= threshold

    tiers = sorted(cutoffs.items(), key=lambda kv: kv[0], reverse=True)  # highest first

    decisions = []
    for b in binders:
        value = resolve(b)
        priority = 0
        for prio, threshold in tiers:
            if passes(value, threshold):
                priority = prio
                break
        decisions.append({
            "sequence": b.get("sequence"),
            "new_priority": priority,
            "value": value,
        })

    # survivor guard
    pool_size = len(binders)
    survivors = sum(1 for d in decisions if d["new_priority"] >= 1)
    required = max(1, math.ceil(min_survivor_fraction * pool_size))
    if survivors < required:
        return {
            "rejected": True,
            "rejection_reason": (
                f"filter on '{metric_name}' would leave {survivors} survivors "
                f"(< required {required} of {pool_size}); rejected to avoid "
                f"over-culling the pool"
            ),
        }

    return {"decisions": decisions}
```

## Interpreting the result

- `decisions` — each binder's `new_priority` (0 = culled, higher = keep with
  greater confidence) alongside the `value` it was judged on.
- `{rejected: True, ...}` — the cutoffs were too aggressive for this pool. Loosen
  the thresholds, flip `direction` if it was set wrong, or lower
  `min_survivor_fraction`, then re-run. Do **not** apply a rejected filter.

## Notes

- This skill is only the bucketing rule. **You (the agent) decide which metric to
  filter on and gather its values** across the pool — e.g. by running the
  `peptide-descriptors` or `contact-eval` skill first — then feed them here.
- Run it repeatedly with different metrics to layer priorities, but re-check the
  survivor guard after each pass so successive filters do not empty the pool.
