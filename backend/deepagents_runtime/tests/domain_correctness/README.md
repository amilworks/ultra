# Domain-correctness invariants

Executable "canonical-output invariant" checks for the scientific domain skills
(materials, biology, ecology). Each test encodes a property that **must hold if
the analysis is done with the correct field-standard library** and would **fail
for the plausible-but-wrong hand-rolled shortcut** the skills warn against — e.g.
"the cubic IPF-Z colour key maps 001→red, 101→green, 111→blue" (a hand-rolled
`R=x,G=y,B=z` mapping paints it blue and fails this test).

## What these guard
1. **The recipes we ship are correct** — every recipe in a skill's
   `references/*-recipe.md` has a matching invariant here, so a wrong recipe (or a
   library API change) is caught.
2. **A regression tripwire** for the domain libraries themselves.

They do **not** run the agent — the runtime backstop against a fresh hallucination
is the `self-check` block embedded in each recipe (the agent copies and runs it
in-sandbox). These tests validate the reference the agent copies from.

## Running
These need the scientific stack (orix, scanpy, geopandas, …) which lives in the
**sandbox image**, not the lean worker. Each test is `pytest.importorskip`-guarded,
so it SKIPS cleanly where a lib is absent and RUNS where present:

```
# inside the code-execution sandbox image (bisque-ultra-codeexec:py311):
python -m pytest backend/deepagents_runtime/tests/domain_correctness -v
```

In lean CI they skip. To actually exercise them, run this dir against the sandbox
image (a dedicated CI job / scheduled run) — see the follow-up in the plots-diagrams
planning doc.
