# Domain-correctness invariants

Executable "canonical-output invariant" checks for the scientific domain skills
(biology, ecology). Each test encodes a property that **must hold if the analysis
is done with the correct field-standard library** and would **fail for the
plausible-but-wrong hand-rolled shortcut** the skills warn against.

## What these guard
1. **The recipes we ship are correct** — every recipe in a skill's
   `references/*-recipe.md` has a matching invariant here, so a wrong recipe (or a
   library API change) is caught.
2. **A regression tripwire** for the domain libraries themselves.

They do **not** run the agent — the runtime backstop against a fresh hallucination
is the `self-check` block embedded in each recipe. These tests validate the
reference the agent copies from.

## Running
These need the scientific stack (scanpy, geopandas, …) which lives in the
**sandbox image**, not the lean worker. In lean tests an absent optional library
skips. Promotion mode fails closed on every skip:

```
# inside the code-execution sandbox image (bisque-ultra-codeexec:py311):
ULTRA_FAIL_ON_DOMAIN_SKIP=1 python -m pytest \
  backend/deepagents_runtime/tests/domain_correctness/ -v
```
