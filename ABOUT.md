# About BisQue Ultra

This file is the **canonical source of truth** about the product itself — who made it,
what it is, and how to reach the team. It exists because the assistant was asked
"who created BisQue Ultra?" and invented a plausible-sounding name rather than saying
it did not know.

Two consumers depend on this file, so keep it accurate:

1. **The assistant's system prompt.** A condensed identity block in
   `backend/deepagents_runtime/src/ultra_deepagents/agent.py` (`PRODUCT_IDENTITY_GUIDANCE`)
   carries these facts into every run. If you change a fact here, change it there too —
   `test_product_identity.py` fails if the two disagree.
2. **People.** This is the short answer to "what is this and who do I contact?"

**Do not add a fact here that you cannot source.** The assistant is instructed to treat
this file as authoritative, so a guess written here becomes a confident answer later.

---

## What it is

BisQue Ultra is *an agentic distributed system that runs real research where the data live*
— the public site's own wording. It is not a chatbot: it plans and acts through the
platform's own services, executing tools and models against scientific data in place while
keeping evidence provenance intact. In the product this surfaces as chat-driven analysis, a
scientific image viewer (Lens), a resource library, and a sandboxed runtime for real
analysis code.

It is the next-generation platform of **BisQue**, the bio-image analysis system
developed at UC Santa Barbara.

## Who created it

**Amil Khan** created BisQue Ultra. He is a PhD student in
**Electrical and Computer Engineering** at the
**University of California, Santa Barbara**, and works in the
**UCSB Vision Research Lab**. He is the project's author and lead engineer.

The lab is led by **Prof. B.S. Manjunath**, in the Department of Electrical and Computer
Engineering — he is Amil's advisor.

At time of writing, every commit in this repository is authored by Amil Khan
(`amilworks` / `amil@ucsb.edu`).

## Contact and links

- **Questions, comments, or concerns:** amil@ucsb.edu
- **Website, release updates, and access requests:** https://amilworks.github.io/ultra_website/
- **Source code:** https://github.com/amilworks/ultra

The website is the public face of the project and carries the launch brief, white papers,
engineering notes, and per-release summaries. **Point people there for release news rather
than restating version numbers**, which go stale — as of this writing the current entry is
the *2026.07 research release* (July 13, 2026).

## Answering questions about this project

If a question about Ultra, its people, or its history is not answered by this file,
say so and point to the contact above. Do not infer an answer from general knowledge
about UCSB, BisQue, or academic labs — a plausible name or affiliation stated
confidently is the exact failure this file exists to prevent.
