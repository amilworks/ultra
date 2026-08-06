/**
 * The fidelity gate for Markdown mode.
 *
 * The dual-mode contract's hard law: opening a note never rewrites it. The
 * page half of that law is behavioral (no save without a doc change); this
 * suite pins the engine half — parse → serialize through the REAL editor
 * pipeline (Milkdown schema + house stringify dialect + highlight handler)
 * must return the exact input, modulo the serializer's single trailing
 * newline (which the page trims on receipt).
 *
 * If an engine upgrade starts restyling any corpus entry, this fails and the
 * upgrade does not ship.
 */

import { describe, expect, it } from "vitest";

import {
  defaultValueCtx,
  Editor,
  remarkStringifyOptionsCtx,
  rootCtx,
} from "@milkdown/kit/core";
import { commonmark } from "@milkdown/kit/preset/commonmark";
import { gfm } from "@milkdown/kit/preset/gfm";
import { getMarkdown } from "@milkdown/kit/utils";

import { HIGHLIGHT_PATTERN } from "@/lib/remarkHighlight";
import { withNotesDialect } from "./notesDialect";
import { ultraHighlight } from "./notesHighlight";
import { notesMath } from "./notesMath";

const roundTrip = async (markdown: string): Promise<string> => {
  const root = document.createElement("div");
  document.body.appendChild(root);
  const editor = await Editor.make()
    .config((ctx) => {
      ctx.set(rootCtx, root);
      ctx.set(defaultValueCtx, markdown);
      ctx.update(remarkStringifyOptionsCtx, withNotesDialect);
    })
    .use(commonmark)
    .use(gfm)
    .use(ultraHighlight)
    .use(notesMath)
    .create();
  const serialized = editor.action(getMarkdown());
  await editor.destroy();
  root.remove();
  return serialized;
};

const expectStable = async (markdown: string) => {
  const once = await roundTrip(markdown);
  expect(once).toBe(`${markdown}\n`);
};

describe("markdown mode round-trips the house dialect byte-stable", () => {
  it("prose with bold, italic, inline code, and snake_case left unescaped", async () => {
    await expectStable(
      "Working notes for the CNN workshop. The **argmax provenance** demo lands hardest — and `MaxPool2d(2, 2)` halves each *spatial* dim.\n\nThe run wrote survey_2026_final.csv without drama."
    );
  });

  it("headings, dash bullets, and a task list", async () => {
    await expectStable(
      "## Why 2×2, stride 2\n\n- transect spacing 40 m\n- flag GSD < 1.2 cm\n\n### Follow-ups\n\n- [x] Re-run the dashboard with 30 epochs\n- [ ] Add an average-pooling comparison panel"
    );
  });

  it("a GFM pipe table with alignment", async () => {
    await expectStable(
      "| Pooling     | Params | Val acc |\n| ----------- | -----: | ------: |\n| Max 2×2     |      0 |   91.4% |\n| Average 2×2 |      0 |   89.9% |"
    );
  });

  it("fenced code keeps its language and body verbatim", async () => {
    await expectStable(
      "```python\nfor epoch in range(30):\n    train(model)  # x == y stays code\n```"
    );
  });

  it("blockquote, divider, ordered list", async () => {
    await expectStable(
      "> Decisions: keep NGFF v0.4 until viewer parity.\n\n---\n\n1. stage the store\n2. verify chunk paths"
    );
  });

  it("==highlight== round-trips as content, not styling sidecar", async () => {
    await expectStable(
      "The ==argmax provenance demo== lands hardest — highlight means ==this matters==."
    );
  });

  it("ultra:// media references survive untouched", async () => {
    await expectStable(
      "![pooled_grid.png](ultra://resource/file_9f21ab04/pooled_grid.png)\n\n[survey notes](ultra://resource/file_c0b8/notes.txt)"
    );
  });

  it("inline LaTeX rides remark-math syntax, exactly like chat", async () => {
    await expectStable(
      "The pooled output is $y_{ij} = \\max(x_{2i,2j}, x_{2i,2j+1})$ per window, and $E=mc^2$ stays inline."
    );
  });

  it("display math round-trips, including aligned environments", async () => {
    await expectStable(
      "$$\n\\hat{y} = \\operatorname{softmax}(Wx + b)\n$$\n\nand a multi-line derivation:\n\n$$\n\\begin{aligned}\n\\mathcal{L} &= -\\sum_i y_i \\log \\hat{y}_i \\\\\n&= \\text{cross-entropy}\n\\end{aligned}\n$$"
    );
  });

  it("empty table cells stay emptiness, never an html <br /> sentinel", async () => {
    const out = await roundTrip("| a | b |\n| -- | -- |\n| c |  |\n|  |  |");
    expect(out).not.toContain("<br");
    expect(out).toContain("| c |");
  });

  it("dollar amounts in prose stay prose when they are not math-shaped", async () => {
    // remark-math only claims $…$ with non-space flanks; "$5 and $10" has a
    // space after the opener, so it stays literal text.
    await expectStable("The reagent costs $5 and $10 per plate at bulk pricing.");
  });
});

describe("the highlight flanking rules stay conservative", () => {
  const matches = (value: string) => HIGHLIGHT_PATTERN.test(value);

  it("marks a plain span and tolerates a single equals inside", () => {
    expect(matches("==this matters==")).toBe(true);
    expect(matches("prefix ==E=mc²== suffix")).toBe(true);
  });

  it("never lights up comparisons, runs of equals, or word-adjacent spans", () => {
    expect(matches("x == y")).toBe(false);
    expect(matches("a====b")).toBe(false);
    expect(matches("====")).toBe(false);
    expect(matches("snake==case==word")).toBe(false);
    expect(matches("== spaced ==")).toBe(false);
  });

  it("stays literal when the closer touches a word", async () => {
    expect(matches("==open==ended")).toBe(false);
    // And an ambiguous span survives a round trip as literal text.
    await expectStable("The check x == y stays plain, as does a====b.");
  });
});
