/**
 * Paste hygiene — a copied chat answer must land as content, not chrome.
 *
 * The fixture mirrors the real shapes: KaTeX's double DOM (MathML +
 * aria-hidden HTML rendering of the same formula), the chat code block with
 * its language header and copy button, and a GFM task list with checkbox
 * inputs. Reproducibility: each rewrite is pinned here, so a paste behaves
 * the same way every time.
 */

import { describe, expect, it } from "vitest";

import { sanitizePastedHtml } from "./notesPaste";

const CHAT_FRAGMENT = `
<div class="pk-markdown">
  <p>The loss is
    <span class="katex">
      <span class="katex-mathml"><math><semantics><mrow><mi>L</mi></mrow><annotation encoding="application/x-tex">\\mathcal{L} = -\\sum_i y_i \\log \\hat{y}_i</annotation></semantics></math></span>
      <span class="katex-html" aria-hidden="true"><span class="base"><span class="mord">L</span></span></span>
    </span>
    per batch.</p>
  <span class="katex-display">
    <span class="katex">
      <span class="katex-mathml"><math><semantics><mrow></mrow><annotation encoding="application/x-tex">\\hat{y} = \\operatorname{softmax}(Wx + b)</annotation></semantics></math></span>
      <span class="katex-html" aria-hidden="true"><span class="base"><span class="mord">softmax</span></span></span>
    </span>
  </span>
  <div class="not-prose group/codeblock flex w-full">
    <div class="flex items-center justify-between">
      <span>Python</span><span>Programming language</span>
      <button type="button" aria-label="Copy code">Copy</button>
    </div>
    <div class="relative">
      <div class="pk-code-render w-full"><pre class="shiki"><code><span>for epoch in range(30):</span>
<span>    train(model)</span></code></pre></div>
    </div>
  </div>
  <ul>
    <li class="task-list-item"><input type="checkbox" checked disabled> re-run the dashboard</li>
    <li class="task-list-item"><input type="checkbox" disabled> export the figure</li>
  </ul>
</div>`;

describe("sanitizePastedHtml", () => {
  const output = sanitizePastedHtml(CHAT_FRAGMENT);

  it("recovers raw TeX from KaTeX and emits math atoms — never doubled symbol soup", () => {
    expect(output).toContain('data-math-inline=""');
    expect(output).toContain("\\mathcal{L} = -\\sum_i y_i \\log \\hat{y}_i");
    expect(output).toContain('data-math-block=""');
    expect(output).toContain("\\operatorname{softmax}(Wx + b)");
    // The MathML/HTML double render is gone entirely.
    expect(output).not.toContain("katex");
    expect(output).not.toContain("aria-hidden");
  });

  it("sheds code-block chrome and keeps the bare pre", () => {
    expect(output).toContain("for epoch in range(30):");
    expect(output).not.toContain("Copy code");
    expect(output).not.toContain("Programming language");
    expect(output).not.toContain("<button");
    expect(output).toMatch(/<pre[^>]*>/);
  });

  it("turns checkbox inputs into the editor's task-item dialect, state intact", () => {
    expect(output).toContain('data-item-type="task"');
    expect(output).toMatch(/data-checked="true"[^>]*> re-run the dashboard/);
    expect(output).toMatch(/data-checked="false"[^>]*> export the figure/);
    expect(output).not.toContain("checkbox");
  });

  it("passes clean HTML through untouched", () => {
    const clean = "<p>Just a <strong>plain</strong> paragraph.</p>";
    expect(sanitizePastedHtml(clean)).toBe(clean);
  });
});
