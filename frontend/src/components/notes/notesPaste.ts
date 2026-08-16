/* Paste hygiene for the Markdown surface.
 *
 * Real pastes are messy: a copied chat answer carries KaTeX's double DOM
 * (MathML + HTML render of the same formula — plain extraction DOUBLES every
 * symbol), code-block chrome ("Python", "Copy code" buttons), and GFM task
 * checkboxes as <input> elements; Word adds style/meta junk. This transform
 * runs before ProseMirror parses pasted HTML and rewrites those shapes into
 * DOM the notes schemas parse natively, so what lands in the note is the
 * CONTENT: formulas as math atoms, code as code, tasks as tasks.
 *
 * Everything here is a rewrite toward plainness — nothing is invented, and
 * unknown HTML passes through to ProseMirror's normal parsing untouched.
 */

const NEEDS_SANITIZING = /katex|pk-code-render|type="checkbox"|<button|<style|<script|<meta|shiki/i;

const texOf = (element: Element): string =>
  element.querySelector('annotation[encoding="application/x-tex"]')?.textContent?.trim() ?? "";

export const sanitizePastedHtml = (html: string): string => {
  if (!NEEDS_SANITIZING.test(html)) {
    return html;
  }
  const doc = new DOMParser().parseFromString(html, "text/html");
  const body = doc.body;

  // KaTeX → math atoms. Display wrappers first (they contain a .katex), then
  // any inline .katex left standing. The TeX source rides the annotation
  // node KaTeX always embeds; without one there is nothing trustworthy to
  // keep, so the render is dropped rather than pasted as symbol soup.
  for (const display of [...body.querySelectorAll(".katex-display")]) {
    const tex = texOf(display);
    if (tex.length > 0) {
      const block = doc.createElement("div");
      block.setAttribute("data-math-block", "");
      block.setAttribute("data-value", tex);
      display.replaceWith(block);
    } else {
      display.remove();
    }
  }
  for (const inline of [...body.querySelectorAll(".katex")]) {
    const tex = texOf(inline);
    if (tex.length > 0) {
      const span = doc.createElement("span");
      span.setAttribute("data-math-inline", "");
      span.setAttribute("data-value", tex);
      inline.replaceWith(span);
    } else {
      inline.remove();
    }
  }

  // Chat/shiki code blocks → a bare <pre>, shedding the header and copy
  // chrome that ride along in the copied fragment.
  for (const render of [...body.querySelectorAll(".pk-code-render")]) {
    const pre = render.querySelector("pre");
    const host = render.closest('[class*="codeblock"]') ?? render;
    if (pre) {
      host.replaceWith(pre);
    } else {
      host.remove();
    }
  }

  // GFM task checkboxes: the <input> becomes the li dialect the editor's
  // list schema reads, so pasted to-dos stay to-dos (state included).
  for (const input of [...body.querySelectorAll('input[type="checkbox"]')]) {
    const item = input.closest("li");
    if (item) {
      item.setAttribute("data-item-type", "task");
      item.setAttribute(
        "data-checked",
        (input as HTMLInputElement).checked || input.hasAttribute("checked") ? "true" : "false"
      );
    }
    input.remove();
  }

  // Chrome and head junk never carry note content.
  for (const el of [...body.querySelectorAll("button, style, script, meta, link, title")]) {
    el.remove();
  }

  return body.innerHTML;
};
