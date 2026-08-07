/* LaTeX in Markdown mode — $inline$ and $$display$$ math.
 *
 * Storage is exactly the chat pipeline's dialect (remark-math: `$…$` /
 * `$$…$$`), so a formula written in a note renders identically in chat and
 * stays plain text for the agent and the Plaintext surface. remark-math
 * registers its own toMarkdown extension, so round-trips need no custom
 * stringify handler.
 *
 * Built in-house on the same composition pattern as notesHighlight —
 * @milkdown/plugin-math is deprecated and version-frozen at 7.5.x, while
 * remark-math and katex are already house dependencies of the chat renderer.
 *
 * Editing model (Google-Docs-simple): formulas render as KaTeX atoms; click
 * one (or type `$…$` and land on it) and it flips to a small raw-TeX editor
 * in place — Enter/blur commits and re-renders, Escape cancels, and an
 * emptied formula deletes itself. Display math edits in a textarea (Enter
 * makes newlines for aligned environments; ⌘Enter or blur commits).
 */

import katex from "katex";
import remarkMath from "remark-math";

import type { Ctx } from "@milkdown/kit/ctx";
import { InputRule } from "@milkdown/kit/prose/inputrules";
import type { Node as ProseNode } from "@milkdown/kit/prose/model";
import type { EditorView, NodeViewConstructor } from "@milkdown/kit/prose/view";
import { $inputRule, $nodeSchema, $remark, $view } from "@milkdown/kit/utils";

const remarkMathPlugin = $remark("notesMath", () => remarkMath);

export const mathInlineSchema = $nodeSchema("math_inline", () => ({
  group: "inline",
  inline: true,
  atom: true,
  attrs: { value: { default: "" } },
  parseDOM: [
    {
      tag: "span[data-math-inline]",
      getAttrs: (dom) => ({ value: (dom as HTMLElement).dataset.value ?? "" }),
    },
  ],
  toDOM: (node) => [
    "span",
    { "data-math-inline": "", "data-value": String(node.attrs.value) },
    String(node.attrs.value),
  ],
  parseMarkdown: {
    match: (node) => node.type === "inlineMath",
    runner: (state, node, type) => {
      state.addNode(type, { value: String(node.value ?? "") });
    },
  },
  toMarkdown: {
    match: (node) => node.type.name === "math_inline",
    runner: (state, node) => {
      state.addNode("inlineMath", undefined, String(node.attrs.value));
    },
  },
}));

export const mathBlockSchema = $nodeSchema("math_block", () => ({
  group: "block",
  atom: true,
  attrs: { value: { default: "" } },
  parseDOM: [
    {
      tag: "div[data-math-block]",
      getAttrs: (dom) => ({ value: (dom as HTMLElement).dataset.value ?? "" }),
    },
  ],
  toDOM: (node) => [
    "div",
    { "data-math-block": "", "data-value": String(node.attrs.value) },
    String(node.attrs.value),
  ],
  parseMarkdown: {
    match: (node) => node.type === "math",
    runner: (state, node, type) => {
      state.addNode(type, { value: String(node.value ?? "") });
    },
  },
  toMarkdown: {
    match: (node) => node.type.name === "math_block",
    runner: (state, node) => {
      state.addNode("math", undefined, String(node.attrs.value));
    },
  },
}));

/* Typing $…$ formats as you go, mirroring remark-math's flanking (the TeX
   must not start or end with whitespace, and a lone $ stays a dollar sign). */
const mathInlineInputRule = $inputRule(
  (ctx) =>
    new InputRule(/(?<!\$)\$([^$\n]+)\$$/, (state, match, start, end) => {
      const value = (match[1] ?? "").trim();
      if (value.length === 0 || /^\s|\s$/.test(match[1] ?? "")) {
        return null;
      }
      return state.tr.replaceRangeWith(
        start,
        end,
        mathInlineSchema.type(ctx).create({ value })
      );
    })
);

/* An empty line holding exactly $$ becomes a display-math block; its editor
   opens on click. */
const mathBlockInputRule = $inputRule(
  (ctx) =>
    new InputRule(/^\$\$\s$/, (state, _match, start, end) => {
      const $start = state.doc.resolve(start);
      if ($start.parent.type.name !== "paragraph" || $start.parent.textContent.trim() !== "$$") {
        return null;
      }
      return state.tr.replaceRangeWith(
        Math.max(0, start - 1),
        end,
        mathBlockSchema.type(ctx).create({ value: "" })
      );
    })
);

const renderKatex = (target: HTMLElement, value: string, displayMode: boolean): void => {
  try {
    katex.render(value.length > 0 ? value : "\\;", target, {
      throwOnError: false,
      displayMode,
    });
  } catch {
    target.textContent = value;
  }
};

type MathEditorField = HTMLInputElement | HTMLTextAreaElement;

/* Shared click-to-edit machinery for both math atoms. */
const createMathView = (displayMode: boolean): NodeViewConstructor => {
  return (node: ProseNode, view: EditorView, getPos: () => number | undefined) => {
    let current = node;
    let editing = false;

    const dom = document.createElement(displayMode ? "div" : "span");
    dom.className = displayMode ? "notes-math-block" : "notes-math-inline";
    dom.setAttribute("title", "Click to edit LaTeX");

    const render = () => {
      if (!editing) {
        renderKatex(dom, String(current.attrs.value), displayMode);
      }
    };

    const commit = (field: MathEditorField, cancel: boolean) => {
      if (!editing) {
        return;
      }
      editing = false;
      const pos = getPos();
      if (cancel || pos == null) {
        render();
        return;
      }
      const value = field.value.trim();
      const tr = view.state.tr;
      if (value.length === 0) {
        // An emptied formula deletes itself — no invisible husk left behind.
        view.dispatch(tr.delete(pos, pos + current.nodeSize));
        view.focus();
        return;
      }
      if (value !== String(current.attrs.value)) {
        view.dispatch(tr.setNodeMarkup(pos, undefined, { value }));
      }
      render();
      view.focus();
    };

    const startEditing = () => {
      if (editing) {
        return;
      }
      editing = true;
      dom.textContent = "";
      const field: MathEditorField = displayMode
        ? document.createElement("textarea")
        : document.createElement("input");
      field.className = displayMode ? "notes-math-editor notes-math-editor-block" : "notes-math-editor";
      field.value = String(current.attrs.value);
      if (displayMode) {
        (field as HTMLTextAreaElement).rows = Math.max(2, field.value.split("\n").length + 1);
      }
      field.setAttribute("aria-label", "LaTeX source");
      field.spellcheck = false;
      dom.appendChild(field);
      field.focus();
      field.select();
      field.addEventListener("keydown", (event) => {
        const keyEvent = event as KeyboardEvent;
        keyEvent.stopPropagation();
        const commitKey = displayMode
          ? keyEvent.key === "Enter" && (keyEvent.metaKey || keyEvent.ctrlKey)
          : keyEvent.key === "Enter";
        if (commitKey) {
          keyEvent.preventDefault();
          commit(field, false);
        } else if (keyEvent.key === "Escape") {
          keyEvent.preventDefault();
          commit(field, true);
        }
      });
      field.addEventListener("blur", () => commit(field, false));
    };

    dom.addEventListener("mousedown", (event) => {
      if (!editing) {
        event.preventDefault();
        startEditing();
      }
    });

    render();

    return {
      dom,
      update: (next: ProseNode) => {
        if (next.type !== current.type) {
          return false;
        }
        current = next;
        render();
        return true;
      },
      selectNode: () => {
        dom.classList.add("is-selected");
      },
      deselectNode: () => {
        dom.classList.remove("is-selected");
      },
      stopEvent: (event) => editing && dom.contains(event.target as Node | null),
      ignoreMutation: () => true,
    };
  };
};

const mathInlineView = $view(mathInlineSchema.node, (): NodeViewConstructor => createMathView(false));
const mathBlockView = $view(mathBlockSchema.node, (): NodeViewConstructor => createMathView(true));

/* Insert an empty display-math block at the selection (ribbon/slash path). */
export const insertMathBlock = (ctx: Ctx, view: EditorView): void => {
  const node = mathBlockSchema.type(ctx).create({ value: "" });
  view.dispatch(view.state.tr.replaceSelectionWith(node));
};

export const notesMath = [
  remarkMathPlugin,
  mathInlineSchema,
  mathBlockSchema,
  mathInlineInputRule,
  mathBlockInputRule,
  mathInlineView,
  mathBlockView,
].flat();
