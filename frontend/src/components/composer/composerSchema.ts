import { Fragment, Schema, Slice, type Node as PMNode } from "prosemirror-model";
import { EditorState, Selection, TextSelection, type Transaction } from "prosemirror-state";

import {
  BRIEF_MENTION_QUERY_MAX_LENGTH,
  briefTokenText,
  parseBriefSegments,
} from "@/features/chat/brief-tokens";

import type { ComposerFileToken } from "./composerHandle";

/* The composer's document: paragraphs of text with inline, atomic file tokens.
   A token is one unit to the caret and to Backspace, and serialises to the
   `@label` run the grammar module already understands — so the text the run
   receives is the same whether it was typed here, pasted, or restored. */

export const composerSchema = new Schema({
  nodes: {
    doc: { content: "paragraph+" },
    paragraph: {
      content: "inline*",
      group: "block",
      parseDOM: [{ tag: "p" }, { tag: "div" }],
      toDOM: () => ["p", 0],
    },
    text: { group: "inline" },
    fileToken: {
      group: "inline",
      inline: true,
      atom: true,
      selectable: true,
      draggable: false,
      attrs: { label: {}, fileId: {} },
      leafText: (node) => briefTokenText(String(node.attrs.label)),
      parseDOM: [
        {
          tag: "span[data-file-id]",
          getAttrs: (dom) => ({
            fileId: (dom as HTMLElement).getAttribute("data-file-id") ?? "",
            label: (dom as HTMLElement).getAttribute("data-label") ?? "",
          }),
        },
      ],
      toDOM: (node) => [
        "span",
        {
          class: "composer-token",
          "data-file-id": String(node.attrs.fileId),
          "data-label": String(node.attrs.label),
        },
        briefTokenText(String(node.attrs.label)),
      ],
    },
  },
});

const TOKEN_PLACEHOLDER = "￼";

const paragraphFromLine = (line: string, registry: readonly ComposerFileToken[]): PMNode => {
  const inline: PMNode[] = [];
  for (const segment of parseBriefSegments(line, registry)) {
    if (segment.kind === "file") {
      inline.push(
        composerSchema.nodes.fileToken.create({ label: segment.label, fileId: segment.fileId })
      );
    } else if (segment.text.length > 0) {
      inline.push(composerSchema.text(segment.text));
    }
  }
  return composerSchema.nodes.paragraph.create(null, inline);
};

export const docFromText = (text: string, registry: readonly ComposerFileToken[]): PMNode => {
  const lines = text.split("\n");
  return composerSchema.nodes.doc.create(
    null,
    lines.map((line) => paragraphFromLine(line, registry))
  );
};

const paragraphText = (paragraph: PMNode): string =>
  paragraph.textBetween(0, paragraph.content.size, undefined, (leaf) =>
    leaf.type.name === "fileToken" ? briefTokenText(String(leaf.attrs.label)) : ""
  );

export const textFromDoc = (doc: PMNode): string => {
  const lines: string[] = [];
  doc.forEach((paragraph) => {
    lines.push(paragraphText(paragraph));
  });
  return lines.join("\n");
};

export const isDocEmpty = (doc: PMNode): boolean =>
  doc.childCount === 1 && doc.firstChild !== null && doc.firstChild.content.size === 0;

export const tokensInDoc = (doc: PMNode): ComposerFileToken[] => {
  const seen = new Set<string>();
  const tokens: ComposerFileToken[] = [];
  doc.descendants((node) => {
    if (node.type.name === "fileToken") {
      const fileId = String(node.attrs.fileId);
      if (!seen.has(fileId)) {
        seen.add(fileId);
        tokens.push({ label: String(node.attrs.label), fileId });
      }
    }
    return true;
  });
  return tokens;
};

export const sameTokens = (
  left: readonly ComposerFileToken[],
  right: readonly ComposerFileToken[]
): boolean =>
  left.length === right.length &&
  left.every((token, index) => token.fileId === right[index].fileId && token.label === right[index].label);

export type ComposerMentionRange = { from: number; to: number; query: string };

const MENTION_PATTERN = new RegExp(
  `(?:^|[\\s(\\[{"'“‘])@([^\\s@${TOKEN_PLACEHOLDER}]{0,${BRIEF_MENTION_QUERY_MAX_LENGTH}})$`
);

/** The `@query` run ending at a collapsed caret, if the caret is inside one. */
export const mentionAtSelection = (state: EditorState): ComposerMentionRange | null => {
  const { selection } = state;
  if (!selection.empty) {
    return null;
  }
  const $from = selection.$from;
  if (!$from.parent.isTextblock) {
    return null;
  }
  const before = state.doc.textBetween($from.start(), $from.pos, undefined, () => TOKEN_PLACEHOLDER);
  const match = MENTION_PATTERN.exec(before);
  if (!match) {
    return null;
  }
  const query = match[1];
  return { from: $from.pos - query.length - 1, to: $from.pos, query };
};

const charAt = (doc: PMNode, pos: number): string => {
  if (pos < 0 || pos >= doc.content.size) {
    return "";
  }
  const $pos = doc.resolve(pos);
  if (!$pos.parent.isTextblock || pos >= $pos.end()) {
    return "";
  }
  return doc.textBetween(pos, pos + 1, undefined, () => TOKEN_PLACEHOLDER);
};

const charBefore = (doc: PMNode, pos: number): string => {
  const $pos = doc.resolve(pos);
  if (!$pos.parent.isTextblock || pos <= $pos.start()) {
    return "";
  }
  return doc.textBetween(pos - 1, pos, undefined, () => TOKEN_PLACEHOLDER);
};

const isSpace = (value: string): boolean => value === " " || value === " ";

/** Replace [from, to) with a token, padding one space after it (and before it
    when it lands against a word), and leave the caret after the padding. */
export const insertTokenAt = (
  state: EditorState,
  from: number,
  to: number,
  token: ComposerFileToken
): Transaction => {
  const node = composerSchema.nodes.fileToken.create({ label: token.label, fileId: token.fileId });
  let tr = state.tr;
  const previous = charBefore(state.doc, from);
  const needsLeadingSpace = previous !== "" && !isSpace(previous);
  tr = tr.replaceWith(from, to, needsLeadingSpace ? [composerSchema.text(" "), node] : [node]);
  let caret = from + (needsLeadingSpace ? 1 : 0) + node.nodeSize;
  const next = charAt(tr.doc, caret);
  if (!isSpace(next)) {
    tr = tr.insertText(" ", caret);
  }
  caret += 1;
  return tr.setSelection(TextSelection.create(tr.doc, caret));
};

/** A file arriving without a mention: at the caret when the editor is focused,
    else at the end of the text. */
export const appendTokenAt = (
  state: EditorState,
  token: ComposerFileToken,
  atCaret: boolean
): Transaction => {
  const position = atCaret ? state.selection.to : Selection.atEnd(state.doc).from;
  return insertTokenAt(state, position, position, token);
};

export const findTokenPosition = (doc: PMNode, fileId: string): number | null => {
  let found: number | null = null;
  doc.descendants((node, pos) => {
    if (found !== null) {
      return false;
    }
    if (node.type.name === "fileToken" && String(node.attrs.fileId) === fileId) {
      found = pos;
      return false;
    }
    return true;
  });
  return found;
};

/** Delete a token and one padding space (the following one when there is one). */
export const removeTokenNode = (state: EditorState, fileId: string): Transaction | null => {
  const pos = findTokenPosition(state.doc, fileId);
  if (pos === null) {
    return null;
  }
  let from = pos;
  let to = pos + 1;
  if (isSpace(charAt(state.doc, to))) {
    to += 1;
  } else if (isSpace(charBefore(state.doc, from))) {
    from -= 1;
  }
  // A token that closed the line takes its leading space with it too, so the
  // text never ends in a stray space.
  if (charAt(state.doc, to) === "" && isSpace(charBefore(state.doc, from))) {
    from -= 1;
  }
  const tr = state.tr.delete(from, to);
  return tr.setSelection(TextSelection.create(tr.doc, Math.min(from, tr.doc.content.size - 1)));
};

/** Swap a token for a bare "@" with the caret after it, so a mention is active again. */
export const reopenMentionAt = (state: EditorState, fileId: string): Transaction | null => {
  const pos = findTokenPosition(state.doc, fileId);
  if (pos === null) {
    return null;
  }
  const tr = state.tr.replaceWith(pos, pos + 1, composerSchema.text("@"));
  return tr.setSelection(TextSelection.create(tr.doc, pos + 1));
};

/** Backspace with the caret right after a token removes the token whole — one
    keystroke, no intermediate node selection. Anywhere else it is ordinary. */
export const deleteTokenBackward = (state: EditorState): Transaction | null => {
  const { selection } = state;
  if (!selection.empty) {
    return null;
  }
  const before = selection.$from.nodeBefore;
  if (!before || before.type.name !== "fileToken") {
    return null;
  }
  const from = selection.from - before.nodeSize;
  const tr = state.tr.delete(from, selection.from);
  return tr.setSelection(TextSelection.create(tr.doc, from));
};

/** Delete with the caret right before a token removes the token whole. */
export const deleteTokenForward = (state: EditorState): Transaction | null => {
  const { selection } = state;
  if (!selection.empty) {
    return null;
  }
  const after = selection.$from.nodeAfter;
  if (!after || after.type.name !== "fileToken") {
    return null;
  }
  return state.tr.delete(selection.from, selection.from + after.nodeSize);
};

/** Insert text at the selection; newlines become paragraphs. */
export const insertMultilineText = (tr: Transaction, text: string): Transaction => {
  const lines = text.split("\n");
  if (lines.length === 1) {
    return text.length > 0 ? tr.insertText(text) : tr;
  }
  const paragraphs = lines.map((line) =>
    composerSchema.nodes.paragraph.create(null, line.length > 0 ? [composerSchema.text(line)] : [])
  );
  return tr.replaceSelection(new Slice(Fragment.from(paragraphs), 1, 1)).scrollIntoView();
};

/** Replace the document from text, keeping the caret near where it was. */
export const replaceDocFromText = (
  state: EditorState,
  text: string,
  registry: readonly ComposerFileToken[],
  caret: "keep" | "end" = "keep"
): Transaction => {
  const next = docFromText(text, registry);
  const previousFrom = state.selection.from;
  const tr = state.tr.replaceWith(0, state.doc.content.size, next.content);
  const size = tr.doc.content.size;
  const target = caret === "end" ? Selection.atEnd(tr.doc) : Selection.near(tr.doc.resolve(Math.max(0, Math.min(previousFrom, size - 1))));
  return tr.setSelection(target);
};

export const emptyEditorState = (
  text: string,
  registry: readonly ComposerFileToken[],
  plugins: EditorState["plugins"]
): EditorState =>
  EditorState.create({ doc: docFromText(text, registry), plugins, schema: composerSchema });
