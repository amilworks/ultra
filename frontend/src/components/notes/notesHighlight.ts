/* The ==highlight== mark inside the Markdown-mode editor.
 *
 * Mirrors preset-gfm's strikethrough construction exactly: a mark schema
 * bridging the `highlight` mdast node (produced by the shared remarkHighlight
 * plugin, serialized by highlightToMarkdown), a toggle command, an as-you-type
 * input rule, and the ⌘⇧H keymap. The editor and the chat renderer read the
 * SAME remark plugin, so a highlight means one thing everywhere.
 */

import { commandsCtx } from "@milkdown/kit/core";
import { markRule } from "@milkdown/kit/prose";
import { toggleMark } from "@milkdown/kit/prose/commands";
import {
  $command,
  $inputRule,
  $markAttr,
  $markSchema,
  $remark,
  $useKeymap,
} from "@milkdown/kit/utils";

import { remarkHighlight } from "@/lib/remarkHighlight";

export const highlightAttr = $markAttr("highlight");

export const highlightSchema = $markSchema("highlight", (ctx) => ({
  parseDOM: [{ tag: "mark" }],
  toDOM: (mark) => ["mark", ctx.get(highlightAttr.key)(mark)],
  parseMarkdown: {
    match: (node) => node.type === "highlight",
    runner: (state, node, markType) => {
      state.openMark(markType);
      state.next(node.children);
      state.closeMark(markType);
    },
  },
  toMarkdown: {
    match: (mark) => mark.type.name === "highlight",
    runner: (state, mark) => {
      state.withMark(mark, "highlight");
    },
  },
}));

const remarkHighlightPlugin = $remark("ultraHighlight", () => remarkHighlight);

export const toggleHighlightCommand = $command(
  "ToggleUltraHighlight",
  (ctx) => () => toggleMark(highlightSchema.type(ctx))
);

/* Typing ==text== highlights as you go — the editor twin of the renderer's
   flanking rules (input rules end at the caret, hence the $ anchor). */
const highlightInputRule = $inputRule((ctx) =>
  markRule(/(?<![=\w])==(?![\s=])((?:[^=\n]|=(?!=))*?[^\s=])==$/, highlightSchema.type(ctx))
);

const highlightKeymap = $useKeymap("ultraHighlightKeymap", {
  ToggleHighlight: {
    shortcuts: "Mod-Shift-h",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () => commands.call(toggleHighlightCommand.key);
    },
  },
});

export const ultraHighlight = [
  highlightAttr,
  highlightSchema,
  remarkHighlightPlugin,
  toggleHighlightCommand,
  highlightInputRule,
  highlightKeymap,
].flat();
