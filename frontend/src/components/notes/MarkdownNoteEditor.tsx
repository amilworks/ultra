/* Markdown mode — the doc-style editing surface over a plain-markdown note.
 *
 * The contract this component keeps (see the dual-mode design):
 * - body_markdown is the ONLY format. The ProseMirror document is a live
 *   view of it; serialization is remark-stringify pinned to the house
 *   dialect (`-` bullets, `---` rules) plus the ==highlight== handler.
 * - Opening a note never rewrites it: markdown leaves this component only
 *   through the listener's markdownUpdated, which fires on real doc changes.
 * - Typography rides the SAME pk-message-content voice chat answers use —
 *   the editable surface IS the rendered note (no separate preview).
 * - ultra://resource media render through node views resolved against the
 *   Resources download endpoint, exactly like the old preview did.
 *
 * The chunk is heavy (ProseMirror + remark), so NotesPage lazy-loads this
 * file only when a Markdown-mode note is open.
 */

import { useEffect, useMemo, useRef } from "react";

import {
  commandsCtx,
  defaultValueCtx,
  editorStateCtx,
  editorViewCtx,
  editorViewOptionsCtx,
  remarkStringifyOptionsCtx,
  rootCtx,
  Editor,
} from "@milkdown/kit/core";
import type { Ctx } from "@milkdown/kit/ctx";
import {
  commonmark,
  emphasisSchema,
  headingSchema,
  imageSchema,
  inlineCodeSchema,
  linkSchema,
  liftListItemCommand,
  sinkListItemCommand,
  strongSchema,
  toggleEmphasisCommand,
  toggleInlineCodeCommand,
  toggleLinkCommand,
  toggleStrongCommand,
  turnIntoTextCommand,
  updateLinkCommand,
  wrapInBlockquoteCommand,
  wrapInBulletListCommand,
  wrapInHeadingCommand,
  wrapInOrderedListCommand,
} from "@milkdown/kit/preset/commonmark";
import {
  addRowAfterCommand,
  gfm,
  goToNextTableCellCommand,
  goToPrevTableCellCommand,
  insertTableCommand,
  toggleStrikethroughCommand,
  strikethroughSchema,
} from "@milkdown/kit/preset/gfm";
import { clipboard } from "@milkdown/kit/plugin/clipboard";
import { history } from "@milkdown/kit/plugin/history";
import { listener, listenerCtx } from "@milkdown/kit/plugin/listener";
import type { MarkType } from "@milkdown/kit/prose/model";
import type { NodeViewConstructor } from "@milkdown/kit/prose/view";
import { $useKeymap, $view, callCommand, insert } from "@milkdown/kit/utils";
import { Milkdown, MilkdownProvider, useEditor } from "@milkdown/react";

import { parseUltraResourceRef, VIDEO_EXTENSION_PATTERN } from "@/lib/ultraResource";
import { withNotesDialect } from "@/components/notes/notesDialect";
import { notesMath } from "@/components/notes/notesMath";
import { sanitizePastedHtml } from "@/components/notes/notesPaste";
// KaTeX styles ride this lazy chunk — the same stylesheet the chat renderer
// loads with its math enhancement.
import "katex/dist/katex.min.css";
import {
  highlightSchema,
  toggleHighlightCommand,
  ultraHighlight,
} from "@/components/notes/notesHighlight";

export type NotesEditorAction =
  | "bold"
  | "italic"
  | "strike"
  | "code"
  | "highlight"
  | "quote"
  | "bullet"
  | "ordered"
  | "h2"
  | "h3"
  | "body"
  | "table";

export type NotesActiveStates = {
  bold: boolean;
  italic: boolean;
  strike: boolean;
  code: boolean;
  highlight: boolean;
  link: boolean;
  linkHref: string | null;
  block: "h2" | "h3" | "body" | "other";
};

export const IDLE_ACTIVE_STATES: NotesActiveStates = {
  bold: false,
  italic: false,
  strike: false,
  code: false,
  highlight: false,
  link: false,
  linkHref: null,
  block: "body",
};

export type NotesEditorHandle = {
  exec: (action: NotesEditorAction) => void;
  applyLink: (href: string) => void;
  removeLink: () => void;
  insertMarkdown: (markdown: string) => void;
  focus: () => void;
};

export type MarkdownNoteEditorProps = {
  defaultMarkdown: string;
  /** Resolves a Resources file id to its download URL (apiClient-bound). */
  resourceUrl: (fileId: string) => string;
  onMarkdownChange: (markdown: string) => void;
  onBlur: () => void;
  onActiveStatesChange: (states: NotesActiveStates) => void;
  /** FilePickerBridge pattern: the page captures the imperative surface. */
  bindApi: (api: NotesEditorHandle | null) => void;
};

/* GDocs-muscle-memory shortcuts on top of the preset defaults (⌘B/⌘I/⌘E and
   ⌘⌥1-6 ship with commonmark). Tab walks table cells and nests list items —
   command chaining returns false elsewhere so focus behaves normally. */
const notesKeymap = $useKeymap("ultraNotesKeymap", {
  StrikeLikeDocs: {
    shortcuts: "Mod-Shift-x",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () => commands.call(toggleStrikethroughCommand.key);
    },
  },
  OrderedLikeDocs: {
    shortcuts: "Mod-Shift-7",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () => commands.call(wrapInOrderedListCommand.key);
    },
  },
  BulletLikeDocs: {
    shortcuts: "Mod-Shift-8",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () => commands.call(wrapInBulletListCommand.key);
    },
  },
  QuoteLikeDocs: {
    shortcuts: "Mod-Shift-9",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () => commands.call(wrapInBlockquoteCommand.key);
    },
  },
  TableOrListTab: {
    shortcuts: "Tab",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () =>
        commands.call(goToNextTableCellCommand.key) ||
        // Docs/Word basic: Tab in the LAST cell grows the table by a row.
        // addRowAfter succeeds only inside a table, so prose is untouched.
        (commands.call(addRowAfterCommand.key) &&
          commands.call(goToNextTableCellCommand.key)) ||
        commands.call(sinkListItemCommand.key);
    },
  },
  TableOrListShiftTab: {
    shortcuts: "Shift-Tab",
    command: (ctx) => {
      const commands = ctx.get(commandsCtx);
      return () =>
        commands.call(goToPrevTableCellCommand.key) || commands.call(liftListItemCommand.key);
    },
  },
});

const markIsActive = (ctx: Ctx, type: MarkType | undefined): boolean => {
  if (!type) {
    return false;
  }
  const state = ctx.get(editorStateCtx);
  const { empty, from, to, $from } = state.selection;
  if (empty) {
    return Boolean(type.isInSet(state.storedMarks ?? $from.marks()));
  }
  return state.doc.rangeHasMark(from, to, type);
};

const readActiveStates = (ctx: Ctx): NotesActiveStates => {
  const state = ctx.get(editorStateCtx);
  const { $from } = state.selection;
  let block: NotesActiveStates["block"] = "body";
  for (let depth = $from.depth; depth >= 0; depth -= 1) {
    const node = $from.node(depth);
    if (node.type === headingSchema.type(ctx)) {
      const level = Number(node.attrs.level ?? 0);
      block = level === 2 ? "h2" : level === 3 ? "h3" : "other";
      break;
    }
  }
  const linkType = linkSchema.type(ctx);
  const linkActive = markIsActive(ctx, linkType);
  let linkHref: string | null = null;
  if (linkActive) {
    const mark = linkType.isInSet(state.storedMarks ?? $from.marks());
    linkHref = mark ? String(mark.attrs.href ?? "") : null;
  }
  return {
    bold: markIsActive(ctx, strongSchema.type(ctx)),
    italic: markIsActive(ctx, emphasisSchema.type(ctx)),
    strike: markIsActive(ctx, strikethroughSchema.type(ctx)),
    code: markIsActive(ctx, inlineCodeSchema.type(ctx)),
    highlight: markIsActive(ctx, highlightSchema.type(ctx)),
    link: linkActive,
    linkHref,
    block,
  };
};

function MarkdownNoteEditorCore({
  defaultMarkdown,
  resourceUrl,
  onMarkdownChange,
  onBlur,
  onActiveStatesChange,
  bindApi,
}: MarkdownNoteEditorProps) {
  /* The editor is created once per mount (the page keys this component by
     note id); callbacks flow through refs so the factory never goes stale. */
  const callbacksRef = useRef({ onMarkdownChange, onBlur, onActiveStatesChange, resourceUrl });
  callbacksRef.current = { onMarkdownChange, onBlur, onActiveStatesChange, resourceUrl };
  const initialMarkdownRef = useRef(defaultMarkdown);
  const activeFrameRef = useRef<number | null>(null);

  /* ultra://resource media render as real players/images inside the editing
     surface — same resolution the Resources browser uses, nothing stored. */
  const mediaView = useMemo(
    () =>
      $view(imageSchema.node, (): NodeViewConstructor => {
        return (node) => {
          const src = String(node.attrs.src ?? "");
          const alt = String(node.attrs.alt ?? "");
          const ref = parseUltraResourceRef(src);
          const url = ref ? callbacksRef.current.resourceUrl(ref.fileId) : src;
          const name = ref?.name || alt;
          const dom = document.createElement("span");
          dom.className = "notes-md-media";
          if (VIDEO_EXTENSION_PATTERN.test(name)) {
            const video = document.createElement("video");
            video.className = "notes-media-video";
            video.src = url;
            video.controls = true;
            video.preload = "metadata";
            video.setAttribute("aria-label", alt || name);
            dom.appendChild(video);
          } else {
            const img = document.createElement("img");
            img.className = "notes-media-img";
            img.src = url;
            img.alt = alt || name;
            img.loading = "lazy";
            dom.appendChild(img);
          }
          return {
            dom,
            selectNode: () => dom.classList.add("is-selected"),
            deselectNode: () => dom.classList.remove("is-selected"),
            ignoreMutation: () => true,
          };
        };
      }),
    []
  );

  const { get } = useEditor(
    (root) =>
      Editor.make()
        .config((ctx) => {
          ctx.set(rootCtx, root);
          ctx.set(defaultValueCtx, initialMarkdownRef.current);
          ctx.update(editorViewOptionsCtx, (options) => ({
            ...options,
            // Pasted chat answers and Word fragments get rewritten toward
            // plainness (KaTeX → math atoms, code chrome shed, checkbox
            // inputs → task items) before ProseMirror parses them.
            transformPastedHTML: sanitizePastedHtml,
            attributes: {
              // House reading voice on the editable surface itself — the
              // styled note IS the editor. pk-markdown carries the list and
              // measure rules (chat's tables ride React components instead,
              // so the notes table chrome lives in the notes CSS block).
              // Prose defaults stay writer-shaped: autocorrect on, sentence
              // caps (unlike the mono dump surface).
              class: "pk-message-content pk-markdown notes-md-prose",
              spellcheck: "true",
              autocapitalize: "sentences",
              autocorrect: "on",
            },
          }));
          /* House dialect (shared with the fidelity gate): serialization
             must not restyle untouched notes — see notesDialect.ts. */
          ctx.update(remarkStringifyOptionsCtx, withNotesDialect);
          ctx
            .get(listenerCtx)
            .markdownUpdated((_listenerCtx, markdown, prevMarkdown) => {
              if (markdown !== prevMarkdown) {
                callbacksRef.current.onMarkdownChange(markdown);
              }
            })
            .blur(() => {
              callbacksRef.current.onBlur();
            })
            .selectionUpdated((listenerInnerCtx) => {
              if (activeFrameRef.current !== null) {
                return;
              }
              activeFrameRef.current = window.requestAnimationFrame(() => {
                activeFrameRef.current = null;
                callbacksRef.current.onActiveStatesChange(readActiveStates(listenerInnerCtx));
              });
            });
        })
        .use(commonmark)
        .use(gfm)
        .use(listener)
        .use(history)
        .use(clipboard)
        .use(ultraHighlight)
        .use(notesMath)
        .use(notesKeymap)
        .use(mediaView),
    []
  );

  useEffect(() => {
    const api: NotesEditorHandle = {
      exec: (action) => {
        const editor = get();
        if (!editor) {
          return;
        }
        switch (action) {
          case "bold":
            editor.action(callCommand(toggleStrongCommand.key));
            break;
          case "italic":
            editor.action(callCommand(toggleEmphasisCommand.key));
            break;
          case "strike":
            editor.action(callCommand(toggleStrikethroughCommand.key));
            break;
          case "code":
            editor.action(callCommand(toggleInlineCodeCommand.key));
            break;
          case "highlight":
            editor.action(callCommand(toggleHighlightCommand.key));
            break;
          case "quote":
            editor.action(callCommand(wrapInBlockquoteCommand.key));
            break;
          case "bullet":
            editor.action(callCommand(wrapInBulletListCommand.key));
            break;
          case "ordered":
            editor.action(callCommand(wrapInOrderedListCommand.key));
            break;
          case "h2":
            editor.action(callCommand(wrapInHeadingCommand.key, 2));
            break;
          case "h3":
            editor.action(callCommand(wrapInHeadingCommand.key, 3));
            break;
          case "body":
            editor.action(callCommand(turnIntoTextCommand.key));
            break;
          case "table":
            editor.action(callCommand(insertTableCommand.key, { row: 3, col: 3 }));
            break;
        }
        api.focus();
      },
      applyLink: (href) => {
        const editor = get();
        if (!editor) {
          return;
        }
        const hasLink = editor.action((ctx) => markIsActive(ctx, linkSchema.type(ctx)));
        editor.action(
          callCommand(hasLink ? updateLinkCommand.key : toggleLinkCommand.key, { href })
        );
        api.focus();
      },
      removeLink: () => {
        const editor = get();
        if (!editor) {
          return;
        }
        editor.action(callCommand(toggleLinkCommand.key, {}));
        api.focus();
      },
      insertMarkdown: (markdown) => {
        get()?.action(insert(markdown));
      },
      focus: () => {
        get()?.action((ctx) => {
          ctx.get(editorViewCtx).focus();
        });
      },
    };
    bindApi(api);
    if (import.meta.env.DEV) {
      // Test hook for the stress harness: real-typing simulation must go
      // through handleTextInput (the path input rules live on), which
      // synthetic DOM events cannot reach. Dev builds only.
      (window as unknown as { __notesEditorView?: unknown }).__notesEditorView = () =>
        get()?.ctx.get(editorViewCtx);
    }
    return () => {
      bindApi(null);
      if (import.meta.env.DEV) {
        delete (window as unknown as { __notesEditorView?: unknown }).__notesEditorView;
      }
    };
  }, [bindApi, get]);

  useEffect(() => {
    return () => {
      if (activeFrameRef.current !== null) {
        window.cancelAnimationFrame(activeFrameRef.current);
      }
    };
  }, []);

  return <Milkdown />;
}

export function MarkdownNoteEditor(props: MarkdownNoteEditorProps) {
  return (
    <MilkdownProvider>
      <MarkdownNoteEditorCore {...props} />
    </MilkdownProvider>
  );
}
