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
  addColAfterCommand,
  addRowAfterCommand,
  gfm,
  goToNextTableCellCommand,
  goToPrevTableCellCommand,
  insertTableCommand,
  toggleStrikethroughCommand,
  strikethroughSchema,
} from "@milkdown/kit/preset/gfm";
import { clipboard } from "@milkdown/kit/plugin/clipboard";
import { cursor } from "@milkdown/kit/plugin/cursor";
import { history } from "@milkdown/kit/plugin/history";
import { listener, listenerCtx } from "@milkdown/kit/plugin/listener";
import { trailing } from "@milkdown/kit/plugin/trailing";
import { exitCode } from "@milkdown/kit/prose/commands";
import type { MarkType, Node as ProseNode } from "@milkdown/kit/prose/model";
import { type EditorState, Plugin, TextSelection } from "@milkdown/kit/prose/state";
import { deleteColumn, deleteRow, deleteTable } from "@milkdown/kit/prose/tables";
import {
  Decoration,
  DecorationSet,
  type NodeViewConstructor,
} from "@milkdown/kit/prose/view";
import { $prose, $useKeymap, $view, callCommand, insert } from "@milkdown/kit/utils";
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
  | "table"
  | "rowBelow"
  | "rowDelete"
  | "colRight"
  | "colDelete"
  | "tableDelete";

export type NotesActiveStates = {
  bold: boolean;
  italic: boolean;
  strike: boolean;
  code: boolean;
  highlight: boolean;
  link: boolean;
  linkHref: string | null;
  block: "h2" | "h3" | "body" | "other";
  /** Caret inside a table — the page reveals contextual row/column controls. */
  inTable: boolean;
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
  inTable: false,
};

export type NotesEditorHandle = {
  exec: (action: NotesEditorAction) => void;
  applyLink: (href: string) => void;
  removeLink: () => void;
  insertMarkdown: (markdown: string) => void;
  focus: () => void;
};

export type NotesEditorAnchor = {
  left: number;
  top: number;
  bottom: number;
};

export type NotesEditorMenuRequest = {
  kind: "blocks" | "resources";
  anchor: NotesEditorAnchor;
};

export type MarkdownNoteEditorProps = {
  defaultMarkdown: string;
  /** Resolves a Resources file id to its download URL (apiClient-bound). */
  resourceUrl: (fileId: string) => string;
  onMarkdownChange: (markdown: string) => void;
  onBlur: (event: FocusEvent) => void;
  onActiveStatesChange: (states: NotesActiveStates) => void;
  onSelectionAnchorChange: (anchor: NotesEditorAnchor | null) => void;
  onCaretAnchorChange: (anchor: NotesEditorAnchor | null) => void;
  onMenuRequest: (request: NotesEditorMenuRequest) => void;
  /** Lets the page own keyboard navigation while a contextual menu is open. */
  onMenuKeyDown: (event: KeyboardEvent) => boolean;
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
  /* ⌘⏎ steps out of a code block (or any block that eats Enter) into a fresh
     paragraph below — the deliberate exit; ArrowDown works too because the
     trailing plugin keeps a paragraph after every last block. */
  ExitCodeBlock: {
    shortcuts: "Mod-Enter",
    command: (ctx) => {
      return () => {
        const view = ctx.get(editorViewCtx);
        return exitCode(view.state, view.dispatch);
      };
    },
  },
});

const markIsActive = (state: EditorState, type: MarkType | undefined): boolean => {
  if (!type) {
    return false;
  }
  const { empty, from, to, $from } = state.selection;
  if (empty) {
    return Boolean(type.isInSet(state.storedMarks ?? $from.marks()));
  }
  return state.doc.rangeHasMark(from, to, type);
};

const readActiveStates = (ctx: Ctx, state: EditorState): NotesActiveStates => {
  const { $from } = state.selection;
  let block: NotesActiveStates["block"] = "body";
  let inTable = false;
  for (let depth = $from.depth; depth >= 0; depth -= 1) {
    const node = $from.node(depth);
    if (node.type.name === "table") {
      inTable = true;
    }
    if (block === "body" && node.type === headingSchema.type(ctx)) {
      const level = Number(node.attrs.level ?? 0);
      block = level === 2 ? "h2" : level === 3 ? "h3" : "other";
    }
  }
  const linkType = linkSchema.type(ctx);
  const linkActive = markIsActive(state, linkType);
  let linkHref: string | null = null;
  if (linkActive) {
    if (state.selection.empty) {
      const mark = linkType.isInSet(state.storedMarks ?? $from.marks());
      linkHref = mark ? String(mark.attrs.href ?? "") : null;
    } else {
      // At a link's left boundary, $from.marks() can describe the preceding
      // plain text. Read the selected content that made the link active.
      state.doc.nodesBetween(state.selection.from, state.selection.to, (node) => {
        const mark = linkType.isInSet(node.marks);
        if (linkHref === null && mark) {
          linkHref = String(mark.attrs.href ?? "");
        }
      });
    }
  }
  return {
    bold: markIsActive(state, strongSchema.type(ctx)),
    italic: markIsActive(state, emphasisSchema.type(ctx)),
    strike: markIsActive(state, strikethroughSchema.type(ctx)),
    code: markIsActive(state, inlineCodeSchema.type(ctx)),
    highlight: markIsActive(state, highlightSchema.type(ctx)),
    link: linkActive,
    linkHref,
    block,
    inTable,
  };
};

/* Clicking a task checkbox toggles it. The box renders in the list gutter —
   OUTSIDE the item's content box — so the click arrives as a position, not
   a node: handleClick (not handleClickOn) is the hook that always fires.
   Any click left of the item's content edge is the checkbox, unambiguously.
   Serialization is just [x] ⇄ [ ]. */
const taskTogglePlugin = $prose(
  () =>
    new Plugin({
      props: {
        decorations: (state) => {
          const decorations: Decoration[] = [];
          state.doc.descendants((node, pos) => {
            if (node.type.name !== "list_item" || node.attrs.checked == null) {
              return;
            }
            decorations.push(
              Decoration.node(pos, pos + node.nodeSize, {
                role: "checkbox",
                "aria-checked": node.attrs.checked ? "true" : "false",
                "aria-label": node.textContent.trim() || "Task",
                tabindex: "0",
              })
            );
          });
          return DecorationSet.create(state.doc, decorations);
        },
        handleKeyDown: (view, event) => {
          if (event.key !== " " && event.key !== "Enter") return false;
          const target = event.target;
          if (!(target instanceof HTMLElement)) return false;
          const taskItem = target.closest<HTMLElement>(
            'li[data-item-type="task"][role="checkbox"]'
          );
          if (!taskItem || !view.dom.contains(taskItem)) return false;
          let insidePos: number;
          try {
            insidePos = view.posAtDOM(taskItem, 0);
          } catch {
            return false;
          }
          // A DOM position at offset zero is normally the first position
          // *inside* the list item. Resolve the enclosing node boundary; using
          // the inside position directly points at its paragraph child.
          const itemPos = [insidePos - 1, insidePos].find((candidate) => {
            if (candidate < 0 || candidate >= view.state.doc.content.size) {
              return false;
            }
            const candidateNode = view.state.doc.nodeAt(candidate);
            return candidateNode?.type.name === "list_item" && candidateNode.attrs.checked != null;
          });
          if (itemPos == null) {
            return false;
          }
          const item = view.state.doc.nodeAt(itemPos);
          if (item?.type.name !== "list_item" || item.attrs.checked == null) {
            return false;
          }
          event.preventDefault();
          view.dispatch(
            view.state.tr.setNodeMarkup(itemPos, undefined, {
              ...item.attrs,
              checked: !item.attrs.checked,
            })
          );
          return true;
        },
        handleClick: (view, pos, event) => {
          const $pos = view.state.doc.resolve(pos);
          let item: ProseNode | null = null;
          let itemPos: number | null = null;
          // Clicks inside the item's text resolve deep — walk the ancestors.
          for (let depth = $pos.depth; depth > 0; depth -= 1) {
            const node = $pos.node(depth);
            if (node.type.name === "list_item" && node.attrs.checked != null) {
              item = node;
              itemPos = $pos.before(depth);
              break;
            }
          }
          // Gutter clicks (where the box actually lives) resolve BETWEEN
          // items, at bullet_list depth — the clicked item is nodeAfter.
          if (!item && $pos.nodeAfter?.type.name === "list_item" && $pos.nodeAfter.attrs.checked != null) {
            item = $pos.nodeAfter;
            itemPos = pos;
          }
          if (!item || itemPos == null) {
            return false;
          }
          const dom = view.nodeDOM(itemPos);
          if (
            !(dom instanceof HTMLElement) ||
            event.clientX > dom.getBoundingClientRect().left - 2
          ) {
            return false;
          }
          view.dispatch(
            view.state.tr.setNodeMarkup(itemPos, undefined, {
              ...item.attrs,
              checked: !item.attrs.checked,
            })
          );
          return true;
        },
      },
    })
);

function MarkdownNoteEditorCore({
  defaultMarkdown,
  resourceUrl,
  onMarkdownChange,
  onBlur,
  onActiveStatesChange,
  onSelectionAnchorChange,
  onCaretAnchorChange,
  onMenuRequest,
  onMenuKeyDown,
  bindApi,
}: MarkdownNoteEditorProps) {
  /* The editor is created once per mount (the page keys this component by
     note id); callbacks flow through refs so the factory never goes stale. */
  const callbacksRef = useRef({
    onMarkdownChange,
    onBlur,
    onActiveStatesChange,
    onSelectionAnchorChange,
    onCaretAnchorChange,
    onMenuRequest,
    onMenuKeyDown,
    resourceUrl,
  });
  useEffect(() => {
    callbacksRef.current = {
      onMarkdownChange,
      onBlur,
      onActiveStatesChange,
      onSelectionAnchorChange,
      onCaretAnchorChange,
      onMenuRequest,
      onMenuKeyDown,
      resourceUrl,
    };
  }, [
    onActiveStatesChange,
    onBlur,
    onCaretAnchorChange,
    onMarkdownChange,
    onMenuKeyDown,
    onMenuRequest,
    onSelectionAnchorChange,
    resourceUrl,
  ]);
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

  const { get, loading } = useEditor(
    (root) => {
      const activeStatesPlugin = $prose(
        (ctx) => new Plugin({
          view: (view) => {
            const publish = (state: EditorState) => {
              const states = readActiveStates(ctx, state);
              if (import.meta.env.DEV) {
                (window as unknown as { __notesLastStates?: unknown }).__notesLastStates = states;
              }
              callbacksRef.current.onActiveStatesChange(states);
            };
            publish(view.state);
            return {
              update: (nextView, previousState) => {
                const state = nextView.state;
                if (
                  state.doc !== previousState.doc ||
                  !state.selection.eq(previousState.selection) ||
                  state.storedMarks !== previousState.storedMarks
                ) {
                  // Marks can change without moving the selection. Publish from
                  // the committed view before a following command opens its UI.
                  publish(state);
                }
              },
            };
          },
        })
      );

      return Editor.make()
        .config((ctx) => {
          ctx.set(rootCtx, root);
          ctx.set(defaultValueCtx, initialMarkdownRef.current);
          ctx.update(editorViewOptionsCtx, (options) => {
            const previousHandleKeyDown = options.handleKeyDown;
            return {
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
                "aria-label": "Note body",
              },
              handleDOMEvents: {
                ...options.handleDOMEvents,
                blur: (view, event) => {
                  // A queued selection update must not reopen tools after
                  // focus has moved to the toolbar or another field.
                  if (activeFrameRef.current !== null) {
                    window.clearTimeout(activeFrameRef.current);
                    activeFrameRef.current = null;
                  }
                  callbacksRef.current.onCaretAnchorChange(null);
                  callbacksRef.current.onBlur(event);
                  return options.handleDOMEvents?.blur?.(view, event) ?? false;
                },
              },
              handleKeyDown: (view, event) => {
                if (callbacksRef.current.onMenuKeyDown(event)) {
                  event.preventDefault();
                  return true;
                }
                if (
                  (event.key === "/" || event.key === "@") &&
                  !event.metaKey &&
                  !event.ctrlKey &&
                  !event.altKey &&
                  view.state.selection.empty &&
                  view.state.selection.$from.parentOffset === 0
                ) {
                  const coords = view.coordsAtPos(view.state.selection.from);
                  event.preventDefault();
                  callbacksRef.current.onMenuRequest({
                    kind: event.key === "/" ? "blocks" : "resources",
                    anchor: {
                      left: coords.left,
                      top: coords.top,
                      bottom: coords.bottom,
                    },
                  });
                  return true;
                }
                return previousHandleKeyDown?.(view, event) ?? false;
              },
            };
          });
          /* House dialect (shared with the fidelity gate): serialization
             must not restyle untouched notes — see notesDialect.ts. */
          ctx.update(remarkStringifyOptionsCtx, withNotesDialect);
          ctx
            .get(listenerCtx)
            .markdownUpdated((_listenerCtx, markdown, prevMarkdown) => {
              // The trailing plugin appends an empty paragraph after terminal
              // tables/lists/code so the caret can escape. Its one-newline
              // hydration delta is editor structure, not a durable note edit.
              if (markdown !== prevMarkdown && markdown !== `${prevMarkdown}\n`) {
                callbacksRef.current.onMarkdownChange(markdown);
              }
            })
            .selectionUpdated((listenerInnerCtx) => {
              if (activeFrameRef.current !== null) {
                return;
              }
              // A short timeout, not requestAnimationFrame: rAF freezes in
              // hidden tabs, which would stall contextual geometry until the next
              // visible frame. A timer coalesces bursts just as well.
              activeFrameRef.current = window.setTimeout(() => {
                activeFrameRef.current = null;
                const view = listenerInnerCtx.get(editorViewCtx);
                if (!view.hasFocus()) {
                  return;
                }
                const selection = view.state.selection;
                const caretCoords = view.coordsAtPos(selection.from);
                const caretAnchor: NotesEditorAnchor = {
                  left: caretCoords.left,
                  top: caretCoords.top,
                  bottom: caretCoords.bottom,
                };
                let anchor: NotesEditorAnchor | null = null;
                if (selection instanceof TextSelection && !selection.empty) {
                  const start = view.coordsAtPos(selection.from);
                  const end = view.coordsAtPos(selection.to);
                  anchor = {
                    left: (Math.min(start.left, end.left) + Math.max(start.right, end.right)) / 2,
                    top: Math.min(start.top, end.top),
                    bottom: Math.max(start.bottom, end.bottom),
                  };
                }
                callbacksRef.current.onSelectionAnchorChange(anchor);
                callbacksRef.current.onCaretAnchorChange(caretAnchor);
              }, 32);
            });
        })
        .use(commonmark)
        .use(gfm)
        .use(listener)
        .use(history)
        .use(clipboard)
        // Drop/gap cursors + a guaranteed trailing paragraph: there is always
        // a place to arrow-down to after a code block, table, or formula.
        .use(cursor)
        .use(trailing)
        .use(ultraHighlight)
        .use(notesMath)
        .use(notesKeymap)
        .use(taskTogglePlugin)
        .use(activeStatesPlugin)
        .use(mediaView);
    },
    []
  );

  useEffect(() => {
    // A non-null handle must be usable immediately, including during a
    // source-to-formatted focus handoff after asynchronous initialization.
    if (loading || !get()) return;
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
          case "rowBelow":
            editor.action(callCommand(addRowAfterCommand.key));
            break;
          case "colRight":
            editor.action(callCommand(addColAfterCommand.key));
            break;
          case "rowDelete":
            editor.action((ctx) => {
              const view = ctx.get(editorViewCtx);
              deleteRow(view.state, view.dispatch);
            });
            break;
          case "colDelete":
            editor.action((ctx) => {
              const view = ctx.get(editorViewCtx);
              deleteColumn(view.state, view.dispatch);
            });
            break;
          case "tableDelete":
            editor.action((ctx) => {
              const view = ctx.get(editorViewCtx);
              deleteTable(view.state, view.dispatch);
            });
            break;
        }
        api.focus();
      },
      applyLink: (href) => {
        const editor = get();
        if (!editor) {
          return;
        }
        const hasLink = editor.action((ctx) =>
          markIsActive(ctx.get(editorViewCtx).state, linkSchema.type(ctx))
        );
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
        editor.action((ctx) => {
          const view = ctx.get(editorViewCtx);
          const { state } = view;
          const { empty, $from, ranges } = state.selection;
          const linkType = linkSchema.type(ctx);
          const tr = state.tr;
          if (empty) {
            // A mark toggle at a caret only changes future typing. Remove
            // the contiguous document link, including other formatting splits.
            const link = linkType.isInSet($from.marks());
            if (link) {
              const parent = $from.parent;
              let child = parent.childBefore($from.parentOffset);
              if (!child.node || !link.isInSet(child.node.marks)) {
                child = parent.childAfter($from.parentOffset);
              }
              if (child.node && link.isInSet(child.node.marks)) {
                let start = $from.start() + child.offset;
                let end = start + child.node.nodeSize;
                for (let i = child.index - 1; i >= 0; i -= 1) {
                  const sibling = parent.child(i);
                  if (!link.isInSet(sibling.marks)) break;
                  start -= sibling.nodeSize;
                }
                for (let i = child.index + 1; i < parent.childCount; i += 1) {
                  const sibling = parent.child(i);
                  if (!link.isInSet(sibling.marks)) break;
                  end += sibling.nodeSize;
                }
                tr.removeMark(start, end, link);
              }
            }
            tr.setStoredMarks(linkType.removeFromSet(state.storedMarks ?? $from.marks()));
          } else {
            // Table selections have a separate range for each selected cell.
            for (const range of ranges) {
              tr.removeMark(range.$from.pos, range.$to.pos, linkType);
            }
          }
          view.dispatch(tr);
        });
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
  }, [bindApi, get, loading]);

  useEffect(() => {
    return () => {
      if (activeFrameRef.current !== null) {
        window.clearTimeout(activeFrameRef.current);
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
