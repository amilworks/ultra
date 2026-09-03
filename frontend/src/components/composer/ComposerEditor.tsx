import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  type MutableRefObject,
} from "react";
import { baseKeymap, splitBlock } from "prosemirror-commands";
import { history, redo, undo } from "prosemirror-history";
import { keymap } from "prosemirror-keymap";
import type { Node as PMNode } from "prosemirror-model";
import { EditorState, Plugin, PluginKey, Selection, type Transaction } from "prosemirror-state";
import { Decoration, DecorationSet, EditorView, type NodeView } from "prosemirror-view";

import { briefFileTokensInText } from "@/features/chat/brief-tokens";

import type { ComposerEditorProps, ComposerHandle, ComposerMention } from "./composerHandle";
import {
  appendTokenAt,
  deleteTokenBackward,
  deleteTokenForward,
  docFromText,
  findTokenPosition,
  insertMultilineText,
  insertTokenAt,
  isDocEmpty,
  mentionAtSelection,
  removeTokenNode,
  reopenMentionAt,
  replaceDocFromText,
  sameTokens,
  textFromDoc,
  tokensInDoc,
} from "./composerSchema";

/* The composer's editor: a ProseMirror view over the schema in composerSchema.
   It owns the text and the tokens; the app owns everything around them.
   Loaded as its own chunk — ComposerFallbackEditor covers the first paint. */

type DecorationState = { placeholder: string; goneFileIds: readonly string[] };

const decorationKey = new PluginKey<DecorationState>("composer-decorations");
const EXTERNAL = "composer-external";

const buildDecorations = (doc: PMNode, config: DecorationState): DecorationSet => {
  const decorations: Decoration[] = [];
  if (isDocEmpty(doc) && config.placeholder) {
    decorations.push(
      Decoration.widget(
        1,
        () => {
          const span = document.createElement("span");
          span.className = "composer-placeholder";
          span.setAttribute("aria-hidden", "true");
          span.textContent = config.placeholder;
          return span;
        },
        { side: 1, ignoreSelection: true, key: `placeholder:${config.placeholder}` }
      )
    );
  }
  if (config.goneFileIds.length > 0) {
    const gone = new Set(config.goneFileIds);
    doc.descendants((node, pos) => {
      if (node.type.name === "fileToken" && gone.has(String(node.attrs.fileId))) {
        decorations.push(Decoration.node(pos, pos + node.nodeSize, { class: "composer-token-gone" }));
      }
      return true;
    });
  }
  return DecorationSet.create(doc, decorations);
};

const decorationPlugin = (initial: DecorationState) =>
  new Plugin<DecorationState>({
    key: decorationKey,
    state: {
      init: () => initial,
      apply: (tr, previous) => (tr.getMeta(decorationKey) as DecorationState | undefined) ?? previous,
    },
    props: {
      decorations(state) {
        return buildDecorations(state.doc, decorationKey.getState(state) ?? initial);
      },
    },
  });

/* Enter belongs to the app (send / queue / steer); Shift-Enter is the new line. */
const keymapWithoutEnter = Object.fromEntries(
  Object.entries(baseKeymap).filter(([key]) => key !== "Enter")
);

const CLOSE_GLYPH =
  '<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none"/></svg>';

class TokenView implements NodeView {
  dom: HTMLSpanElement;
  private kind: HTMLSpanElement;
  private name: HTMLSpanElement;
  private remove: HTMLButtonElement;

  constructor(
    private node: PMNode,
    private readonly view: EditorView,
    private readonly getPos: () => number | undefined,
    private readonly propsRef: MutableRefObject<ComposerEditorProps>,
    private readonly live: Set<TokenView>
  ) {
    live.add(this);
    this.dom = document.createElement("span");
    this.dom.className = "composer-token";
    this.dom.setAttribute("contenteditable", "false");
    this.kind = document.createElement("span");
    this.kind.className = "composer-token-kind";
    this.name = document.createElement("span");
    this.name.className = "composer-token-name";
    this.remove = document.createElement("button");
    this.remove.type = "button";
    this.remove.tabIndex = -1;
    this.remove.className = "composer-token-remove";
    this.remove.innerHTML = CLOSE_GLYPH;
    this.remove.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    this.remove.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const fileId = String(this.node.attrs.fileId);
      const tr = removeTokenNode(this.view.state, fileId);
      if (tr) {
        this.view.dispatch(tr);
      }
      this.propsRef.current.onTokenRemoveClick?.(fileId);
      this.view.focus();
    });
    this.dom.append(this.kind, this.name, this.remove);
    this.render();
  }

  /* The details behind a token (library title, kind, gone) live in app state
     that changes AFTER the node was drawn — a pick stages its file a render
     later. The editor calls this on every token when that state moves. */
  refresh(): void {
    this.render();
  }

  private render(): void {
    const label = String(this.node.attrs.label);
    const fileId = String(this.node.attrs.fileId);
    const details = this.propsRef.current.tokenDetails(fileId);
    this.dom.setAttribute("data-file-id", fileId);
    this.dom.setAttribute("data-label", label);
    this.dom.title = details?.title ?? `@${label}`;
    this.kind.textContent = details?.kind ?? "";
    this.kind.hidden = !details?.kind;
    this.name.textContent = label;
    this.remove.setAttribute("aria-label", `Remove ${label}`);
    this.dom.classList.toggle("composer-token-gone", Boolean(details?.gone));
  }

  update(node: PMNode): boolean {
    if (node.type !== this.node.type) {
      return false;
    }
    this.node = node;
    this.render();
    return true;
  }

  selectNode(): void {
    this.dom.classList.add("composer-token-selected");
  }

  deselectNode(): void {
    this.dom.classList.remove("composer-token-selected");
  }

  stopEvent(event: Event): boolean {
    const target = event.target as Node | null;
    return target !== null && this.remove.contains(target);
  }

  ignoreMutation(): boolean {
    return true;
  }

  destroy(): void {
    this.live.delete(this);
  }
}

const ariaAttributes = (props: ComposerEditorProps): Record<string, string> => {
  const attributes: Record<string, string> = {
    class: "composer-editor",
    role: "textbox",
    "aria-multiline": "true",
    "aria-label": props.ariaLabel,
    "aria-autocomplete": "list",
    "aria-expanded": props.mentionOpen ? "true" : "false",
    "aria-disabled": props.disabled ? "true" : "false",
  };
  if (props.mentionOpen && props.listboxId) {
    attributes["aria-controls"] = props.listboxId;
  }
  if (props.mentionOpen && props.activeOptionId) {
    attributes["aria-activedescendant"] = props.activeOptionId;
  }
  return attributes;
};

const sameMention = (left: ComposerMention | null, right: ComposerMention | null): boolean =>
  left === right || (left !== null && right !== null && left.query === right.query);

/** The doc's tokens no longer agree with what the text + registry would produce
    — a label was registered after its `@label` was typed (draft recovery). */
const tokensDrift = (doc: PMNode, text: string, registry: ComposerEditorProps["tokens"]): boolean => {
  const expected = briefFileTokensInText(text, registry).map((token) => token.fileId);
  const present = tokensInDoc(doc).map((token) => token.fileId);
  return expected.length !== present.length || expected.some((id, index) => id !== present[index]);
};

export const ComposerEditor = forwardRef<ComposerHandle, ComposerEditorProps>(function ComposerEditor(
  props,
  ref
) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const lastText = useRef(props.value);
  const lastTokens = useRef(briefFileTokensInText(props.value, props.tokens));
  const lastMention = useRef<ComposerMention | null>(null);
  const plainPaste = useRef(false);
  const tokenViews = useRef(new Set<TokenView>());

  useLayoutEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }
    const initial = propsRef.current;
    const state = EditorState.create({
      doc: docFromText(initial.value, initial.tokens),
      plugins: [
        decorationPlugin({ placeholder: initial.placeholder, goneFileIds: initial.goneFileIds }),
        history(),
        keymap({
          "Shift-Enter": splitBlock,
          "Mod-z": undo,
          "Shift-Mod-z": redo,
          "Mod-y": redo,
          // A token is one unit to the keyboard: Backspace after it, or Delete
          // before it, removes it whole. Everything else falls through.
          Backspace: (state, dispatch) => {
            const tr = deleteTokenBackward(state);
            if (!tr) {
              return false;
            }
            dispatch?.(tr);
            return true;
          },
          Delete: (state, dispatch) => {
            const tr = deleteTokenForward(state);
            if (!tr) {
              return false;
            }
            dispatch?.(tr);
            return true;
          },
        }),
        keymap(keymapWithoutEnter),
      ],
    });
    const emit = (next: EditorState, tr: Transaction) => {
      const current = propsRef.current;
      const external = Boolean(tr.getMeta(EXTERNAL));
      if (tr.docChanged) {
        const text = textFromDoc(next.doc);
        const tokens = tokensInDoc(next.doc);
        if (!external && text !== lastText.current) {
          lastText.current = text;
          current.onValueChange(text);
        } else {
          lastText.current = text;
        }
        if (!sameTokens(tokens, lastTokens.current)) {
          lastTokens.current = tokens;
          if (!external) {
            current.onTokensChange(tokens);
          }
        }
      }
      if (tr.docChanged || tr.selectionSet) {
        const range = mentionAtSelection(next);
        const mention = range ? { query: range.query } : null;
        if (!sameMention(mention, lastMention.current)) {
          lastMention.current = mention;
          current.onMentionChange(mention);
        }
      }
    };
    const view = new EditorView(mount, {
      state,
      editable: () => !propsRef.current.disabled,
      attributes: ariaAttributes(initial),
      dispatchTransaction(tr) {
        const next = view.state.apply(tr);
        view.updateState(next);
        emit(next, tr);
      },
      handleKeyDown(_view, event) {
        const current = propsRef.current;
        if (current.onKeyDown?.(event)) {
          return true;
        }
        if (
          event.key === "Enter" &&
          !event.shiftKey &&
          !event.altKey &&
          !event.isComposing &&
          !view.composing
        ) {
          return current.onEnter(event);
        }
        return false;
      },
      handlePaste(_view, event) {
        if (plainPaste.current) {
          return false;
        }
        const current = propsRef.current;
        if (current.onPaste?.(event)) {
          return true;
        }
        const text = event.clipboardData?.getData("text/plain") ?? "";
        if (!text) {
          return false;
        }
        event.preventDefault();
        plainPaste.current = true;
        try {
          view.pasteText(text, event);
        } finally {
          plainPaste.current = false;
        }
        return true;
      },
      handleDOMEvents: {
        focus: () => {
          propsRef.current.onFocusChange(true);
          return false;
        },
        blur: () => {
          propsRef.current.onFocusChange(false);
          if (lastMention.current !== null) {
            lastMention.current = null;
            propsRef.current.onMentionChange(null);
          }
          return false;
        },
      },
      nodeViews: {
        fileToken: (node, editorView, getPos) =>
          new TokenView(
            node,
            editorView,
            getPos as () => number | undefined,
            propsRef,
            tokenViews.current
          ),
      },
    });
    viewRef.current = view;
    lastText.current = initial.value;
    lastTokens.current = tokensInDoc(state.doc);
    initial.onReady?.();
    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, []);

  /* External text: a conversation switch, a recalled prompt, a queued follow-up
     clearing the draft, a registry that learned a label after the fact. */
  useEffect(() => {
    const view = viewRef.current;
    if (!view) {
      return;
    }
    const current = textFromDoc(view.state.doc);
    if (current !== props.value || tokensDrift(view.state.doc, props.value, props.tokens)) {
      const tr = replaceDocFromText(view.state, props.value, props.tokens, current === "" ? "end" : "keep");
      view.dispatch(tr.setMeta(EXTERNAL, true).setMeta("addToHistory", false));
    }
  }, [props.value, props.tokens]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) {
      return;
    }
    view.dispatch(
      view.state.tr.setMeta(decorationKey, {
        placeholder: props.placeholder,
        goneFileIds: props.goneFileIds,
      })
    );
  }, [props.placeholder, props.goneFileIds]);

  useEffect(() => {
    for (const tokenView of tokenViews.current) {
      tokenView.refresh();
    }
  }, [props.tokenDetails, props.goneFileIds]);

  const { ariaLabel, mentionOpen, listboxId, activeOptionId, disabled } = props;
  useEffect(() => {
    viewRef.current?.setProps({
      attributes: ariaAttributes({
        ...propsRef.current,
        ariaLabel,
        mentionOpen,
        listboxId,
        activeOptionId,
        disabled,
      }),
    });
  }, [ariaLabel, mentionOpen, listboxId, activeOptionId, disabled]);

  useImperativeHandle(
    ref,
    (): ComposerHandle => ({
      get element() {
        return viewRef.current?.dom ?? null;
      },
      get disabled() {
        return propsRef.current.disabled;
      },
      get value() {
        const view = viewRef.current;
        return view ? textFromDoc(view.state.doc) : propsRef.current.value;
      },
      focus(options) {
        const view = viewRef.current;
        if (!view) {
          return;
        }
        if (options?.caret === "end") {
          view.dispatch(
            view.state.tr.setSelection(Selection.atEnd(view.state.doc)).setMeta("addToHistory", false)
          );
        }
        view.focus();
      },
      isFocused() {
        return viewRef.current?.hasFocus() ?? false;
      },
      setValue(text) {
        const view = viewRef.current;
        if (!view) {
          return;
        }
        view.dispatch(replaceDocFromText(view.state, text, propsRef.current.tokens, "end"));
      },
      insertText(text) {
        const view = viewRef.current;
        if (!view) {
          return;
        }
        if (!view.hasFocus()) {
          view.dispatch(view.state.tr.setSelection(Selection.atEnd(view.state.doc)));
        }
        view.dispatch(insertMultilineText(view.state.tr, text));
      },
      acceptMention(token) {
        const view = viewRef.current;
        if (!view) {
          return;
        }
        const range = mentionAtSelection(view.state);
        if (!range) {
          view.dispatch(appendTokenAt(view.state, token, view.hasFocus()));
          return;
        }
        view.dispatch(insertTokenAt(view.state, range.from, range.to, token));
        view.focus();
      },
      appendToken(token) {
        const view = viewRef.current;
        if (!view || findTokenPosition(view.state.doc, token.fileId) !== null) {
          return;
        }
        view.dispatch(appendTokenAt(view.state, token, view.hasFocus()));
      },
      removeToken(fileId) {
        const view = viewRef.current;
        if (!view) {
          return;
        }
        const tr = removeTokenNode(view.state, fileId);
        if (tr) {
          view.dispatch(tr);
        }
      },
      reopenMentionFor(fileId) {
        const view = viewRef.current;
        if (!view) {
          return;
        }
        const tr = reopenMentionAt(view.state, fileId);
        if (tr) {
          view.dispatch(tr);
          view.focus();
        }
      },
      mentionRect() {
        const view = viewRef.current;
        if (!view) {
          return null;
        }
        const range = mentionAtSelection(view.state);
        if (!range) {
          return null;
        }
        try {
          const coords = view.coordsAtPos(range.from);
          return { left: coords.left, top: coords.top, bottom: coords.bottom };
        } catch {
          return null;
        }
      },
    }),
    []
  );

  return <div ref={mountRef} className="composer-editor-mount" />;
});

export default ComposerEditor;
