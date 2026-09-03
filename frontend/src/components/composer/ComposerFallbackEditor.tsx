import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
} from "react";

import {
  briefBackspaceTarget,
  briefCaretAfterArrow,
  briefDeleteTarget,
  briefFileTokensInText,
  briefMentionQueryAtCaret,
  insertBriefToken,
  parseBriefSegments,
  removeBriefSegment,
} from "@/features/chat/brief-tokens";

import type { ComposerEditorProps, ComposerHandle, ComposerMention } from "./composerHandle";

/* A plain textarea that speaks the same handle as the ProseMirror editor. It
   covers the first paint while the editor chunk loads and stays if that chunk
   never arrives: tokens are `@label` runs, atomic to Backspace and the arrows
   through the grammar module, just without pills. */

const MAX_HEIGHT = 240;

export const ComposerFallbackEditor = forwardRef<ComposerHandle, ComposerEditorProps>(
  function ComposerFallbackEditor(props, ref) {
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);
    const propsRef = useRef(props);
    propsRef.current = props;
    const lastMention = useRef<ComposerMention | null>(null);
    const lastTokenKey = useRef<string | null>(null);
    const pendingCaret = useRef<number | null>(null);

    useLayoutEffect(() => {
      const node = textareaRef.current;
      if (!node) {
        return;
      }
      node.style.height = "auto";
      node.style.height = `${Math.min(node.scrollHeight, MAX_HEIGHT)}px`;
      if (pendingCaret.current !== null) {
        const caret = Math.min(pendingCaret.current, node.value.length);
        pendingCaret.current = null;
        node.setSelectionRange(caret, caret);
      }
    }, [props.value]);

    useEffect(() => {
      const tokens = briefFileTokensInText(props.value, props.tokens);
      const key = tokens.map((token) => `${token.fileId}:${token.label}`).join("|");
      if (lastTokenKey.current !== null && key !== lastTokenKey.current) {
        propsRef.current.onTokensChange(tokens);
      }
      lastTokenKey.current = key;
    }, [props.value, props.tokens]);

    useEffect(() => {
      propsRef.current.onReady?.();
    }, []);

    const emitMention = useCallback(() => {
      const node = textareaRef.current;
      if (!node) {
        return;
      }
      const query =
        node.selectionStart === node.selectionEnd
          ? briefMentionQueryAtCaret(node.value, node.selectionStart, propsRef.current.tokens)
          : null;
      const mention = query ? { query: query.query } : null;
      const previous = lastMention.current;
      if ((mention === null) !== (previous === null) || (mention && previous && mention.query !== previous.query)) {
        lastMention.current = mention;
        propsRef.current.onMentionChange(mention);
      }
    }, []);

    const commit = useCallback((text: string, caret: number) => {
      pendingCaret.current = caret;
      propsRef.current.onValueChange(text);
    }, []);

    useImperativeHandle(
      ref,
      (): ComposerHandle => ({
        get element() {
          return textareaRef.current;
        },
        get disabled() {
          return propsRef.current.disabled;
        },
        get value() {
          return textareaRef.current?.value ?? propsRef.current.value;
        },
        focus(options) {
          const node = textareaRef.current;
          if (!node) {
            return;
          }
          node.focus({ preventScroll: options?.preventScroll ?? true });
          if (options?.caret === "end") {
            const end = node.value.length;
            node.setSelectionRange(end, end);
          }
        },
        isFocused() {
          return textareaRef.current !== null && document.activeElement === textareaRef.current;
        },
        setValue(text) {
          commit(text, text.length);
        },
        insertText(text) {
          const node = textareaRef.current;
          const value = node?.value ?? propsRef.current.value;
          const focused = node !== null && document.activeElement === node;
          const start = focused ? node.selectionStart : value.length;
          const end = focused ? node.selectionEnd : value.length;
          commit(value.slice(0, start) + text + value.slice(end), start + text.length);
        },
        acceptMention(token) {
          const node = textareaRef.current;
          const value = node?.value ?? propsRef.current.value;
          const caret = node ? node.selectionStart : value.length;
          const query = node ? briefMentionQueryAtCaret(value, caret, propsRef.current.tokens) : null;
          const edit = query
            ? insertBriefToken(value, query.start, caret, token.label)
            : insertBriefToken(value, caret, caret, token.label);
          commit(edit.text, edit.caret);
        },
        appendToken(token) {
          const node = textareaRef.current;
          const value = node?.value ?? propsRef.current.value;
          if (briefFileTokensInText(value, [token]).length > 0) {
            return;
          }
          const focused = node !== null && document.activeElement === node;
          const caret = focused ? node.selectionStart : value.length;
          const edit = insertBriefToken(value, caret, caret, token.label);
          commit(edit.text, focused ? edit.caret : edit.text.length);
        },
        removeToken(fileId) {
          const value = textareaRef.current?.value ?? propsRef.current.value;
          const segment = parseBriefSegments(value, propsRef.current.tokens).find(
            (candidate) => candidate.kind === "file" && candidate.fileId === fileId
          );
          if (!segment || segment.kind !== "file") {
            return;
          }
          const edit = removeBriefSegment(value, segment);
          commit(edit.text, edit.caret);
        },
        reopenMentionFor(fileId) {
          const value = textareaRef.current?.value ?? propsRef.current.value;
          const segment = parseBriefSegments(value, propsRef.current.tokens).find(
            (candidate) => candidate.kind === "file" && candidate.fileId === fileId
          );
          if (!segment || segment.kind !== "file") {
            return;
          }
          const text = `${value.slice(0, segment.start)}@${value.slice(segment.end)}`;
          commit(text, segment.start + 1);
          textareaRef.current?.focus();
        },
        mentionRect() {
          return null;
        },
      }),
      [commit]
    );

    return (
      <textarea
        ref={textareaRef}
        className="composer-editor composer-editor-fallback"
        value={props.value}
        placeholder={props.placeholder}
        aria-label={props.ariaLabel}
        aria-autocomplete="list"
        aria-expanded={props.mentionOpen}
        aria-controls={props.mentionOpen ? props.listboxId : undefined}
        aria-activedescendant={props.mentionOpen ? props.activeOptionId : undefined}
        disabled={props.disabled}
        rows={1}
        onChange={(event) => {
          propsRef.current.onValueChange(event.target.value);
        }}
        onSelect={emitMention}
        onKeyUp={emitMention}
        onClick={emitMention}
        onFocus={() => propsRef.current.onFocusChange(true)}
        onBlur={() => {
          propsRef.current.onFocusChange(false);
          if (lastMention.current !== null) {
            lastMention.current = null;
            propsRef.current.onMentionChange(null);
          }
        }}
        onPaste={(event) => {
          if (propsRef.current.onPaste?.(event.nativeEvent)) {
            return;
          }
        }}
        onKeyDown={(event) => {
          const native = event.nativeEvent;
          if (propsRef.current.onKeyDown?.(native)) {
            return;
          }
          const node = event.currentTarget;
          if (event.key === "Enter" && !event.shiftKey && !event.altKey && !native.isComposing) {
            propsRef.current.onEnter(native);
            return;
          }
          if (node.selectionStart !== node.selectionEnd) {
            return;
          }
          const segments = parseBriefSegments(node.value, propsRef.current.tokens);
          const caret = node.selectionStart;
          if (event.key === "Backspace") {
            const target = briefBackspaceTarget(segments, caret);
            if (target) {
              event.preventDefault();
              const edit = removeBriefSegment(node.value, target);
              commit(edit.text, edit.caret);
            }
            return;
          }
          if (event.key === "Delete") {
            const target = briefDeleteTarget(segments, caret);
            if (target) {
              event.preventDefault();
              const edit = removeBriefSegment(node.value, target);
              commit(edit.text, edit.caret);
            }
            return;
          }
          if ((event.key === "ArrowLeft" || event.key === "ArrowRight") && !event.shiftKey) {
            const next = briefCaretAfterArrow(segments, caret, event.key === "ArrowLeft" ? -1 : 1);
            if (next !== null) {
              event.preventDefault();
              node.setSelectionRange(next, next);
              emitMention();
            }
          }
        }}
      />
    );
  }
);

export default ComposerFallbackEditor;
