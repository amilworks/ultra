import { forwardRef } from "react";
import { ChevronDown, ChevronUp, X } from "lucide-react";

type TranscriptFindBarProps = {
  query: string;
  matchCount: number;
  /** Zero-based; rendered one-based. */
  currentIndex: number;
  onQueryChange: (value: string) => void;
  onNext: () => void;
  onPrevious: () => void;
  onClose: () => void;
};

/**
 * The ⌘F bar for find-within-conversation. Presentational — matching,
 * navigation and highlighting live in App, which owns the message data.
 *
 * Quiet by the house rules: panel surface, hairline border, muted count in
 * tabular figures (it ticks while navigating), and the shared
 * .chat-message-action buttons so hover/focus/touch-target behaviour is
 * identical to every other icon control in the chat.
 *
 * The ref exposes the input so ⌘F can focus-and-select it — pressing ⌘F with
 * the bar already open re-selects the query, matching browser find.
 */
export const TranscriptFindBar = forwardRef<HTMLInputElement, TranscriptFindBarProps>(
  function TranscriptFindBar(
    { query, matchCount, currentIndex, onQueryChange, onNext, onPrevious, onClose },
    inputRef
  ) {
    return (
      <div className="chat-find-bar" role="search" aria-label="Find in conversation">
        <input
          ref={inputRef}
          type="text"
          value={query}
          placeholder="Find in conversation"
          aria-label="Find in conversation"
          spellCheck={false}
          autoComplete="off"
          onChange={(event) => onQueryChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.nativeEvent.isComposing) {
              return;
            }
            if (event.key === "Enter") {
              event.preventDefault();
              if (event.shiftKey) {
                onPrevious();
              } else {
                onNext();
              }
            } else if (event.key === "Escape") {
              event.preventDefault();
              onClose();
            }
          }}
        />
        {query.trim() ? (
          <span className="chat-find-count" aria-live="polite">
            {matchCount > 0 ? `${currentIndex + 1} of ${matchCount}` : "Not found"}
          </span>
        ) : null}
        <button
          type="button"
          className="chat-message-action"
          aria-label="Previous match"
          disabled={matchCount === 0}
          onClick={onPrevious}
        >
          <ChevronUp className="size-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="chat-message-action"
          aria-label="Next match"
          disabled={matchCount === 0}
          onClick={onNext}
        >
          <ChevronDown className="size-4" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="chat-message-action"
          aria-label="Close find"
          onClick={onClose}
        >
          <X className="size-4" aria-hidden="true" />
        </button>
      </div>
    );
  }
);
