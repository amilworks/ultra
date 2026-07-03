import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  type MutableRefObject,
} from "react";
import { useStickToBottomContext } from "use-stick-to-bottom";

export type ConversationScrollMemory = {
  scrollTop: number;
  wasNearBottom: boolean;
};

const SCROLL_RESTORE_BOTTOM_THRESHOLD_PX = 280;

export const captureConversationScrollMemory = (
  scrollElement: HTMLElement
): ConversationScrollMemory => {
  const maxScrollTop = Math.max(scrollElement.scrollHeight - scrollElement.clientHeight, 0);
  const scrollTop = Math.min(Math.max(scrollElement.scrollTop, 0), maxScrollTop);
  return {
    scrollTop,
    wasNearBottom: maxScrollTop - scrollTop <= SCROLL_RESTORE_BOTTOM_THRESHOLD_PX,
  };
};

export function ChatAutoScroll({
  conversationId,
  conversationHydrated,
  scrollRequestKey,
  scrollMemoryRef,
  scrollElementRef,
  scrollWriteBlockRef,
  onScrolledAwayChange,
}: {
  conversationId: string | null;
  conversationHydrated: boolean;
  scrollRequestKey: number;
  scrollMemoryRef: MutableRefObject<Record<string, ConversationScrollMemory>>;
  scrollElementRef: MutableRefObject<HTMLElement | null>;
  scrollWriteBlockRef: MutableRefObject<string | null>;
  // Lifts a "scrolled away from the bottom" signal (with hysteresis) so the
  // composer can collapse while the user reads back through a long answer.
  onScrolledAwayChange?: (away: boolean) => void;
}) {
  const { scrollRef, scrollToBottom, stopScroll } = useStickToBottomContext();
  const restoredConversationIdRef = useRef<string | null>(null);
  const liveConversationIdRef = useRef<string | null>(conversationId);
  const previousScrollRequestKeyRef = useRef(scrollRequestKey);
  const scrolledAwayRef = useRef(false);
  const setScrolledAway = useCallback(
    (away: boolean) => {
      if (scrolledAwayRef.current === away) {
        return;
      }
      scrolledAwayRef.current = away;
      onScrolledAwayChange?.(away);
    },
    [onScrolledAwayChange]
  );

  const rememberScrollPosition = useCallback(
    (targetConversationId: string | null) => {
      const scrollElement = scrollRef.current;
      if (!targetConversationId || !scrollElement) {
        return;
      }
      scrollMemoryRef.current[targetConversationId] = captureConversationScrollMemory(scrollElement);
    },
    [scrollMemoryRef, scrollRef]
  );

  useLayoutEffect(() => {
    liveConversationIdRef.current = conversationId;
  }, [conversationId]);

  useLayoutEffect(() => {
    const scrollElement = scrollRef.current;
    scrollElementRef.current = scrollElement;
    return () => {
      if (scrollElementRef.current === scrollElement) {
        scrollElementRef.current = null;
      }
    };
  }, [scrollElementRef, scrollRef]);

  useLayoutEffect(() => {
    if (!conversationId) {
      restoredConversationIdRef.current = null;
      return;
    }
    if (!conversationHydrated || restoredConversationIdRef.current === conversationId) {
      return;
    }
    restoredConversationIdRef.current = conversationId;
    let rafIdOne = 0;
    let rafIdTwo = 0;
    rafIdOne = requestAnimationFrame(() => {
      rafIdTwo = requestAnimationFrame(() => {
        const remembered = scrollMemoryRef.current[conversationId];
        if (remembered && !remembered.wasNearBottom) {
          const scrollElement = scrollRef.current;
          if (!scrollElement) {
            return;
          }
          stopScroll();
          const maxScrollTop = Math.max(scrollElement.scrollHeight - scrollElement.clientHeight, 0);
          scrollElement.scrollTop = Math.min(remembered.scrollTop, maxScrollTop);
          rememberScrollPosition(conversationId);
          if (scrollWriteBlockRef.current === conversationId) {
            scrollWriteBlockRef.current = null;
          }
          return;
        }
        scrollToBottom({ animation: "instant", ignoreEscapes: true });
        if (scrollWriteBlockRef.current === conversationId) {
          scrollWriteBlockRef.current = null;
        }
      });
    });
    return () => {
      if (rafIdOne) {
        cancelAnimationFrame(rafIdOne);
      }
      if (rafIdTwo) {
        cancelAnimationFrame(rafIdTwo);
      }
    };
  }, [
    conversationHydrated,
    conversationId,
    rememberScrollPosition,
    scrollMemoryRef,
    scrollRef,
    scrollToBottom,
    scrollWriteBlockRef,
    stopScroll,
  ]);

  useEffect(() => {
    if (!conversationId || !conversationHydrated) {
      return;
    }
    const scrollElement = scrollRef.current;
    if (!scrollElement) {
      return;
    }
    // A fresh conversation starts at the bottom, so the composer is expanded.
    setScrolledAway(false);
    const handleScroll = () => {
      // Hysteresis so a tiny scroll doesn't flicker the composer: collapse once
      // ~160px from the bottom, expand again only within ~48px of it.
      const distanceFromBottom =
        scrollElement.scrollHeight - scrollElement.scrollTop - scrollElement.clientHeight;
      setScrolledAway(
        scrolledAwayRef.current ? distanceFromBottom > 48 : distanceFromBottom > 160
      );
      if (
        liveConversationIdRef.current !== conversationId ||
        scrollWriteBlockRef.current === conversationId
      ) {
        return;
      }
      rememberScrollPosition(conversationId);
    };
    scrollElement.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      scrollElement.removeEventListener("scroll", handleScroll);
    };
  }, [
    conversationHydrated,
    conversationId,
    rememberScrollPosition,
    scrollRef,
    scrollWriteBlockRef,
    setScrolledAway,
  ]);

  useEffect(() => {
    if (!conversationId || scrollRequestKey === previousScrollRequestKeyRef.current) {
      return;
    }
    previousScrollRequestKeyRef.current = scrollRequestKey;
    scrollToBottom({ animation: "smooth", ignoreEscapes: true });
  }, [conversationId, scrollRequestKey, scrollToBottom]);

  return null;
}
