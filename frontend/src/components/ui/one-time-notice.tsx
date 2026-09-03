import * as React from "react";
import { X } from "lucide-react";
import { Popover as PopoverPrimitive } from "radix-ui";

import { cn } from "@/lib/utils";
import { useLocalStorageState } from "@/lib/useLocalStorageState";

export const ONE_TIME_NOTICE_STORAGE_KEY = "bisque.frontend.oneTimeNotices.v1";

type OneTimeNoticeAction = {
  label: string;
  onSelect: () => void;
};

type OneTimeNoticeProps = {
  noticeId: string;
  audienceId: string;
  title: string;
  description: string;
  children: React.ReactNode;
  enabled?: boolean;
  action?: OneTimeNoticeAction;
  side?: React.ComponentProps<typeof PopoverPrimitive.Content>["side"];
  align?: React.ComponentProps<typeof PopoverPrimitive.Content>["align"];
  sideOffset?: number;
  className?: string;
  anchorClassName?: string;
  onDismiss?: () => void;
};

const noticeReceiptKey = (audienceId: string, noticeId: string): string => {
  const normalizedAudienceId = audienceId.trim() || "signed-in-user";
  return `${normalizedAudienceId}:${noticeId.trim()}`;
};

function OneTimeNotice({
  noticeId,
  audienceId,
  title,
  description,
  children,
  enabled = true,
  action,
  side = "bottom",
  align = "center",
  sideOffset = 14,
  className,
  anchorClassName,
  onDismiss,
}: OneTimeNoticeProps) {
  const titleId = React.useId();
  const descriptionId = React.useId();
  const receiptKey = noticeReceiptKey(audienceId, noticeId);
  const [receipts, setReceipts] = useLocalStorageState<Record<string, true>>(
    ONE_TIME_NOTICE_STORAGE_KEY,
    {}
  );
  const safeReceipts = React.useMemo(
    () =>
      receipts && typeof receipts === "object" && !Array.isArray(receipts) ? receipts : {},
    [receipts]
  );
  const seen = safeReceipts[receiptKey] === true;
  const open = enabled && !seen;

  const dismiss = React.useCallback(() => {
    if (!seen) {
      const nextReceipts = { ...safeReceipts, [receiptKey]: true as const };
      // Persist before invoking an optional action. The action may navigate or
      // unmount the anchor before the storage hook's effect gets a chance to run.
      try {
        window.localStorage.setItem(ONE_TIME_NOTICE_STORAGE_KEY, JSON.stringify(nextReceipts));
      } catch {
        // A blocked storage write should not make the notice impossible to close
        // for the current session.
      }
      setReceipts(nextReceipts);
    }
    onDismiss?.();
  }, [onDismiss, receiptKey, safeReceipts, seen, setReceipts]);

  const handleAction = React.useCallback(() => {
    dismiss();
    action?.onSelect();
  }, [action, dismiss]);

  return (
    <PopoverPrimitive.Root open={open}>
      <PopoverPrimitive.Anchor asChild>
        <span
          className={cn("one-time-notice-anchor", anchorClassName)}
          data-one-time-notice={noticeId}
          data-notice-state={open ? "open" : seen ? "seen" : "ineligible"}
        >
          {children}
        </span>
      </PopoverPrimitive.Anchor>
      {open ? (
        <PopoverPrimitive.Portal>
          <PopoverPrimitive.Content
            role="dialog"
            aria-modal="false"
            aria-live="polite"
            aria-labelledby={titleId}
            aria-describedby={descriptionId}
            side={side}
            align={align}
            sideOffset={sideOffset}
            collisionPadding={12}
            onOpenAutoFocus={(event) => event.preventDefault()}
            onCloseAutoFocus={(event) => event.preventDefault()}
            onFocusOutside={(event) => event.preventDefault()}
            onEscapeKeyDown={(event) => {
              event.preventDefault();
              dismiss();
            }}
            onPointerDownOutside={dismiss}
            className={cn("one-time-notice", className)}
          >
            <div className="one-time-notice-header">
              <h2 id={titleId} className="one-time-notice-title">
                {title}
              </h2>
              <button
                type="button"
                className="one-time-notice-dismiss"
                aria-label={`Dismiss ${title}`}
                onClick={dismiss}
              >
                <X aria-hidden="true" />
              </button>
            </div>
            <p id={descriptionId} className="one-time-notice-description">
              {description}
            </p>
            {action ? (
              <button type="button" className="one-time-notice-action" onClick={handleAction}>
                {action.label}
              </button>
            ) : null}
            <PopoverPrimitive.Arrow className="one-time-notice-arrow" width={20} height={10} />
          </PopoverPrimitive.Content>
        </PopoverPrimitive.Portal>
      ) : null}
    </PopoverPrimitive.Root>
  );
}

export { OneTimeNotice };
export type { OneTimeNoticeAction, OneTimeNoticeProps };
