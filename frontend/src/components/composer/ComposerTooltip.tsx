import type { ReactNode } from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

/* The composer's tooltip: the house primitive with the composer's timing. A
   disabled control renders bare — a tooltip on a control that cannot act is
   noise, and its trigger would still be announced. */
export function ComposerTooltip({
  label,
  disabled = false,
  className = "app-composer-tooltip",
  children,
}: {
  label: ReactNode;
  disabled?: boolean;
  className?: string;
  children: ReactNode;
}) {
  if (disabled) {
    return <>{children}</>;
  }
  return (
    <Tooltip delayDuration={350}>
      <TooltipTrigger asChild onClick={(event) => event.stopPropagation()}>
        {children}
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8} className={className}>
        {label}
      </TooltipContent>
    </Tooltip>
  );
}
