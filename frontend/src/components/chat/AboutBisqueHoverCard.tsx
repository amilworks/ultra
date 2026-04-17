import { Info } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useBreakpoint } from "@/hooks/use-breakpoint";

export function AboutBisqueHoverCard() {
  const isMobile = useBreakpoint(640);

  if (isMobile) {
    return (
      <Sheet>
        <SheetTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="size-11 shrink-0 rounded-full text-muted-foreground transition-colors hover:text-foreground"
            aria-label="About BisQue Ultra"
          >
            <Info className="size-4" />
          </Button>
        </SheetTrigger>
        <SheetContent
          side="bottom"
          className="rounded-t-[1.75rem] px-4 pb-[calc(1.35rem+env(safe-area-inset-bottom,0px))] pt-3"
        >
          <SheetHeader className="space-y-2 text-left">
            <SheetTitle className="text-base">About BisQue Ultra</SheetTitle>
            <SheetDescription className="text-sm leading-6 text-muted-foreground">
              BisQue Ultra is a scientific imaging workbench for reproducible,
              tool-guided analysis.
            </SheetDescription>
          </SheetHeader>
          <div className="mt-4 space-y-3">
            <p className="text-sm leading-6 text-muted-foreground">
              It combines BisQue-backed data management, a FastAPI control plane,
              and a React interface so image review, model calls, and long-running
              workflows stay visible in one place.
            </p>
            <p className="text-sm leading-6 text-muted-foreground">
              The goal is a system that another team can trust, extend, and
              operate without relying on hidden local context.
            </p>
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <HoverCard openDelay={120} closeDelay={120}>
      <HoverCardTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          className="size-11 shrink-0 rounded-full text-muted-foreground transition-colors hover:text-foreground md:size-8"
          aria-label="About BisQue Ultra"
        >
          <Info className="size-4" />
        </Button>
      </HoverCardTrigger>
      <HoverCardContent
        align="end"
        className="w-[min(24rem,calc(100vw-1.5rem))] space-y-3 rounded-2xl border px-4 py-3.5"
      >
        <div className="space-y-1.5">
          <p className="text-sm font-semibold">About BisQue Ultra</p>
          <p className="text-sm leading-6 text-muted-foreground">
            BisQue Ultra is a scientific imaging workbench for reproducible,
            tool-guided analysis.
          </p>
          <p className="text-sm leading-6 text-muted-foreground">
            It combines BisQue-backed data management, a FastAPI control plane,
            and a React interface so image review, model calls, and long-running
            workflows stay visible in one place.
          </p>
        </div>
        <p className="text-sm leading-6 text-muted-foreground">
          The goal is a system that another team can trust, extend, and
          operate without relying on hidden local context.
        </p>
      </HoverCardContent>
    </HoverCard>
  );
}
