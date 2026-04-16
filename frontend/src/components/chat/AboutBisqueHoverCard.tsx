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
              BisQue Ultra was created by Amil Khan, a PhD student in the UCSB Vision
              Research Lab.
            </SheetDescription>
          </SheetHeader>
          <div className="mt-4 space-y-3">
            <p className="text-sm leading-6 text-muted-foreground">
              The lab is led by Professor B.S. Manjunath, principal investigator of the
              Vision Research Lab at UCSB, with research spanning computer vision, image
              processing, and machine learning applications.
            </p>
            <p className="text-sm leading-6 text-muted-foreground">
              For praise, feature requests, or the occasional friendly complaint, please
              begin with the creator.
            </p>
            <a
              href="https://example.com/vision-lab"
              target="_blank"
              rel="noreferrer"
              className="inline-flex text-sm font-medium text-foreground underline underline-offset-4 transition-colors hover:text-primary"
            >
              Visit UCSB VRL
            </a>
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
            BisQue Ultra was created by Amil Khan, a PhD student in the UCSB Vision
            Research Lab.
          </p>
          <p className="text-sm leading-6 text-muted-foreground">
            The lab is led by Professor B.S. Manjunath, principal investigator of the
            Vision Research Lab at UCSB, with research spanning computer vision, image
            processing, and machine learning applications.
          </p>
        </div>
        <p className="text-sm leading-6 text-muted-foreground">
          For praise, feature requests, or the occasional friendly complaint, please
          begin with the creator.
        </p>
        <a
          href="https://example.com/vision-lab"
          target="_blank"
          rel="noreferrer"
          className="inline-flex text-sm font-medium text-foreground underline underline-offset-4 transition-colors hover:text-primary"
        >
          Visit UCSB VRL
        </a>
      </HoverCardContent>
    </HoverCard>
  );
}
