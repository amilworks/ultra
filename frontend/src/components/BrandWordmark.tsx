import type { ComponentPropsWithoutRef } from "react";

import { cn } from "@/lib/utils";

type BrandWordmarkProps = Omit<ComponentPropsWithoutRef<"span">, "children" | "role">;

export function BrandWordmark({
  className,
  "aria-label": ariaLabel = "BisQue Ultra",
  ...props
}: BrandWordmarkProps) {
  return (
    <span
      {...props}
      className={cn("brand-wordmark", className)}
      role="img"
      aria-label={ariaLabel}
    >
      <span className="brand-wordmark__bisque" aria-hidden="true">
        BisQue
      </span>
      <span className="brand-wordmark__ultra" aria-hidden="true">
        Ultra
      </span>
    </span>
  );
}
