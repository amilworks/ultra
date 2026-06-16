import { cn } from "@/lib/utils";
import { CircularLoader } from "@/components/prompt-kit";

type RunningStatusPillProps = {
  className?: string;
  label?: string;
  size?: "default" | "compact";
};

export function RunningStatusPill({
  className,
  label = "Running",
  size = "default",
}: RunningStatusPillProps) {
  return (
    <span
      className={cn("running-status-pill", className)}
      data-size={size}
      role={size === "default" ? "status" : undefined}
      aria-live={size === "default" ? "polite" : undefined}
      aria-label={label}
      title={label}
    >
      <CircularLoader
        decorative
        size={size === "compact" ? "sm" : "md"}
        className="running-status-pill-loader"
      />
      <span className="sr-only">{label}</span>
    </span>
  );
}
