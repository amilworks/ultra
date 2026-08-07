import { cn } from "@/lib/utils";

type RunningStatusPillProps = {
  className?: string;
  label?: string;
  size?: "default" | "compact";
};

/* The point of light — Meridian's reserved brass, carrying its one meaning:
   this conversation's instrument is running right now. A still, breathing
   point rather than a spinner: the language replaces motion-as-busyness with
   light, and reserves the product's only warm colour for exactly this. */
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
      <span className="running-status-point" aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </span>
  );
}
