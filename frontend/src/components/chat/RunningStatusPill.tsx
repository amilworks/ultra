import { cn } from "@/lib/utils";

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
      <span className="running-status-pill-beacon" aria-hidden="true">
        <span className="running-status-pill-beacon-core" />
      </span>
      <span className="sr-only">{label}</span>
    </span>
  );
}
