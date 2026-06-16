import { cn } from "../../lib/cn";

type LoaderProps = {
  className?: string;
  text?: string;
  size?: "sm" | "md" | "lg";
};

type CircularLoaderProps = {
  className?: string;
  decorative?: boolean;
  label?: string;
  size?: "sm" | "md" | "lg";
};

export function Loader({ className, text, size = "md" }: LoaderProps) {
  return (
    <span className={cn("pk-loader", `pk-loader-${size}`, className)} role="status" aria-live="polite">
      <span className="pk-loader-dot" />
      <span className="pk-loader-dot" />
      <span className="pk-loader-dot" />
      {text ? <span className="pk-loader-text">{text}</span> : null}
    </span>
  );
}

export function CircularLoader({
  className,
  decorative = false,
  label = "Loading",
  size = "md",
}: CircularLoaderProps) {
  return (
    <span
      className={cn("pk-circular-loader", `pk-circular-loader-${size}`, className)}
      role={decorative ? undefined : "status"}
      aria-hidden={decorative ? "true" : undefined}
      aria-live={decorative ? undefined : "polite"}
    >
      {decorative ? null : <span className="sr-only">{label}</span>}
    </span>
  );
}
