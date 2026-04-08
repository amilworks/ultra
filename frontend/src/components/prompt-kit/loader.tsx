import { cn } from "../../lib/cn";

type LoaderProps = {
  className?: string;
  text?: string;
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
