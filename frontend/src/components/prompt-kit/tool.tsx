import React from "react";
import { cn } from "../../lib/cn";

export type ToolPart = {
  type: string;
  state: string;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  toolCallId?: string;
  errorText?: string;
};

type ToolProps = {
  toolPart: ToolPart;
  defaultOpen?: boolean;
  className?: string;
};

export function Tool({ toolPart, defaultOpen = false, className }: ToolProps) {
  const [open, setOpen] = React.useState(defaultOpen);

  return (
    <section className={cn("pk-tool", className)}>
      <button
        type="button"
        className="pk-tool-header"
        onClick={() => {
          setOpen((current) => !current);
        }}
      >
        <span className="pk-tool-name">{toolPart.type}</span>
        <span className={cn("pk-tool-state", `pk-tool-state-${toolPart.state.toLowerCase()}`)}>
          {toolPart.state}
        </span>
      </button>
      {open ? (
        <div className="pk-tool-body">
          {toolPart.toolCallId ? (
            <p className="pk-tool-line">
              <strong>Call:</strong> {toolPart.toolCallId}
            </p>
          ) : null}
          {toolPart.input ? (
            <div className="pk-tool-section">
              <strong>Input</strong>
              <pre>{JSON.stringify(toolPart.input, null, 2)}</pre>
            </div>
          ) : null}
          {toolPart.output ? (
            <div className="pk-tool-section">
              <strong>Output</strong>
              <pre>{JSON.stringify(toolPart.output, null, 2)}</pre>
            </div>
          ) : null}
          {toolPart.errorText ? (
            <div className="pk-tool-section">
              <strong>Error</strong>
              <p className="pk-tool-error">{toolPart.errorText}</p>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
