import { useState, type MouseEvent } from "react";
import { Eye, Loader2 } from "lucide-react";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { classifyTextResource, formatChipLabel } from "@/lib/textFormat";
import type { ResourceRecord, ResourceTextHead } from "@/types";

const PEEK_BYTES = 8192;
const PEEK_LINES = 14;

type Props = {
  resource: ResourceRecord;
  fetchHead: (fileId: string, maxBytes: number) => Promise<ResourceTextHead>;
};

// A calm hover/tap "quick look" for text-like resources: opening the popover
// lazily fetches a tiny head (8 KB) and shows the first lines, so the Resources
// list becomes fast triage without opening the full viewer. Renders nothing for
// non-text resources.
export function ResourceQuickPeek({ resource, fetchHead }: Props) {
  const kind = classifyTextResource(resource);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [head, setHead] = useState<ResourceTextHead | null>(null);

  if (!kind) {
    return null;
  }

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    if (next && !head && !loading) {
      setLoading(true);
      setError(null);
      fetchHead(resource.file_id, PEEK_BYTES)
        .then((result) => setHead(result))
        .catch((err: unknown) => setError(err instanceof Error ? err.message : "Could not load preview"))
        .finally(() => setLoading(false));
    }
  };

  const stop = (event: MouseEvent) => event.stopPropagation();
  const lines = head ? head.text.split("\n").slice(0, PEEK_LINES) : [];

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="resource-quick-peek-trigger"
          aria-label={`Quick look at ${resource.original_name}`}
          title="Quick look"
          onClick={stop}
          onDoubleClick={stop}
        >
          <Eye aria-hidden="true" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" side="bottom" className="resource-quick-peek" onClick={stop}>
        <div className="resource-quick-peek-head">
          <span className="resource-quick-peek-name" title={resource.original_name}>
            {resource.original_name}
          </span>
          <span className="resource-quick-peek-chip">{formatChipLabel(kind, resource.original_name)}</span>
        </div>
        {loading ? (
          <div className="resource-quick-peek-status">
            <Loader2 className="text-viewer-spin" aria-hidden="true" />
            <span>Loading…</span>
          </div>
        ) : error ? (
          <div className="resource-quick-peek-status">{error}</div>
        ) : (
          <pre className="resource-quick-peek-body">
            {lines.join("\n")}
            {head?.truncated ? "\n…" : ""}
          </pre>
        )}
      </PopoverContent>
    </Popover>
  );
}
