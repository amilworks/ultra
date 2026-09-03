import type { ResourceRecord } from "@/types";

const KIND_LABEL_MAX = 5;

/** A short, mono kind chip: the extension when there is one, else the kind. */
export const resourceMentionKindLabel = (resource: ResourceRecord): string => {
  const name = String(resource.original_name ?? "");
  const extension = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1) : "";
  const cleaned = extension.replace(/[^a-z0-9]/gi, "");
  if (cleaned.length > 0 && cleaned.length <= KIND_LABEL_MAX) {
    return cleaned.toUpperCase();
  }
  const kind = String(resource.resource_kind ?? "file").replace(/[^a-z0-9]/gi, "");
  return (kind || "file").slice(0, KIND_LABEL_MAX).toUpperCase();
};

/** The DOM id of one option, stable for aria-activedescendant on the textarea. */
export const resourceMentionOptionId = (listboxId: string, fileId: string): string =>
  `${listboxId}-${fileId.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
