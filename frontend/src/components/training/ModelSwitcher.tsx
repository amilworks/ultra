// The model-agnostic seam (§14.4-H): a quiet Select over the registry, shown
// even with one model. A benchmark-only model reads "— benchmark only".

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { TrainingModelRecord } from "../../types";

export function ModelSwitcher({
  models,
  selected,
  onSelect,
}: {
  models: TrainingModelRecord[];
  selected: string;
  onSelect: (key: string) => void;
}) {
  if (models.length === 0) {
    return null;
  }
  return (
    <Select value={selected} onValueChange={onSelect}>
      <SelectTrigger size="sm" aria-label="Model">
        <SelectValue placeholder="Select model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model.key} value={model.key}>
            {model.name}
            {!model.supports_training ? " — benchmark only" : ""}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
