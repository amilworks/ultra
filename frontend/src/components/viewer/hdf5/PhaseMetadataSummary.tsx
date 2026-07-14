import { describePhaseMetadata } from "./formatters";

type PhaseMetadataSummaryProps = {
  phaseNames: string[];
  source?: string | null;
  provenance?: string | null;
};

export function PhaseMetadataSummary({
  phaseNames,
  source,
  provenance,
}: PhaseMetadataSummaryProps) {
  if (phaseNames.length === 0) {
    return null;
  }
  const description = describePhaseMetadata(source, provenance);
  return (
    <div className="viewer-hdf-context-card-section" data-hdf5-material-phases="true">
      <strong>{description.label}</strong>
      <p>{phaseNames.join(" • ")}</p>
      {description.detail ? (
        <p className="viewer-hdf-context-provenance">{description.detail}</p>
      ) : null}
    </div>
  );
}
