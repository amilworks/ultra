import { Loader } from "@/components/prompt-kit";

export function UploadFlightChip({
  inFlightCount,
  onOpen,
}: {
  inFlightCount: number;
  onOpen: () => void;
}) {
  if (inFlightCount <= 0) {
    return null;
  }
  return (
    <button
      type="button"
      className="upload-flight-chip"
      onClick={onOpen}
      aria-label={`Uploading ${inFlightCount} ${
        inFlightCount === 1 ? "file" : "files"
      }. Open Resources.`}
    >
      <Loader className="upload-flight-chip-spinner" aria-hidden="true" />
      <span className="upload-flight-chip-label">Uploading {inFlightCount}...</span>
    </button>
  );
}
