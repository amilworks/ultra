import { formatBytes, shortId, toLocalDateTime } from "../lib/format";
import type { ArtifactRecord, RunEvent, RunResponse } from "../types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type RunPanelProps = {
  runId?: string | null;
  run?: RunResponse | null;
  events: RunEvent[];
  artifacts: ArtifactRecord[];
  loading: boolean;
  error?: string | null;
  onRefresh: () => Promise<void>;
  artifactDownloadUrl: (path: string) => string;
};

export function RunPanel({
  runId,
  run,
  events,
  artifacts,
  loading,
  error,
  onRefresh,
  artifactDownloadUrl,
}: RunPanelProps) {
  return (
    <Card className="border-white/60 bg-white/82 shadow-xl backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-3">
        <CardTitle>Run details</CardTitle>
        <div className="row gap-xs">
          {runId ? <Badge variant="secondary">{shortId(runId)}</Badge> : null}
          <Button variant="outline" size="sm" onClick={() => void onRefresh()}>
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {!runId ? (
          <p className="text-muted-foreground text-sm">No run yet. Submit a message first.</p>
        ) : null}
        {error ? <p className="error-text">{error}</p> : null}
        {run ? (
          <div className="flex flex-col gap-2">
            <div className="kv-grid">
              <span>Status</span>
              <strong>{run.status}</strong>
              <span>Goal</span>
              <strong>{run.goal || "-"}</strong>
              <span>Updated</span>
              <strong>{toLocalDateTime(run.updated_at)}</strong>
            </div>
            {run.error ? <p className="error-text">{run.error}</p> : null}
          </div>
        ) : null}
        {events.length > 0 ? (
          <details open>
            <summary>Events ({events.length})</summary>
            <ul className="list">
              {events.map((event, index) => (
                <li key={`${event.event_type}-${event.ts ?? index}`} className="event-item">
                  <strong>{event.event_type}</strong>
                  <span>{event.ts ? toLocalDateTime(event.ts) : ""}</span>
                  <code>{JSON.stringify(event.payload ?? {}, null, 0)}</code>
                </li>
              ))}
            </ul>
          </details>
        ) : null}
        {artifacts.length > 0 ? (
          <details open>
            <summary>Artifacts ({artifacts.length})</summary>
            <ul className="list">
              {artifacts.map((artifact) => (
                <li key={artifact.path} className="artifact-item">
                  <a href={artifactDownloadUrl(artifact.path)} target="_blank" rel="noreferrer">
                    {artifact.path}
                  </a>
                  <span>{formatBytes(artifact.size_bytes)}</span>
                  <span>{artifact.mime_type ?? "file"}</span>
                </li>
              ))}
            </ul>
          </details>
        ) : null}
      </CardContent>
    </Card>
  );
}
