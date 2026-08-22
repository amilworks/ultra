import { useCallback, useEffect, useMemo, useState } from "react";
import { Download, Film, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type {
  ApiClient,
  UploadVideoExportRequest,
  UploadVideoExportResponse,
} from "@/lib/api";

import { downloadBlob, exportFileStem } from "./captureView";

const PREVIEW_FRAME_LIMIT = 240;
const COMPLETE_FRAME_LIMIT = 1200;
const VIDEO_FPS = 24;

type VolumeVideoExportDialogProps = {
  apiClient: ApiClient;
  fileId: string;
  originalName: string;
  zCount: number;
  tCount: number;
  currentZ: number;
  currentT: number;
  channels: number[];
  channelColors: string[];
  strictScalarSlice: boolean;
  enhancement?: string;
  negative?: boolean;
  scalarRenderMode: "intensity" | "mask";
  scalarThresholdValue?: number;
  portalContainer?: HTMLElement | null;
};

const frameCountFor = (
  sourceFrames: number,
  profile: "preview" | "complete"
): number =>
  profile === "preview"
    ? Math.min(sourceFrames, PREVIEW_FRAME_LIMIT)
    : sourceFrames;

const durationLabel = (frames: number): string => {
  const seconds = frames / VIDEO_FPS;
  return seconds < 10 ? `${seconds.toFixed(1)} s` : `${Math.round(seconds)} s`;
};

const exportStatusLabel = (status: UploadVideoExportResponse): string => {
  if (status.status === "queued") {
    return "Waiting for the image renderer…";
  }
  if (status.status === "progress") {
    return `Rendering ${status.frames_completed} of ${status.frames_total} frames…`;
  }
  if (status.status === "ready") {
    return "MP4 ready to download.";
  }
  return "The video could not be completed. You can retry this export.";
};

export function VolumeVideoExportDialog({
  apiClient,
  fileId,
  originalName,
  zCount,
  tCount,
  currentZ,
  currentT,
  channels,
  channelColors,
  strictScalarSlice,
  enhancement,
  negative,
  scalarRenderMode,
  scalarThresholdValue,
  portalContainer,
}: VolumeVideoExportDialogProps) {
  const initialMode = zCount > 1 ? "z_sweep" : "time_series";
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"z_sweep" | "time_series">(
    initialMode
  );
  const [profile, setProfile] = useState<"preview" | "complete">("preview");
  const [statusState, setStatusState] = useState<{
    recipeKey: string;
    value: UploadVideoExportResponse;
  } | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [errorState, setErrorState] = useState<{
    recipeKey: string;
    message: string;
  } | null>(null);

  const sourceFrames = mode === "z_sweep" ? zCount : tCount;
  const completeAllowed = sourceFrames <= COMPLETE_FRAME_LIMIT;
  const effectiveProfile = profile === "complete" && !completeAllowed ? "preview" : profile;
  const outputFrames = frameCountFor(sourceFrames, effectiveProfile);
  const selectedColors = useMemo(
    () => channels.map((channel) => channelColors[channel]).filter(Boolean),
    [channelColors, channels]
  );
  const recipeKey = [
    fileId,
    mode,
    effectiveProfile,
    zCount,
    tCount,
    currentZ,
    currentT,
    channels.join(","),
    selectedColors.join(","),
    strictScalarSlice ? "scalar" : "composite",
    enhancement ?? "",
    negative ? "negative" : "positive",
    scalarRenderMode,
    scalarThresholdValue ?? "",
  ].join("\u0000");
  const status = statusState?.recipeKey === recipeKey ? statusState.value : null;
  const error = errorState?.recipeKey === recipeKey ? errorState.message : "";
  const setCurrentStatus = useCallback((value: UploadVideoExportResponse) => {
    setStatusState({ recipeKey, value });
  }, [recipeKey]);
  const setCurrentError = useCallback((message: string) => {
    setErrorState({ recipeKey, message });
  }, [recipeKey]);

  useEffect(() => {
    if (!open || !status || !["queued", "progress"].includes(status.status)) {
      return;
    }
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      try {
        const next = await apiClient.getUploadVideoExport(
          fileId,
          status.render_id,
          { signal: controller.signal }
        );
        setCurrentStatus(next);
        setCurrentError("");
      } catch (pollError) {
        if (!controller.signal.aborted) {
          setCurrentError(
            pollError instanceof Error
              ? pollError.message
              : "Video status is temporarily unavailable."
          );
          timer = window.setTimeout(poll, 2500);
        }
      }
    };
    timer = window.setTimeout(poll, 1500);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [apiClient, fileId, open, setCurrentError, setCurrentStatus, status]);

  const startExport = async () => {
    setSubmitting(true);
    setCurrentError("");
    const request: UploadVideoExportRequest = {
      mode,
      profile: effectiveProfile,
      fixed_z: Math.max(0, Math.min(zCount - 1, currentZ)),
      fixed_t: Math.max(0, Math.min(tCount - 1, currentT)),
      channels,
      channel_colors: strictScalarSlice ? undefined : selectedColors,
      enhancement: strictScalarSlice ? enhancement : undefined,
      negative: strictScalarSlice ? Boolean(negative) : undefined,
      scalar_render_mode: scalarRenderMode,
      scalar_threshold_value:
        scalarRenderMode === "mask" ? scalarThresholdValue : undefined,
      scalar_threshold_foreground:
        scalarRenderMode === "mask" ? "above" : undefined,
    };
    try {
      setCurrentStatus(await apiClient.createUploadVideoExport(fileId, request));
    } catch (requestError) {
      setCurrentError(
        requestError instanceof Error
          ? requestError.message
          : "Video export could not be started."
      );
    } finally {
      setSubmitting(false);
    }
  };

  const downloadExport = async () => {
    if (!status || status.status !== "ready") {
      return;
    }
    setDownloading(true);
    setCurrentError("");
    try {
      const blob = await apiClient.downloadUploadVideoExport(
        fileId,
        status.render_id
      );
      const suffix = status.mode === "z_sweep" ? "z-sweep" : "time-series";
      downloadBlob(blob, status.filename || `${exportFileStem(originalName)}-${suffix}.mp4`);
    } catch (downloadError) {
      setCurrentError(
        downloadError instanceof Error
          ? downloadError.message
          : "The MP4 could not be downloaded."
      );
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen);
        if (nextOpen) {
          setCurrentError("");
        }
      }}
    >
      <DialogTrigger asChild>
        <Button type="button" variant="outline" size="sm">
          <Film data-icon="inline-start" />
          Export video
        </Button>
      </DialogTrigger>
      <DialogContent
        portalContainer={portalContainer}
        className="viewer-video-export-dialog"
      >
        <DialogHeader className="viewer-video-export-head">
          <DialogTitle>Export video</DialogTitle>
          <DialogDescription>
            Create an MP4 from the calibrated planes shown in Lens.
          </DialogDescription>
        </DialogHeader>

        <div className="viewer-video-export-body">
          <section className="viewer-video-export-section" aria-labelledby="video-motion-label">
            <span id="video-motion-label" className="viewer-video-export-label">
              Sequence
            </span>
            {zCount > 1 && tCount > 1 ? (
              <div className="viewer-video-export-options">
                <button
                  type="button"
                  className="viewer-video-export-option"
                  data-selected={mode === "z_sweep"}
                  aria-pressed={mode === "z_sweep"}
                  onClick={() => {
                    setMode("z_sweep");
                    if (zCount > COMPLETE_FRAME_LIMIT) {
                      setProfile("preview");
                    }
                  }}
                >
                  <strong>Z sweep</strong>
                  <span>At time {currentT + 1}</span>
                </button>
                <button
                  type="button"
                  className="viewer-video-export-option"
                  data-selected={mode === "time_series"}
                  aria-pressed={mode === "time_series"}
                  onClick={() => {
                    setMode("time_series");
                    if (tCount > COMPLETE_FRAME_LIMIT) {
                      setProfile("preview");
                    }
                  }}
                >
                  <strong>Time series</strong>
                  <span>At Z {currentZ + 1}</span>
                </button>
              </div>
            ) : (
              <div className="viewer-video-export-sequence">
                <span>
                  <strong>{mode === "z_sweep" ? "Z sweep" : "Time series"}</strong>
                  <small>
                    {mode === "z_sweep"
                      ? `Move through depth at time ${currentT + 1}`
                      : `Play time at Z slice ${currentZ + 1}`}
                  </small>
                </span>
                <span>{sourceFrames} planes</span>
              </div>
            )}
          </section>

          <section className="viewer-video-export-section" aria-labelledby="video-detail-label">
            <span id="video-detail-label" className="viewer-video-export-label">
              Frame range
            </span>
            <div className="viewer-video-export-options">
              <button
                type="button"
                className="viewer-video-export-option"
                data-selected={effectiveProfile === "preview"}
                aria-pressed={effectiveProfile === "preview"}
                onClick={() => setProfile("preview")}
              >
                <strong>Preview</strong>
                <span>{Math.min(sourceFrames, PREVIEW_FRAME_LIMIT)} frames</span>
              </button>
              <button
                type="button"
                className="viewer-video-export-option"
                data-selected={effectiveProfile === "complete"}
                aria-pressed={effectiveProfile === "complete"}
                disabled={!completeAllowed}
                title={
                  completeAllowed
                    ? "Include every source frame"
                    : `Complete exports are limited to ${COMPLETE_FRAME_LIMIT} frames`
                }
                onClick={() => setProfile("complete")}
              >
                <strong>Complete</strong>
                <span>
                  {completeAllowed ? `${sourceFrames} frames` : `Limit ${COMPLETE_FRAME_LIMIT}`}
                </span>
              </button>
            </div>
            <p className="viewer-video-export-help">
              {effectiveProfile === "preview" && outputFrames < sourceFrames
                ? "Uniform sampling includes the first and last source plane."
                : "Every source plane will be included."}
            </p>
          </section>

          <dl className="viewer-video-export-summary" aria-live="polite">
            <div>
              <dt>Frames</dt>
              <dd>{outputFrames} of {sourceFrames}</dd>
            </div>
            <div>
              <dt>Duration</dt>
              <dd>{durationLabel(outputFrames)}</dd>
            </div>
            <div>
              <dt>Rate</dt>
              <dd>{VIDEO_FPS} fps</dd>
            </div>
          </dl>

          {status ? (
            <div className="viewer-video-export-status" data-status={status.status}>
              {["queued", "progress"].includes(status.status) ? (
                <LoaderCircle className="viewer-video-export-spinner" aria-hidden="true" />
              ) : null}
              <span>{exportStatusLabel(status)}</span>
              {status.status === "progress" ? (
                <progress
                  aria-label="Video rendering progress"
                  max={status.frames_total}
                  value={status.frames_completed}
                />
              ) : null}
            </div>
          ) : null}
          {error ? <p className="viewer-video-export-error">{error}</p> : null}
        </div>

        <DialogFooter className="viewer-video-export-footer">
          <p className="viewer-video-export-note">
            <strong>Presentation copy</strong>
            <span>H.264 compression changes pixel values. Source data stays unchanged.</span>
          </p>
          {status?.status === "ready" ? (
            <Button type="button" onClick={downloadExport} disabled={downloading}>
              {downloading ? (
                <LoaderCircle className="viewer-video-export-spinner" aria-hidden="true" />
              ) : (
                <Download data-icon="inline-start" />
              )}
              Download MP4
            </Button>
          ) : (
            <Button
              type="button"
              onClick={startExport}
              disabled={submitting || ["queued", "progress"].includes(status?.status ?? "")}
            >
              {submitting ? (
                <LoaderCircle className="viewer-video-export-spinner" aria-hidden="true" />
              ) : (
                <Film data-icon="inline-start" />
              )}
              Create MP4
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
