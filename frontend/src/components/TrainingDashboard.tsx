// The GoldGate answer page (plan section 14): one calm column that answers,
// in order, "is the model OK?" (status line), "what changed?" (digest), and
// "is anything waiting on me?" (the morphing focal card - host of the page's
// ONLY filled primary), with the evidence demoted to four disclosures. The
// file path, export name, lazy loader, and 5-prop App.tsx contract are
// unchanged from the tabbed dashboard this replaces.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { ApiError, type ApiClient } from "@/lib/api";
import {
  loadTrainingSnapshot,
  type TrainingModelSnapshot,
} from "@/features/training/client";
import {
  LAST_VISIT_STORAGE_PREFIX,
  deriveDigest,
  deriveOverlay,
  deriveTrainingPhase,
  hasCapability,
  type TrainingPhaseState,
} from "@/features/training/state";
import type { ResourceRecord, TrainingJobRecord, TrainingVersionPromoteRequest } from "../types";
import { FocalCard } from "./training/FocalCard";
import { HowItWorks } from "./training/HowItWorks";
import { ModelSwitcher } from "./training/ModelSwitcher";
import { TrainingDialogs, type TrainingDialogKind } from "./training/RollbackControl";
import { SinceDigest } from "./training/SinceDigest";
import { StatusLine } from "./training/StatusLine";
import { TrainingStatTiles } from "./training/TrainingStatTiles";
import { VersionHistory } from "./training/VersionHistory";
import { formatCount, formatTimestamp } from "./training/format";

type TrainingDashboardProps = {
  apiClient: ApiClient;
  resources: ResourceRecord[];
  resourcesLoading: boolean;
  resourcesError: string | null;
  isAdmin: boolean;
};

const DEFAULT_MODEL_KEY = "yolov5_rarespot";
const POLL_INTERVAL_MS = 5000;
const POLLING_PHASES = new Set(["gold-freezing", "retrain-running", "benchmark-running", "canary-soaking"]);

const normalizeError = (error: unknown): string => {
  if (error instanceof ApiError) {
    if (typeof error.detail === "string" && error.detail.trim().length > 0) {
      return error.detail.trim();
    }
    if (error.detail && typeof error.detail === "object") {
      const message = String((error.detail as Record<string, unknown>).message ?? "").trim();
      if (message) {
        return message;
      }
      const detail = String((error.detail as Record<string, unknown>).detail ?? "").trim();
      if (detail) {
        return detail;
      }
      const errorText = String((error.detail as Record<string, unknown>).error ?? "").trim();
      if (errorText) {
        return errorText;
      }
    }
    return `Request failed with status ${error.status}`;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  return "Request failed.";
};

function InferenceTryout({
  apiClient,
  modelKey,
  resources,
  resourcesLoading,
  resourcesError,
}: {
  apiClient: ApiClient;
  modelKey: string;
  resources: ResourceRecord[];
  resourcesLoading: boolean;
  resourcesError: string | null;
}) {
  const [selection, setSelection] = useState<Record<string, boolean>>({});
  const [job, setJob] = useState<TrainingJobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selected = useMemo(
    () => Object.entries(selection).filter(([, on]) => on).map(([id]) => id),
    [selection]
  );

  const run = async () => {
    if (selected.length === 0) {
      setError("Select at least one image.");
      return;
    }
    try {
      setError(null);
      const payload = await apiClient.createInferenceJob({
        model_key: modelKey,
        file_ids: selected,
        confirm_launch: true,
      });
      setJob(payload.job);
    } catch (cause) {
      setError(normalizeError(cause));
    }
  };

  return (
    <div style={{ display: "grid", gap: "0.6rem" }}>
      {resourcesLoading ? (
        <p className="training-gloss">Loading resources…</p>
      ) : resourcesError ? (
        <p className="training-inline-error">{resourcesError}</p>
      ) : resources.length === 0 ? (
        <p className="training-gloss">No images uploaded yet.</p>
      ) : (
        <div style={{ maxHeight: 280, overflowY: "auto", display: "grid", gap: "0.35rem" }}>
          {resources.map((resource) => (
            <label key={resource.file_id} style={{ display: "flex", alignItems: "center", gap: "0.5rem", minHeight: 44 }}>
              <Checkbox
                checked={Boolean(selection[resource.file_id])}
                onCheckedChange={(checked) =>
                  setSelection((previous) => ({ ...previous, [resource.file_id]: checked === true }))
                }
              />
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {resource.original_name}
              </span>
            </label>
          ))}
        </div>
      )}
      <div className="training-actions">
        <Button variant="outline" onClick={() => void run()}>
          Run inference ({selected.length})
        </Button>
      </div>
      {error ? <p className="training-inline-error">{error}</p> : null}
      {job ? (
        <p className="training-gloss">
          Job {job.job_id} — {job.status}
          {job.result?.prediction_count != null ? ` · ${formatCount(Number(job.result.prediction_count))} predictions` : ""}
        </p>
      ) : null}
    </div>
  );
}

export function TrainingDashboard({
  apiClient,
  resources,
  resourcesLoading,
  resourcesError,
  isAdmin,
}: TrainingDashboardProps) {
  void isAdmin; // accepted per the App.tsx contract; the page renders identically
  const [modelKey, setModelKey] = useState(DEFAULT_MODEL_KEY);
  const [snapshot, setSnapshot] = useState<TrainingModelSnapshot | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const [transientNote, setTransientNote] = useState<string | null>(null);
  const [dialog, setDialog] = useState<TrainingDialogKind>(null);
  const historyRef = useRef<HTMLDetailsElement | null>(null);
  const focalRef = useRef<HTMLDivElement | null>(null);
  // Read once per model at render time (pure), written back on unmount.
  const lastVisit = useMemo(
    () => window.localStorage.getItem(LAST_VISIT_STORAGE_PREFIX + modelKey),
    [modelKey]
  );

  const refresh = useCallback(async () => {
    try {
      const next = await loadTrainingSnapshot(apiClient, modelKey);
      setSnapshot(next);
      setErrorMessage(null);
    } catch (cause) {
      setErrorMessage(normalizeError(cause));
    }
  }, [apiClient, modelKey]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refresh();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  useEffect(() => {
    const storageKey = LAST_VISIT_STORAGE_PREFIX + modelKey;
    return () => {
      window.localStorage.setItem(storageKey, new Date().toISOString());
    };
  }, [modelKey]);

  const state: TrainingPhaseState | null = useMemo(
    () => (snapshot ? deriveTrainingPhase(snapshot) : null),
    [snapshot]
  );
  const overlay = snapshot ? deriveOverlay(snapshot.model) : null;

  // Polling: 5s re-arm only in the in-flight phases + canary soak; a
  // visibility flip refetches once at rest (a second operator may have acted).
  useEffect(() => {
    if (!state || !POLLING_PHASES.has(state.phase)) {
      return;
    }
    const timer = window.setTimeout(() => {
      if (document.visibilityState === "visible") {
        void refresh();
      }
    }, POLL_INTERVAL_MS);
    return () => window.clearTimeout(timer);
  }, [state, snapshot, refresh]);

  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        void refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [refresh]);

  const act = useCallback(
    async (label: string, action: () => Promise<unknown>, successNote?: string) => {
      try {
        setBusyMessage(label);
        setErrorMessage(null);
        await action();
        if (successNote) {
          setTransientNote(successNote);
          window.setTimeout(() => setTransientNote(null), 4000);
        }
        await refresh();
        focalRef.current?.focus();
      } catch (cause) {
        setErrorMessage(normalizeError(cause));
      } finally {
        setBusyMessage(null);
      }
    },
    [refresh]
  );

  if (!snapshot && !errorMessage) {
    return (
      <section className="mx-auto flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="training-panel">
          <p className="training-gloss">Loading training status…</p>
        </div>
      </section>
    );
  }

  if (!snapshot || !state) {
    return (
      <section className="mx-auto flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        <div className="training-panel">
          <p className="training-inline-error">{errorMessage ?? "Training status is unavailable."}</p>
        </div>
      </section>
    );
  }

  const { status, versions, retrainRequests, models, model } = snapshot;
  const notConfigured = overlay === "not-configured";
  const canSync = !notConfigured && hasCapability(model, "SYNC");
  const digestEvents = deriveDigest(status.recent_events, lastVisit);
  const activeVersion = state.active;
  const retiredVersion = versions.find((row) => row.status === "retired") ?? null;
  const goldDraftSummary = status.gold
    ? `${formatCount(status.gold.per_slice_counts?.prior_train ?? status.gold.qualifying_prior_frames ?? 0)} prior-survey frames · ${formatCount(status.gold.per_slice_counts?.held_out_test ?? 0)} held-out frames${(status.gold.per_slice_counts?.held_out_test ?? 0) === 0 ? " — held-out joins at the next post-checkpoint survey." : "."}`
    : "";

  const confirmDialog = (kind: Exclude<TrainingDialogKind, null>, reason?: string) => {
    if (kind === "rollback" && activeVersion) {
      void act("Rolling back…", () =>
        apiClient.rollbackTrainingModelVersion(activeVersion.version_id, {
          target_version_id: retiredVersion?.version_id ?? null,
        })
      );
    }
    if (kind === "override" && state.canary) {
      void act("Promoting with override…", () =>
        apiClient.promoteTrainingModelVersion(state.canary!.version_id, {
          override_reason: reason ?? "",
        } satisfies TrainingVersionPromoteRequest)
      );
    }
    if (kind === "dismiss" && state.candidate) {
      void act("Dismissing candidate…", () => apiClient.rejectModelVersion(state.candidate!.version_id, reason));
    }
    if (kind === "freeze") {
      void act("Submitting the gold freeze…", async () => {
        const draft = await apiClient.createGoldSetDraft(modelKey);
        const goldSetId = String(
          (draft as Record<string, unknown>).gold_set_id ?? status.gold?.gold_set_id ?? ""
        );
        await apiClient.freezeGoldSet(modelKey, goldSetId);
      });
    }
  };

  return (
    <section className="mx-auto flex-1 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="training-panel">
        <div className="training-header-row">
          <div className="training-header-title">
            <h2>Training</h2>
            <ModelSwitcher models={models} selected={modelKey} onSelect={setModelKey} />
          </div>
          {canSync ? (
            <Button
              variant="outline"
              onClick={() =>
                void act(
                  "Syncing reviewed annotations from BisQue…",
                  () => apiClient.syncTrainingModel(modelKey),
                  "Synced."
                )
              }
            >
              Sync now
            </Button>
          ) : null}
        </div>
        <p className="training-subtitle">Gold-gated continual finetuning — sync, retrain, promote, roll back.</p>

        {!model ? <p className="training-gloss">lineage not found for {modelKey}</p> : null}

        <StatusLine state={state} status={status} versions={versions} onRollback={() => setDialog("rollback")} />

        {notConfigured ? (
          <p className="training-gloss" style={{ marginTop: "0.9rem" }}>
            Training services aren't configured on this deployment. Data and history are read-only.
          </p>
        ) : null}

        <SinceDigest
          events={digestEvents}
          demoted={state.demoted}
          firstVisit={lastVisit == null}
          onOpenHistory={() => {
            if (historyRef.current) {
              historyRef.current.open = true;
              historyRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
            }
          }}
        />

        <div ref={focalRef} tabIndex={-1}>
          {notConfigured ? null : (
            <FocalCard
              state={state}
              status={status}
              onFreezeGold={() => setDialog("freeze")}
              onRetryFreeze={() => setDialog("freeze")}
              onRunBaseline={() =>
                void act("Running the baseline benchmark…", () =>
                  apiClient.runTrainingBenchmark(modelKey, { mode: "canonical_only" })
                )
              }
              onRequestRetrain={() =>
                void act("Submitting the retraining request…", () =>
                  apiClient.requestTrainingRetrain(modelKey, { confirm_launch: true })
                )
              }
              onRunBenchmark={() =>
                state.candidate
                  ? void act("Running the benchmark…", () =>
                      apiClient.runTrainingBenchmark(modelKey, {
                        mode: "promotion_packet",
                        version_id: state.candidate!.version_id,
                      })
                    )
                  : undefined
              }
              onPromoteCanary={() =>
                state.candidate
                  ? void act("Promoting to canary…", () =>
                      apiClient.promoteTrainingModelVersion(state.candidate!.version_id, {})
                    )
                  : undefined
              }
              onPromoteActive={() =>
                state.canary
                  ? void act("Promoting to active…", () =>
                      apiClient.promoteTrainingModelVersion(state.canary!.version_id, {})
                    )
                  : undefined
              }
              onOverridePromote={() => setDialog("override")}
              onDismissCandidate={() => setDialog("dismiss")}
            />
          )}
        </div>

        {busyMessage ? <p className="training-inline-busy">{busyMessage}</p> : null}
        {transientNote ? <p className="training-inline-busy">{transientNote}</p> : null}
        {errorMessage ? <p className="training-inline-error">{errorMessage}</p> : null}

        <Separator className="mt-5" />

        <details className="training-disclosure">
          <summary>
            <span className="training-disclosure-chevron" aria-hidden="true">
              ›
            </span>
            Data — {formatCount(Number(status.reviewed_images ?? 0))} reviewed
            {status.last_sync_at ? ` · last sync ${formatTimestamp(status.last_sync_at)}` : ""}
          </summary>
          <div className="training-disclosure-body">
            <TrainingStatTiles status={status} />
          </div>
        </details>

        <details className="training-disclosure" ref={historyRef}>
          <summary>
            <span className="training-disclosure-chevron" aria-hidden="true">
              ›
            </span>
            History
            {retrainRequests.length > 0 ? ` — ${formatCount(retrainRequests.length)} retrain requests` : ""}
          </summary>
          <div className="training-disclosure-body">
            <VersionHistory status={status} versions={versions} retrainRequests={retrainRequests} />
          </div>
        </details>

        <details className="training-disclosure">
          <summary>
            <span className="training-disclosure-chevron" aria-hidden="true">
              ›
            </span>
            How this works
          </summary>
          <div className="training-disclosure-body">
            <HowItWorks />
          </div>
        </details>

        <details className="training-disclosure">
          <summary>
            <span className="training-disclosure-chevron" aria-hidden="true">
              ›
            </span>
            Try the model
          </summary>
          <div className="training-disclosure-body">
            <InferenceTryout
              apiClient={apiClient}
              modelKey={modelKey}
              resources={resources}
              resourcesLoading={resourcesLoading}
              resourcesError={resourcesError}
            />
          </div>
        </details>

        <TrainingDialogs
          kind={dialog}
          targetVersionId={
            dialog === "rollback"
              ? retiredVersion?.version_id ?? "the previous version"
              : state.candidate?.version_id ?? state.canary?.version_id ?? ""
          }
          activeVersionId={activeVersion?.version_id ?? ""}
          goldDraftSummary={goldDraftSummary}
          onConfirm={confirmDialog}
          onClose={() => setDialog(null)}
        />
      </div>
    </section>
  );
}
