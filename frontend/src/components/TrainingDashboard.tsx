import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ApiError, type ApiClient } from "@/lib/api";
import type {
  PrairieRetrainRecord,
  PrairieStatusResponse,
  ResourceRecord,
  TrainingJobRecord,
  TrainingLineageRecord,
  TrainingModelRecord,
  TrainingModelVersionRecord,
} from "../types";

type TrainingDashboardProps = {
  apiClient: ApiClient;
  resources: ResourceRecord[];
  resourcesLoading: boolean;
  resourcesError: string | null;
  isAdmin: boolean;
};

const PRAIRIE_MODEL_KEY = "yolov5_rarespot";

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

const statusBadgeClass = (status: string): string => {
  const normalized = status.toLowerCase();
  if (normalized === "healthy" || normalized === "succeeded" || normalized === "active") {
    return "border-emerald-500/45 bg-emerald-500/15 text-emerald-700 dark:border-emerald-400/45 dark:bg-emerald-500/20 dark:text-emerald-300";
  }
  if (
    normalized === "queued" ||
    normalized === "running" ||
    normalized === "watch" ||
    normalized === "canary"
  ) {
    return "border-amber-500/45 bg-amber-500/15 text-amber-700 dark:border-amber-400/45 dark:bg-amber-500/20 dark:text-amber-300";
  }
  if (
    normalized === "retrain recommended" ||
    normalized === "needs human review" ||
    normalized === "failed" ||
    normalized === "canceled"
  ) {
    return "border-destructive/45 bg-destructive/10 text-destructive";
  }
  return "border-border bg-muted/60 text-muted-foreground";
};

const formatTs = (value: string | null | undefined): string => {
  const token = String(value || "").trim();
  if (!token) {
    return "N/A";
  }
  const date = new Date(token);
  if (Number.isNaN(date.getTime())) {
    return token;
  }
  return date.toLocaleString();
};

const formatCount = (value: number): string =>
  new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(
    Number.isFinite(value) ? value : 0
  );

const firstGatingReason = (row: PrairieRetrainRecord): string | null => {
  const summary = row.gating_summary;
  if (!summary || typeof summary !== "object") {
    return null;
  }
  const reasons = (summary as Record<string, unknown>).reasons;
  if (!Array.isArray(reasons) || reasons.length <= 0) {
    return null;
  }
  const first = String(reasons[0] ?? "").trim();
  return first || null;
};

export function TrainingDashboard({
  apiClient,
  resources,
  resourcesLoading,
  resourcesError,
  isAdmin,
}: TrainingDashboardProps) {
  const [models, setModels] = useState<TrainingModelRecord[]>([]);
  const [status, setStatus] = useState<PrairieStatusResponse | null>(null);
  const [retrainRequests, setRetrainRequests] = useState<PrairieRetrainRecord[]>([]);
  const [lineage, setLineage] = useState<TrainingLineageRecord | null>(null);
  const [versions, setVersions] = useState<TrainingModelVersionRecord[]>([]);
  const [busyMessage, setBusyMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [syncErrors, setSyncErrors] = useState<string[]>([]);
  const [selectedInferenceModel, setSelectedInferenceModel] = useState<string>(PRAIRIE_MODEL_KEY);
  const [inferenceSelection, setInferenceSelection] = useState<Record<string, boolean>>({});
  const [inferenceJob, setInferenceJob] = useState<TrainingJobRecord | null>(null);
  const [retrainNote, setRetrainNote] = useState("");

  const activeVersion = useMemo(
    () =>
      versions.find((row) => row.status === "active") ??
      versions.find((row) => row.version_id === status?.active_model_version) ??
      null,
    [versions, status?.active_model_version]
  );
  const candidateVersion = useMemo(
    () =>
      versions.find((row) => row.status === "canary") ??
      versions.find((row) => row.status === "candidate") ??
      null,
    [versions]
  );
  const selectedInferenceFileIds = useMemo(
    () =>
      Object.entries(inferenceSelection)
        .filter(([, selected]) => selected)
        .map(([fileId]) => fileId),
    [inferenceSelection]
  );
  const activeRetrain = useMemo(
    () => retrainRequests.find((row) => ["queued", "running", "paused"].includes(row.status)) ?? null,
    [retrainRequests]
  );
  const candidateBenchmarkReady = useMemo(
    () =>
      Boolean(
        candidateVersion?.metrics?.promotion_benchmark_ready ??
          candidateVersion?.metrics?.benchmark_ready
      ),
    [candidateVersion]
  );
  const candidateGuardrailPassed = useMemo(() => {
    const metadata = candidateVersion?.metadata;
    if (!metadata || typeof metadata !== "object") {
      return false;
    }
    const guardrails = (metadata as Record<string, unknown>).guardrails;
    return Boolean(
      guardrails &&
        typeof guardrails === "object" &&
        (guardrails as Record<string, unknown>).passed === true
    );
  }, [candidateVersion]);
  const candidateGuardrailReasons = useMemo(() => {
    const metadata = candidateVersion?.metadata;
    if (!metadata || typeof metadata !== "object") {
      return [];
    }
    const guardrails = (metadata as Record<string, unknown>).guardrails;
    if (!guardrails || typeof guardrails !== "object") {
      return [];
    }
    const reasons = (guardrails as Record<string, unknown>).reasons;
    if (!Array.isArray(reasons)) {
      return [];
    }
    return reasons.map((row) => String(row)).filter((row) => row.trim().length > 0);
  }, [candidateVersion]);
  const retrainGateReady = Boolean(status?.retrain_gate);
  const retrainGateReasons = useMemo(
    () =>
      Array.isArray(status?.retrain_gate_reasons)
        ? status.retrain_gate_reasons.map((row) => String(row)).filter((row) => row.trim().length > 0)
        : [],
    [status?.retrain_gate_reasons]
  );
  const retrainGateCounts = useMemo(
    () =>
      status?.retrain_gate_counts && typeof status.retrain_gate_counts === "object"
        ? status.retrain_gate_counts
        : {},
    [status?.retrain_gate_counts]
  );
  const canonicalBenchmarkReady = Boolean(status?.canonical_benchmark_ready);
  const promotionBenchmarkReady = Boolean(status?.promotion_benchmark_ready ?? status?.benchmark_ready);
  const syncSummary = useMemo(() => {
    const reviewedImages = Number(status?.reviewed_images ?? 0);
    const unreviewedImages = Number(status?.unreviewed_images ?? 0);
    const totalImages = reviewedImages + unreviewedImages;
    const reviewedCoverage = totalImages > 0 ? (reviewedImages / totalImages) * 100 : 0;
    return {
      reviewedImages,
      unreviewedImages,
      totalImages,
      reviewedCoverage,
      burrowBoxes: Number(status?.class_counts?.burrow ?? 0),
      prairieDogBoxes: Number(status?.class_counts?.prairie_dog ?? 0),
      unsupportedBoxes: Number(status?.unsupported_class_counts?.prairie_dog_in_burrow ?? 0),
    };
  }, [status]);

  const loadLineageAndVersions = useCallback(async (): Promise<void> => {
    const domains = await apiClient.listTrainingDomains(500);
    for (const domain of domains.domains) {
      const lineages = await apiClient.listDomainLineages(domain.domain_id, { limit: 500 });
      const found =
        lineages.lineages.find(
          (row) => row.model_key === PRAIRIE_MODEL_KEY && row.scope === "shared"
        ) ??
        lineages.lineages.find((row) => row.model_key === PRAIRIE_MODEL_KEY) ??
        null;
      if (!found) {
        continue;
      }
      setLineage(found);
      const versionsPayload = await apiClient.listLineageVersions(found.lineage_id, { limit: 200 });
      setVersions(versionsPayload.versions);
      return;
    }
    setLineage(null);
    setVersions([]);
  }, [apiClient]);

  const loadDashboard = useCallback(async (): Promise<void> => {
    const [modelsPayload, statusPayload, retrainPayload] = await Promise.all([
      apiClient.listTrainingModels(),
      apiClient.getPrairieActiveLearningStatus(),
      apiClient.listPrairieRetrainRequests(),
    ]);
    setModels(modelsPayload.models);
    setStatus(statusPayload);
    setRetrainRequests(retrainPayload.requests);
    setSelectedInferenceModel((previous) => {
      const candidates = modelsPayload.models.filter((row) => row.supports_inference);
      if (candidates.some((row) => row.key === previous)) {
        return previous;
      }
      return candidates[0]?.key ?? PRAIRIE_MODEL_KEY;
    });
    await loadLineageAndVersions();
  }, [apiClient, loadLineageAndVersions]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        setBusyMessage("Loading prairie active-learning dashboard...");
        await loadDashboard();
        if (!cancelled) {
          setErrorMessage(null);
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(normalizeError(error));
        }
      } finally {
        if (!cancelled) {
          setBusyMessage(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadDashboard]);

  useEffect(() => {
    if (!activeRetrain && !inferenceJob) {
      return;
    }
    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          if (activeRetrain) {
            const retrainPayload = await apiClient.listPrairieRetrainRequests();
            setRetrainRequests(retrainPayload.requests);
            await loadLineageAndVersions();
            const statusPayload = await apiClient.getPrairieActiveLearningStatus();
            setStatus(statusPayload);
          }
          if (inferenceJob && ["queued", "running", "paused"].includes(inferenceJob.status)) {
            const updated = await apiClient.getInferenceJobResult(inferenceJob.job_id);
            setInferenceJob(updated.job);
          }
        } catch (error) {
          setErrorMessage(normalizeError(error));
        }
      })();
    }, 1300);
    return () => window.clearTimeout(timer);
  }, [activeRetrain, apiClient, inferenceJob, loadLineageAndVersions]);

  const handleSyncNow = async (): Promise<void> => {
    try {
      setBusyMessage("Syncing Prairie_Dog_Active_Learning from BisQue...");
      const payload = await apiClient.syncPrairieActiveLearningDataset();
      setSyncErrors(payload.errors);
      await loadDashboard();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(normalizeError(error));
    } finally {
      setBusyMessage(null);
    }
  };

  const handleRequestRetrain = async (): Promise<void> => {
    try {
      setBusyMessage("Submitting retraining request...");
      await apiClient.requestPrairieRetrain({
        confirm_launch: true,
        note: retrainNote.trim() || undefined,
      });
      setRetrainNote("");
      await loadDashboard();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(normalizeError(error));
    } finally {
      setBusyMessage(null);
    }
  };

  const handleRunBenchmark = async (
    mode: "canonical_only" | "promotion_packet"
  ): Promise<void> => {
    try {
      setBusyMessage(
        mode === "promotion_packet"
          ? "Refreshing promotion benchmark packet..."
          : "Running canonical benchmark on active model..."
      );
      await apiClient.runPrairieBenchmark({ mode });
      await loadDashboard();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(normalizeError(error));
    } finally {
      setBusyMessage(null);
    }
  };

  const handlePromoteCandidate = async (): Promise<void> => {
    if (!candidateVersion) {
      return;
    }
    try {
      setBusyMessage("Promoting candidate model version...");
      await apiClient.promoteTrainingModelVersion(candidateVersion.version_id, {});
      await loadDashboard();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(normalizeError(error));
    } finally {
      setBusyMessage(null);
    }
  };

  const handleRollback = async (): Promise<void> => {
    if (!activeVersion) {
      return;
    }
    try {
      setBusyMessage("Rolling back active model version...");
      await apiClient.rollbackTrainingModelVersion(activeVersion.version_id, {});
      await loadDashboard();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(normalizeError(error));
    } finally {
      setBusyMessage(null);
    }
  };

  const handleRunInference = async (): Promise<void> => {
    if (selectedInferenceFileIds.length === 0) {
      setErrorMessage("Select at least one image for inference.");
      return;
    }
    try {
      setBusyMessage("Launching inference run...");
      const payload = await apiClient.createInferenceJob({
        model_key: selectedInferenceModel,
        model_version:
          selectedInferenceModel === PRAIRIE_MODEL_KEY
            ? status?.active_model_version ?? activeVersion?.version_id ?? undefined
            : undefined,
        file_ids: selectedInferenceFileIds,
        confirm_launch: true,
      });
      setInferenceJob(payload.job);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(normalizeError(error));
    } finally {
      setBusyMessage(null);
    }
  };

  const lastSucceededRetrain = retrainRequests.find((row) => row.status === "succeeded") ?? null;
  const selectedModel = models.find((row) => row.key === selectedInferenceModel) ?? null;
  const latestSyncErrors = syncErrors.length > 0 ? syncErrors : [];

  return (
    <section className="mx-auto flex-1 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="mx-auto flex max-w-7xl flex-col gap-4">
        {errorMessage ? (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {errorMessage}
          </div>
        ) : null}
        {busyMessage ? (
          <div className="rounded-md border border-border bg-muted/60 px-3 py-2 text-sm text-muted-foreground">
            {busyMessage}
          </div>
        ) : null}

        <Tabs defaultValue="overview">
          <TabsList className="grid w-full grid-cols-2 gap-1 md:grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="data-sync">Data Sync</TabsTrigger>
            <TabsTrigger value="retraining">Retraining</TabsTrigger>
            <TabsTrigger value="inference-results">Inference Results</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Model Health</CardTitle>
                  <CardDescription>Current prairie model status</CardDescription>
                </CardHeader>
                <CardContent>
                  <Badge variant="outline" className={statusBadgeClass(String(status?.model_health || ""))}>
                    {status?.model_health ?? "Needs Human Review"}
                  </Badge>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Active Version</CardTitle>
                  <CardDescription>Shared YOLO lineage pointer</CardDescription>
                </CardHeader>
                <CardContent className="text-sm">{status?.active_model_version ?? "N/A"}</CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Last Retrain</CardTitle>
                  <CardDescription>Most recent completed retrain</CardDescription>
                </CardHeader>
                <CardContent className="text-sm">
                  {formatTs(lastSucceededRetrain?.finished_at ?? null)}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Dataset</CardTitle>
                  <CardDescription>Reviewed vs unreviewed images</CardDescription>
                </CardHeader>
                <CardContent className="text-sm">
                  {status?.reviewed_images ?? 0} reviewed / {status?.unreviewed_images ?? 0} unreviewed
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Burrow Labels</CardTitle>
                  <CardDescription>Reviewed gt2 annotation count</CardDescription>
                </CardHeader>
                <CardContent className="text-sm">
                  {syncSummary.burrowBoxes}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Prairie Dog Labels</CardTitle>
                  <CardDescription>Reviewed gt2 annotation count</CardDescription>
                </CardHeader>
                <CardContent className="text-sm">
                  {syncSummary.prairieDogBoxes}
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Unsupported Observations</CardTitle>
                  <CardDescription>Unsupported gt2 telemetry only</CardDescription>
                </CardHeader>
                <CardContent className="text-sm">{syncSummary.unsupportedBoxes}</CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Sync Window</CardTitle>
                  <CardDescription>Last and next scheduled sync</CardDescription>
                </CardHeader>
                <CardContent className="space-y-1 text-sm">
                  <p>Last: {formatTs(status?.last_sync_at)}</p>
                  <p>Next: {formatTs(status?.next_sync_at)}</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Benchmark Readiness</CardTitle>
                  <CardDescription>Canonical + active holdout packet</CardDescription>
                </CardHeader>
                <CardContent className="space-y-1 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge
                      variant="outline"
                      className={
                        canonicalBenchmarkReady ? statusBadgeClass("healthy") : statusBadgeClass("failed")
                      }
                    >
                      {canonicalBenchmarkReady ? "Canonical Ready" : "Canonical Missing"}
                    </Badge>
                    <Badge
                      variant="outline"
                      className={
                        promotionBenchmarkReady ? statusBadgeClass("healthy") : statusBadgeClass("failed")
                      }
                    >
                      {promotionBenchmarkReady ? "Promotion Ready" : "Promotion Missing"}
                    </Badge>
                  </div>
                  <p>Last benchmark: {formatTs(status?.last_benchmark_at)}</p>
                </CardContent>
              </Card>
            </div>
            {!isAdmin ? null : (
              <p className="mt-2 text-xs text-muted-foreground">
                Admin mode is enabled. Global health remains available in the Admin panel.
              </p>
            )}
          </TabsContent>

          <TabsContent value="data-sync">
            <div className="grid gap-4 xl:grid-cols-[1.1fr,1fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Managed Dataset Source</CardTitle>
                  <CardDescription>
                    Dataset is fixed to <span className="font-medium">Prairie_Dog_Active_Learning</span>.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="rounded-md border border-border bg-muted/40 p-3 text-sm">
                    <p>
                      <span className="font-medium">Dataset:</span> {status?.dataset_name ?? "Prairie_Dog_Active_Learning"}
                    </p>
                    <p>
                      <span className="font-medium">Dataset ID:</span> {status?.dataset_id ?? "not created"}
                    </p>
                    <p>
                      <span className="font-medium">Last sync:</span> {formatTs(status?.last_sync_at)}
                    </p>
                    <p>
                      <span className="font-medium">Next sync:</span> {formatTs(status?.next_sync_at)}
                    </p>
                  </div>
                  <Button onClick={() => void handleSyncNow()}>Sync now</Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Sync Summary</CardTitle>
                  <CardDescription>gt2-reviewed coverage and class telemetry</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="rounded-lg border border-border/70 bg-muted/20 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        Review coverage
                      </p>
                      <Badge variant="outline" className="bg-background/80">
                        {syncSummary.reviewedCoverage.toFixed(0)}%
                      </Badge>
                    </div>
                    <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-emerald-500/80"
                        style={{ width: `${Math.max(0, Math.min(100, syncSummary.reviewedCoverage))}%` }}
                      />
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <div className="rounded-md border border-border/60 bg-background/80 p-2">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Reviewed</p>
                        <p className="text-base font-semibold leading-tight">
                          {formatCount(syncSummary.reviewedImages)}
                        </p>
                      </div>
                      <div className="rounded-md border border-border/60 bg-background/80 p-2">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Unreviewed</p>
                        <p className="text-base font-semibold leading-tight">
                          {formatCount(syncSummary.unreviewedImages)}
                        </p>
                      </div>
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">
                      Total images: {formatCount(syncSummary.totalImages)}
                    </p>
                  </div>

                  <Separator />

                  <div className="space-y-2 rounded-lg border border-border/70 bg-muted/20 p-3">
                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      Class telemetry (gt2)
                    </p>
                    <div className="flex items-center justify-between rounded-md border border-border/60 bg-background/80 px-2 py-1.5 text-sm">
                      <span>burrow</span>
                      <Badge variant="secondary">{formatCount(syncSummary.burrowBoxes)}</Badge>
                    </div>
                    <div className="flex items-center justify-between rounded-md border border-border/60 bg-background/80 px-2 py-1.5 text-sm">
                      <span>prairie_dog</span>
                      <Badge variant="secondary">{formatCount(syncSummary.prairieDogBoxes)}</Badge>
                    </div>
                    <div className="flex items-center justify-between rounded-md border border-border/60 bg-background/80 px-2 py-1.5 text-sm">
                      <span className="text-muted-foreground">prairie_dog_in_burrow (unsupported)</span>
                      <Badge variant="outline" className="border-amber-500/45 bg-amber-500/10 text-amber-700 dark:border-amber-400/45 dark:bg-amber-500/20 dark:text-amber-300">
                        {formatCount(syncSummary.unsupportedBoxes)}
                      </Badge>
                    </div>
                  </div>

                  {latestSyncErrors.length > 0 ? (
                    <div className="rounded-lg border border-destructive/35 bg-destructive/10 p-3 text-xs text-destructive">
                      <div className="mb-2 flex items-center gap-1.5 font-medium">
                        <AlertTriangle className="size-3.5" />
                        Sync warnings
                      </div>
                      {latestSyncErrors.slice(0, 4).map((row, index) => (
                        <p key={`sync-err-${index}`}>{row}</p>
                      ))}
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="retraining">
            <div className="grid gap-4 xl:grid-cols-[1.05fr,1fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Request Retraining</CardTitle>
                  <CardDescription>
                    Fixed backend profile is used. No user hyperparameter controls in this release.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Label htmlFor="prairie-retrain-note">Request note (optional)</Label>
                  <Input
                    id="prairie-retrain-note"
                    value={retrainNote}
                    onChange={(event) => setRetrainNote(event.target.value)}
                    placeholder="Example: New windy-season camera batch uploaded."
                  />
                  <Button
                    onClick={() => void handleRequestRetrain()}
                    disabled={!retrainGateReady || Boolean(activeRetrain)}
                  >
                    Request Retraining
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => void handleRunBenchmark("canonical_only")}
                  >
                    Run Canonical Benchmark
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => void handleRunBenchmark("promotion_packet")}
                  >
                    Refresh Promotion Packet
                  </Button>
                  <div className="rounded-md border border-border bg-muted/30 p-3 text-sm">
                    <div className="flex items-center justify-between gap-2">
                      <span>Retrain gate</span>
                      <Badge variant="outline" className={statusBadgeClass(retrainGateReady ? "healthy" : "failed")}>
                        {retrainGateReady ? "Ready" : "Blocked"}
                      </Badge>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      reviewed={formatCount(Number(retrainGateCounts.reviewed_images ?? 0))} • objects=
                      {formatCount(Number(retrainGateCounts.total_objects ?? 0))}
                    </p>
                    {retrainGateReasons.length > 0 ? (
                      <p className="mt-2 text-xs text-muted-foreground">{retrainGateReasons[0]}</p>
                    ) : null}
                  </div>
                  <div className="rounded-md border border-border bg-muted/30 p-3 text-sm">
                    {activeRetrain ? (
                      <div className="flex items-center gap-2">
                        <span>Active request:</span>
                        <Badge variant="outline" className={statusBadgeClass(activeRetrain.status)}>
                          {activeRetrain.status}
                        </Badge>
                        <span className="truncate text-xs text-muted-foreground">
                          {activeRetrain.training_job_id}
                        </span>
                      </div>
                    ) : (
                      <span className="text-muted-foreground">No active retraining request.</span>
                    )}
                  </div>
                  <div className="max-h-[280px] overflow-y-auto rounded-md border border-border">
                    {retrainRequests.length === 0 ? (
                      <p className="px-3 py-2 text-sm text-muted-foreground">No retraining requests yet.</p>
                    ) : (
                      retrainRequests.map((row) => (
                        <div key={row.request_id} className="border-b border-border px-3 py-2 text-sm last:border-b-0">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className={statusBadgeClass(row.status)}>
                              {row.status}
                            </Badge>
                            <span className="truncate text-xs text-muted-foreground">{row.training_job_id}</span>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            created {formatTs(row.created_at)}{" "}
                            {row.finished_at ? `• finished ${formatTs(row.finished_at)}` : ""}
                          </p>
                          {row.error ? <p className="text-xs text-destructive">{row.error}</p> : null}
                          {firstGatingReason(row) ? (
                            <p className="text-xs text-muted-foreground">{firstGatingReason(row)}</p>
                          ) : null}
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Candidate vs Active</CardTitle>
                  <CardDescription>Promotion and rollback controls for the shared lineage</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <p>
                    <span className="font-medium">Lineage:</span> {lineage?.lineage_id ?? "N/A"}
                  </p>
                  <div className="rounded-md border border-border bg-muted/20 p-3">
                    <p className="font-medium">Active</p>
                    <p className="text-xs text-muted-foreground">{activeVersion?.version_id ?? "none"}</p>
                    <p className="text-xs text-muted-foreground">
                      mAP50: {String((activeVersion?.metrics?.map50 as number | undefined) ?? "N/A")}
                    </p>
                  </div>
                  <div className="rounded-md border border-border bg-muted/20 p-3">
                    <p className="font-medium">Candidate</p>
                    <p className="text-xs text-muted-foreground">{candidateVersion?.version_id ?? "none"}</p>
                    <p className="text-xs text-muted-foreground">
                      mAP50: {String((candidateVersion?.metrics?.map50 as number | undefined) ?? "N/A")}
                    </p>
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant="outline" className={candidateBenchmarkReady ? statusBadgeClass("healthy") : statusBadgeClass("failed")}>
                        {candidateBenchmarkReady ? "Benchmark Ready" : "Benchmark Missing"}
                      </Badge>
                      <Badge variant="outline" className={candidateGuardrailPassed ? statusBadgeClass("healthy") : statusBadgeClass("failed")}>
                        {candidateGuardrailPassed ? "Guardrails Pass" : "Guardrails Blocked"}
                      </Badge>
                    </div>
                    {candidateGuardrailReasons.length > 0 ? (
                      <p className="mt-2 text-xs text-muted-foreground">{candidateGuardrailReasons[0]}</p>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="default"
                      onClick={() => void handlePromoteCandidate()}
                      disabled={!candidateVersion || !candidateBenchmarkReady || !candidateGuardrailPassed}
                    >
                      Promote Candidate
                    </Button>
                    <Button variant="outline" onClick={() => void handleRollback()} disabled={!activeVersion}>
                      Rollback Active
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="inference-results">
            <div className="grid gap-4 xl:grid-cols-[1.2fr,1fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Quick Inference</CardTitle>
                  <CardDescription>Select images and run asynchronous inference.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-1">
                    <Label>Inference model</Label>
                    <Select value={selectedInferenceModel} onValueChange={setSelectedInferenceModel}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {models
                          .filter((row) => row.supports_inference)
                          .map((row) => (
                            <SelectItem key={row.key} value={row.key}>
                              {row.name}
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                    {selectedModel ? (
                      <p className="text-xs text-muted-foreground">
                        {selectedModel.description}
                        {!selectedModel.supports_training ? " Training not supported." : ""}
                      </p>
                    ) : null}
                  </div>
                  <Button onClick={() => void handleRunInference()}>
                    Run inference ({selectedInferenceFileIds.length})
                  </Button>
                  <div className="max-h-[320px] overflow-y-auto rounded-md border border-border">
                    {resourcesLoading ? (
                      <p className="px-3 py-2 text-sm text-muted-foreground">Loading resources...</p>
                    ) : resourcesError ? (
                      <p className="px-3 py-2 text-sm text-destructive">{resourcesError}</p>
                    ) : resources.length === 0 ? (
                      <p className="px-3 py-2 text-sm text-muted-foreground">No images uploaded yet.</p>
                    ) : (
                      resources.map((resource) => (
                        <div key={resource.file_id} className="border-b border-border px-3 py-2 text-sm last:border-b-0">
                          <label className="flex cursor-pointer items-center gap-2">
                            <Checkbox
                              checked={Boolean(inferenceSelection[resource.file_id])}
                              onCheckedChange={(checked) =>
                                setInferenceSelection((prev) => ({
                                  ...prev,
                                  [resource.file_id]: checked === true,
                                }))
                              }
                            />
                            <span className="truncate">{resource.original_name}</span>
                          </label>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Inference Job</CardTitle>
                  <CardDescription>Live status and result summary</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  {inferenceJob ? (
                    <>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={statusBadgeClass(inferenceJob.status)}>
                          {inferenceJob.status}
                        </Badge>
                        <span className="truncate text-xs text-muted-foreground">{inferenceJob.job_id}</span>
                      </div>
                      <p>Model: {inferenceJob.model_key}</p>
                      <p>Version: {inferenceJob.model_version ?? "builtin"}</p>
                      <p>
                        Predictions: {Number((inferenceJob.result?.prediction_count as number | undefined) ?? 0)}
                      </p>
                      {inferenceJob.result?.counts_by_class ? (
                        <div className="rounded-md border border-border bg-muted/30 p-2 text-xs">
                          {Object.entries(
                            inferenceJob.result.counts_by_class as Record<string, number>
                          ).map(([key, value]) => (
                            <p key={key}>
                              {key}: {Number(value || 0)}
                            </p>
                          ))}
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <p className="text-muted-foreground">No inference run yet.</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
}
