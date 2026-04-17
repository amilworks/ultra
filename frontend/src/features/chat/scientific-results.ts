export type ScientificResultProgressEvent = {
  event?: string;
  tool?: string;
  summary?: Record<string, unknown> | null;
};

export type ScientificToolInvocation = {
  tool?: string;
  status?: string;
  output_summary?: Record<string, unknown> | null;
  output_envelope?: Record<string, unknown> | null;
};

export type ScientificResultArtifact = {
  path: string;
  title: string;
  sourcePath?: string;
  url: string;
  downloadUrl?: string;
  previewable: boolean;
  resultGroupId?: string | null;
};

export type ScientificResultFigure = {
  key: string;
  kind: string;
  title: string;
  file?: string;
  summary?: string;
  url: string;
  downloadUrl?: string;
  previewable: boolean;
};

export type ScientificResultFileRow = {
  file: string;
  coveragePercent: number | null;
  objectCount: number | null;
  activeSliceCount: number | null;
  zSliceCount: number | null;
  largestComponentVoxels: number | null;
  technicalSummary?: string;
};

export type ScientificResultGroup = {
  resultGroupId: string;
  reportPath?: string;
  summaryCsvPath?: string;
  heroFigure: ScientificResultFigure | null;
  secondaryFigures: ScientificResultFigure[];
  fileRows: ScientificResultFileRow[];
  metrics: {
    coveragePercent: number | null;
    objectCount: number | null;
    activeSliceCount: number | null;
    zSliceCount: number | null;
    largestComponentVoxels: number | null;
  };
  technicalSummary?: string;
};

const normalizeToolName = (value: unknown): string =>
  String(value ?? "").trim().toLowerCase();

const toRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;

const toNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const artifactLookupKeys = (value: string): string[] => {
  const token = String(value || "").trim();
  if (!token) {
    return [];
  }
  const normalized = token.replace(/\\/g, "/").toLowerCase();
  const parts = normalized.split("/").filter(Boolean);
  const filename = parts[parts.length - 1] ?? normalized;
  const keys = new Set<string>([normalized, filename]);
  if (filename.includes("__")) {
    keys.add(filename.slice(filename.indexOf("__") + 2));
  }
  return Array.from(keys);
};

const firstString = (...values: unknown[]): string => {
  for (const value of values) {
    const token = String(value ?? "").trim();
    if (token) {
      return token;
    }
  }
  return "";
};

const resultGroupIdFromEnvelope = (value: Record<string, unknown> | null): string =>
  firstString(
    value?.result_group_id,
    toRecord(value?.latest_result_refs)?.latest_segmentation_result_group_id,
    value?.output_directory
  );

const resultGroupIdFromProgressEvent = (event: ScientificResultProgressEvent): string =>
  firstString(
    toRecord(event.summary)?.result_group_id,
    toRecord(event.summary)?.output_directory
  );

const scientificFigureSortWeight = (kind: string): number => {
  const normalized = kind.trim().toLowerCase();
  if (normalized === "overlay_mip") {
    return 0;
  }
  if (normalized === "overlay_mid_z") {
    return 1;
  }
  if (normalized === "raw_preview") {
    return 2;
  }
  if (normalized === "probability_preview") {
    return 3;
  }
  if (normalized === "mask_preview") {
    return 4;
  }
  return 5;
};

const isBrowserUrl = (value: string): boolean => /^(?:[a-z]+:)?\/\//i.test(value);

export const buildScientificResultGroups = ({
  progressEvents,
  toolInvocations,
  runArtifacts,
  runId,
  buildArtifactDownloadUrl,
}: {
  progressEvents: ScientificResultProgressEvent[];
  toolInvocations: ScientificToolInvocation[];
  runArtifacts: ScientificResultArtifact[];
  runId?: string;
  buildArtifactDownloadUrl?: (runId: string, path: string) => string;
}): ScientificResultGroup[] => {
  const artifactByKey = new Map<string, ScientificResultArtifact[]>();
  runArtifacts.forEach((artifact) => {
    const lookupValues = new Set<string>([
      artifact.path,
      artifact.title,
      artifact.sourcePath ?? "",
      artifact.resultGroupId ?? "",
    ]);
    lookupValues.forEach((value) => {
      artifactLookupKeys(value).forEach((key) => {
        const existing = artifactByKey.get(key) ?? [];
        existing.push(artifact);
        artifactByKey.set(key, existing);
      });
    });
  });

  type MutableGroup = {
    resultGroupId: string;
    reportPath?: string;
    summaryCsvPath?: string;
    figures: ScientificResultFigure[];
    fileRows: ScientificResultFileRow[];
    metrics: ScientificResultGroup["metrics"];
    technicalSummary?: string;
    seenFigureKeys: Set<string>;
  };

  const groups = new Map<string, MutableGroup>();
  const getGroup = (rawGroupId: string): MutableGroup => {
    const normalized =
      rawGroupId.trim() || `scientific-result-${Math.max(1, groups.size + 1)}`;
    const existing = groups.get(normalized);
    if (existing) {
      return existing;
    }
    const created: MutableGroup = {
      resultGroupId: normalized,
      figures: [],
      fileRows: [],
      metrics: {
        coveragePercent: null,
        objectCount: null,
        activeSliceCount: null,
        zSliceCount: null,
        largestComponentVoxels: null,
      },
      seenFigureKeys: new Set<string>(),
    };
    groups.set(normalized, created);
    return created;
  };

  for (const invocation of toolInvocations) {
    if (normalizeToolName(invocation.status) !== "completed") {
      continue;
    }
    const toolName = normalizeToolName(invocation.tool);
    if (toolName !== "segment_image_megaseg" && toolName !== "quantify_segmentation_masks") {
      continue;
    }
    const envelope = toRecord(invocation.output_envelope);
    if (!envelope) {
      continue;
    }
    const group = getGroup(resultGroupIdFromEnvelope(envelope));
    if (toolName === "segment_image_megaseg") {
      group.reportPath = firstString(group.reportPath, envelope.report_path);
      group.summaryCsvPath = firstString(group.summaryCsvPath, envelope.summary_csv_path);
      const scientificSummary = toRecord(envelope.scientific_summary);
      const scientificFiles = Array.isArray(scientificSummary?.files)
        ? scientificSummary?.files
        : [];
      for (const rawRow of scientificFiles) {
        const row = toRecord(rawRow);
        if (!row) {
          continue;
        }
        const normalizedRow: ScientificResultFileRow = {
          file: firstString(row.file, row.path, "image"),
          coveragePercent: toNumber(row.coverage_percent),
          objectCount: toNumber(row.object_count),
          activeSliceCount: toNumber(row.active_slice_count),
          zSliceCount: toNumber(row.z_slice_count),
          largestComponentVoxels: toNumber(row.largest_component_voxels),
          technicalSummary: firstString(row.technical_summary) || undefined,
        };
        const key = normalizedRow.file.toLowerCase();
        const existingIndex = group.fileRows.findIndex(
          (candidate) => candidate.file.toLowerCase() === key
        );
        if (existingIndex >= 0) {
          group.fileRows[existingIndex] = {
            ...group.fileRows[existingIndex],
            ...normalizedRow,
          };
        } else {
          group.fileRows.push(normalizedRow);
        }
      }
      const firstRow = group.fileRows[0];
      group.metrics.coveragePercent =
        firstRow?.coveragePercent ?? group.metrics.coveragePercent;
      group.metrics.objectCount = firstRow?.objectCount ?? group.metrics.objectCount;
      group.metrics.activeSliceCount =
        firstRow?.activeSliceCount ?? group.metrics.activeSliceCount;
      group.metrics.zSliceCount = firstRow?.zSliceCount ?? group.metrics.zSliceCount;
      group.metrics.largestComponentVoxels =
        firstRow?.largestComponentVoxels ?? group.metrics.largestComponentVoxels;
      group.technicalSummary =
        firstRow?.technicalSummary ?? group.technicalSummary;

      const visualizationRows = Array.isArray(envelope.visualization_paths)
        ? envelope.visualization_paths
        : [];
      for (const rawFigure of visualizationRows) {
        const figure = toRecord(rawFigure);
        if (!figure) {
          continue;
        }
        const path = firstString(figure.path);
        if (!path) {
          continue;
        }
        const matchingArtifact =
          artifactLookupKeys(firstString(figure.result_group_id, path, figure.file))
            .flatMap((key) => artifactByKey.get(key) ?? [])
            .find(
              (artifact) =>
                artifactLookupKeys(path).some((key) =>
                  artifactLookupKeys(artifact.path).includes(key)
                ) ||
                artifactLookupKeys(path).some((key) =>
                  artifactLookupKeys(firstString(artifact.sourcePath)).includes(key)
                ) ||
                artifactLookupKeys(firstString(figure.file)).some((key) =>
                  artifactLookupKeys(firstString(artifact.sourcePath)).includes(key)
                )
            ) ?? null;
        const figureKey = firstString(path, figure.title, figure.kind);
        if (!figureKey || group.seenFigureKeys.has(figureKey)) {
          continue;
        }
        const fallbackFigureUrl =
          runId && buildArtifactDownloadUrl && !isBrowserUrl(path)
            ? buildArtifactDownloadUrl(runId, path)
            : path;
        group.seenFigureKeys.add(figureKey);
        group.figures.push({
          key: figureKey,
          kind: firstString(figure.kind, "figure").toLowerCase(),
          title: firstString(figure.title, "Megaseg figure"),
          file: firstString(figure.file) || undefined,
          summary: undefined,
          url: matchingArtifact?.url ?? fallbackFigureUrl,
          downloadUrl:
            matchingArtifact?.downloadUrl ?? matchingArtifact?.url ?? fallbackFigureUrl,
          previewable: matchingArtifact?.previewable ?? true,
        });
      }
    }

    if (toolName === "quantify_segmentation_masks") {
      const summary = toRecord(envelope.summary);
      const rows = Array.isArray(envelope.rows) ? envelope.rows : [];
      const firstRow = toRecord(rows[0]);
      group.metrics.coveragePercent =
        group.metrics.coveragePercent ??
        toNumber(firstRow?.coverage_percent) ??
        toNumber(summary?.mean_coverage_percent);
      group.metrics.objectCount =
        group.metrics.objectCount ??
        toNumber(firstRow?.object_count) ??
        toNumber(summary?.mean_object_count);
    }
  }

  for (const progressEvent of progressEvents) {
    if (
      normalizeToolName(progressEvent.event) !== "completed" ||
      normalizeToolName(progressEvent.tool) !== "segment_image_megaseg"
    ) {
      continue;
    }
    getGroup(resultGroupIdFromProgressEvent(progressEvent));
  }

  return Array.from(groups.values())
    .filter(
      (group) =>
        group.figures.length > 0 ||
        group.fileRows.length > 0 ||
        Boolean(group.reportPath) ||
        Boolean(group.summaryCsvPath)
    )
    .map((group) => {
      const sortedFigures = [...group.figures].sort(
        (left, right) =>
          scientificFigureSortWeight(left.kind) - scientificFigureSortWeight(right.kind) ||
          left.title.localeCompare(right.title)
      );
      return {
        resultGroupId: group.resultGroupId,
        reportPath: group.reportPath,
        summaryCsvPath: group.summaryCsvPath,
        heroFigure: sortedFigures[0] ?? null,
        secondaryFigures: sortedFigures.slice(1),
        fileRows: group.fileRows,
        metrics: group.metrics,
        technicalSummary: group.technicalSummary,
      } satisfies ScientificResultGroup;
    })
    .sort((left, right) => left.resultGroupId.localeCompare(right.resultGroupId));
};
