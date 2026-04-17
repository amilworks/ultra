type HydrationRunArtifact = {
  path?: string;
  url?: string;
  downloadUrl?: string;
};

type HydrationProgressEvent = {
  tool?: string;
};

type HydrationCardImage = {
  url?: string;
  downloadUrl?: string;
};

type HydrationCardFigure = {
  url?: string;
  downloadUrl?: string;
};

type HydrationToolCard = {
  tool?: string;
  images?: HydrationCardImage[];
  yoloFigureAvailability?: { missingAnnotatedFigure?: boolean } | null;
  megasegInsights?:
    | {
        heroFigure?: HydrationCardFigure | null;
        secondaryFigures?: HydrationCardFigure[];
      }
    | null;
};

type HydrationMessage = {
  role?: string;
  runId?: string | null;
  runArtifacts?: HydrationRunArtifact[];
  progressEvents?: HydrationProgressEvent[];
  responseMetadata?: Record<string, unknown> | null;
};

const VISUAL_TOOL_NAMES = new Set<string>([
  "segment_image_megaseg",
  "segment_image_sam2",
  "segment_image_sam3",
  "estimate_depth_pro",
  "yolo_detect",
  "segment_video_sam2",
]);

const looksLikeLocalFilesystemPath = (value: string): boolean => {
  const normalized = String(value || "").trim();
  if (!normalized) {
    return false;
  }
  if (normalized.startsWith("file://")) {
    return true;
  }
  if (/^[A-Za-z]:[\\/]/.test(normalized)) {
    return true;
  }
  return /^\/(?:srv|mnt|private|Users)\//.test(normalized);
};

const visualToolName = (value: unknown): string =>
  String(value ?? "").trim().toLowerCase();

const messageHasVisualToolSignal = (message: HydrationMessage): boolean => {
  const progressEvents = Array.isArray(message.progressEvents)
    ? message.progressEvents
    : [];
  if (
    progressEvents.some((event) => VISUAL_TOOL_NAMES.has(visualToolName(event?.tool)))
  ) {
    return true;
  }
  const responseMetadata =
    message.responseMetadata &&
    typeof message.responseMetadata === "object" &&
    !Array.isArray(message.responseMetadata)
      ? message.responseMetadata
      : {};
  const toolInvocations = Array.isArray(responseMetadata.tool_invocations)
    ? responseMetadata.tool_invocations
    : [];
  return toolInvocations.some((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return false;
    }
    return VISUAL_TOOL_NAMES.has(visualToolName((entry as { tool?: unknown }).tool));
  });
};

export const shouldHydrateRunArtifacts = (
  message: HydrationMessage,
  toolCards: HydrationToolCard[]
): boolean => {
  if (String(message.role || "").trim().toLowerCase() !== "assistant") {
    return false;
  }
  if (!String(message.runId || "").trim()) {
    return false;
  }
  if (
    toolCards.some((card) => Boolean(card.yoloFigureAvailability?.missingAnnotatedFigure))
  ) {
    return true;
  }

  const hasVisualToolCard = toolCards.some((card) => {
    if (VISUAL_TOOL_NAMES.has(visualToolName(card.tool))) {
      return true;
    }
    if ((card.images?.length ?? 0) > 0) {
      return true;
    }
    return Boolean(card.megasegInsights);
  });
  const hasVisualSignal = hasVisualToolCard || messageHasVisualToolSignal(message);
  if (!hasVisualSignal) {
    return false;
  }

  const runArtifacts = Array.isArray(message.runArtifacts) ? message.runArtifacts : [];
  if (runArtifacts.length === 0) {
    return true;
  }

  const candidateUrls = new Set<string>();
  runArtifacts.forEach((artifact) => {
    [artifact?.url, artifact?.downloadUrl, artifact?.path].forEach((value) => {
      const token = String(value || "").trim();
      if (token) {
        candidateUrls.add(token);
      }
    });
  });
  toolCards.forEach((card) => {
    (card.images ?? []).forEach((image) => {
      [image?.url, image?.downloadUrl].forEach((value) => {
        const token = String(value || "").trim();
        if (token) {
          candidateUrls.add(token);
        }
      });
    });
    const heroFigure = card.megasegInsights?.heroFigure;
    if (heroFigure) {
      [heroFigure.url, heroFigure.downloadUrl].forEach((value) => {
        const token = String(value || "").trim();
        if (token) {
          candidateUrls.add(token);
        }
      });
    }
    (card.megasegInsights?.secondaryFigures ?? []).forEach((figure) => {
      [figure.url, figure.downloadUrl].forEach((value) => {
        const token = String(value || "").trim();
        if (token) {
          candidateUrls.add(token);
        }
      });
    });
  });

  return Array.from(candidateUrls).some(looksLikeLocalFilesystemPath);
};
