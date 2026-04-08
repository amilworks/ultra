import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  MoveDiagonal2,
  Trash2,
  Undo2,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { ApiClient } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  Sam3ImageAnnotation,
  Sam3InteractiveRequest,
  UploadViewerInfo,
  UploadedFileRecord,
} from "@/types";

type PromptLabel = 0 | 1;
type InteractiveSegmentationModel = "medsam";
type InteractionMode = "point" | "box";
type TrackerPromptMode = "single_object_refine" | "per_positive_point_instance";

type AnnotationPoint = {
  id: string;
  x: number;
  y: number;
  label: PromptLabel;
  order: number;
};

type AnnotationBox = {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: PromptLabel;
  order: number;
};

type ImageAnnotations = {
  points: AnnotationPoint[];
  boxes: AnnotationBox[];
};

type SelectedAnnotation =
  | {
      kind: "point" | "box";
      id: string;
    }
  | null;

type PointerDraft = {
  pointerId: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type ImageSize = {
  width: number;
  height: number;
};

type ImageViewerState = {
  sourceSize: ImageSize;
  imageUrl: string;
  loading: boolean;
  error: string | null;
};

const DEFAULT_IMAGE_SIZE: ImageSize = { width: 1, height: 1 };

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const clamp01 = (value: number): number => clamp(value, 0, 1);

const toPromptLabel = (value: string): PromptLabel => (value === "0" ? 0 : 1);
const inferInteractiveModelFromPrompt = (
  value: string | null | undefined
): InteractiveSegmentationModel => {
  void value;
  return "medsam";
};

const modelUiCopy: Record<
  InteractiveSegmentationModel,
  { label: string; note: string; running: string }
> = {
  medsam: {
    label: "MedSAM2",
    note: "Interactive scientific segmentation now defaults to MedSAM2 for point-guided masks and per-object previews.",
    running: "Running MedSAM2…",
  },
};

const makeAnnotationId = (): string =>
  `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;

const normalizeBox = (box: AnnotationBox): AnnotationBox | null => {
  const x1 = Math.min(box.x1, box.x2);
  const x2 = Math.max(box.x1, box.x2);
  const y1 = Math.min(box.y1, box.y2);
  const y2 = Math.max(box.y1, box.y2);
  if (x2 - x1 < 1 || y2 - y1 < 1) {
    return null;
  }
  return {
    ...box,
    x1,
    x2,
    y1,
    y2,
  };
};

const countPromptBreakdown = (
  annotations: ImageAnnotations
): { positivePoints: number; negativePoints: number; boxes: number } => {
  const positivePoints = annotations.points.filter((point) => point.label === 1).length;
  const negativePoints = annotations.points.filter((point) => point.label === 0).length;
  return {
    positivePoints,
    negativePoints,
    boxes: annotations.boxes.length,
  };
};

const toErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message.trim();
  }
  return "Viewer metadata is unavailable for this file.";
};

const resolveSourceImageSize = (viewerInfo: UploadViewerInfo): ImageSize => {
  const planePixelSize = viewerInfo.viewer?.default_plane?.pixel_size;
  if (
    planePixelSize &&
    Number.isFinite(planePixelSize.width) &&
    Number.isFinite(planePixelSize.height) &&
    planePixelSize.width > 0 &&
    planePixelSize.height > 0
  ) {
    return {
      width: Math.max(1, Math.round(planePixelSize.width)),
      height: Math.max(1, Math.round(planePixelSize.height)),
    };
  }
  return {
    width: Math.max(1, Math.round(viewerInfo.axis_sizes?.X || 1)),
    height: Math.max(1, Math.round(viewerInfo.axis_sizes?.Y || 1)),
  };
};

const resolveViewerImageUrl = (
  fileId: string,
  viewerInfo: UploadViewerInfo,
  apiClient: ApiClient
): string => {
  if (viewerInfo.service_urls?.display) {
    return apiClient.uploadDisplayUrl(fileId, viewerInfo.service_urls.display);
  }
  return apiClient.uploadPreviewUrl(fileId);
};

const getSourceCoordinatesFromEvent = (
  event: { clientX: number; clientY: number },
  imageElement: HTMLImageElement | null,
  sourceSize: ImageSize
): { x: number; y: number } | null => {
  if (!imageElement) {
    return null;
  }
  const rect = imageElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const ratioX = clamp01((event.clientX - rect.left) / rect.width);
  const ratioY = clamp01((event.clientY - rect.top) / rect.height);
  const maxX = Math.max(0, sourceSize.width - 1);
  const maxY = Math.max(0, sourceSize.height - 1);
  return {
    x: clamp(Math.round(ratioX * maxX), 0, maxX),
    y: clamp(Math.round(ratioY * maxY), 0, maxY),
  };
};

type Sam3AnnotationDialogProps = {
  open: boolean;
  files: UploadedFileRecord[];
  apiClient: ApiClient;
  busy: boolean;
  portalContainer?: HTMLElement | null;
  conversationId?: string | null;
  initialPromptText?: string | null;
  onOpenChange: (open: boolean) => void;
  onSubmit: (payload: Sam3InteractiveRequest) => Promise<void>;
};

export function Sam3AnnotationDialog({
  open,
  files,
  apiClient,
  busy,
  portalContainer,
  conversationId,
  initialPromptText,
  onOpenChange,
  onSubmit,
}: Sam3AnnotationDialogProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [promptLabel, setPromptLabel] = useState<PromptLabel>(1);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("point");
  const [segmentationModel, setSegmentationModel] =
    useState<InteractiveSegmentationModel>(
      inferInteractiveModelFromPrompt(initialPromptText)
    );
  const [trackerPromptMode, setTrackerPromptMode] =
    useState<TrackerPromptMode>("per_positive_point_instance");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [annotationsByFileId, setAnnotationsByFileId] = useState<
    Record<string, ImageAnnotations>
  >({});
  const [selectedAnnotation, setSelectedAnnotation] =
    useState<SelectedAnnotation>(null);
  const [draftBox, setDraftBox] = useState<PointerDraft | null>(null);
  const [viewerStateByFileId, setViewerStateByFileId] = useState<
    Record<string, ImageViewerState>
  >({});

  const imageRef = useRef<HTMLImageElement | null>(null);
  const annotationCounterRef = useRef(0);

  useEffect(() => {
    if (!open) {
      return;
    }
    annotationCounterRef.current = 0;
    setSegmentationModel(inferInteractiveModelFromPrompt(initialPromptText));
    setTrackerPromptMode("per_positive_point_instance");
    setPromptLabel(1);
    setInteractionMode("point");
    setAdvancedOpen(false);
    setSelectedAnnotation(null);
    setDraftBox(null);
  }, [initialPromptText, open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setAnnotationsByFileId((previous) => {
      const next: Record<string, ImageAnnotations> = {};
      files.forEach((file) => {
        next[file.file_id] = previous[file.file_id] ?? { points: [], boxes: [] };
      });
      return next;
    });
    setViewerStateByFileId((previous) => {
      const next: Record<string, ImageViewerState> = {};
      files.forEach((file) => {
        next[file.file_id] = previous[file.file_id] ?? {
          sourceSize: DEFAULT_IMAGE_SIZE,
          imageUrl: apiClient.uploadPreviewUrl(file.file_id),
          loading: true,
          error: null,
        };
      });
      return next;
    });
    setActiveIndex((current) => {
      if (files.length === 0) {
        return 0;
      }
      return Math.min(Math.max(current, 0), files.length - 1);
    });
  }, [apiClient, files, open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    let cancelled = false;
    files.forEach((file) => {
      setViewerStateByFileId((previous) => ({
        ...previous,
        [file.file_id]: {
          sourceSize: previous[file.file_id]?.sourceSize ?? DEFAULT_IMAGE_SIZE,
          imageUrl:
            previous[file.file_id]?.imageUrl ?? apiClient.uploadPreviewUrl(file.file_id),
          loading: true,
          error: null,
        },
      }));
      void apiClient
        .getUploadViewer(file.file_id)
        .then((viewerInfo) => {
          if (cancelled) {
            return;
          }
          setViewerStateByFileId((previous) => ({
            ...previous,
            [file.file_id]: {
              sourceSize: resolveSourceImageSize(viewerInfo),
              imageUrl: resolveViewerImageUrl(file.file_id, viewerInfo, apiClient),
              loading: false,
              error: null,
            },
          }));
        })
        .catch((error) => {
          if (cancelled) {
            return;
          }
          setViewerStateByFileId((previous) => ({
            ...previous,
            [file.file_id]: {
              sourceSize: previous[file.file_id]?.sourceSize ?? DEFAULT_IMAGE_SIZE,
              imageUrl:
                previous[file.file_id]?.imageUrl ?? apiClient.uploadPreviewUrl(file.file_id),
              loading: false,
              error: toErrorMessage(error),
            },
          }));
        });
    });
    return () => {
      cancelled = true;
    };
  }, [apiClient, files, open]);

  useEffect(() => {
    setSelectedAnnotation(null);
    setDraftBox(null);
  }, [activeIndex]);

  const activeFile = files[activeIndex] ?? null;
  const activeFileId = activeFile?.file_id ?? null;
  const activeViewerState: ImageViewerState = useMemo(() => {
    if (!activeFileId) {
      return {
        sourceSize: DEFAULT_IMAGE_SIZE,
        imageUrl: "",
        loading: false,
        error: null,
      };
    }
    return (
      viewerStateByFileId[activeFileId] ?? {
        sourceSize: DEFAULT_IMAGE_SIZE,
        imageUrl: apiClient.uploadPreviewUrl(activeFileId),
        loading: true,
        error: null,
      }
    );
  }, [activeFileId, apiClient, viewerStateByFileId]);

  const activeAnnotations: ImageAnnotations = useMemo(() => {
    if (!activeFileId) {
      return { points: [], boxes: [] };
    }
    return annotationsByFileId[activeFileId] ?? { points: [], boxes: [] };
  }, [activeFileId, annotationsByFileId]);

  const activeBreakdown = useMemo(
    () => countPromptBreakdown(activeAnnotations),
    [activeAnnotations]
  );

  const totals = useMemo(() => {
    return files.reduce(
      (acc, file) => {
        const item = annotationsByFileId[file.file_id] ?? { points: [], boxes: [] };
        const breakdown = countPromptBreakdown(item);
        return {
          positivePoints: acc.positivePoints + breakdown.positivePoints,
          negativePoints: acc.negativePoints + breakdown.negativePoints,
          boxes: acc.boxes + breakdown.boxes,
        };
      },
      { positivePoints: 0, negativePoints: 0, boxes: 0 }
    );
  }, [annotationsByFileId, files]);

  const selectedModelCopy = modelUiCopy[segmentationModel];
  const selectedDetails = useMemo(() => {
    if (!selectedAnnotation) {
      return null;
    }
    if (selectedAnnotation.kind === "point") {
      const point = activeAnnotations.points.find(
        (item) => item.id === selectedAnnotation.id
      );
      if (!point) {
        return null;
      }
      return {
        label: point.label === 1 ? "Include point" : "Exclude point",
        detail: `${point.x}, ${point.y}`,
      };
    }
    const box = activeAnnotations.boxes.find((item) => item.id === selectedAnnotation.id);
    if (!box) {
      return null;
    }
    return {
      label: box.label === 1 ? "Include box" : "Exclude box",
      detail: `${box.x1}, ${box.y1} → ${box.x2}, ${box.y2}`,
    };
  }, [activeAnnotations.boxes, activeAnnotations.points, selectedAnnotation]);

  const canPlacePrompts =
    Boolean(activeFileId) &&
    !busy &&
    !activeViewerState.loading &&
    !activeViewerState.error &&
    activeViewerState.sourceSize.width > 0 &&
    activeViewerState.sourceSize.height > 0;

  const hasSubmittablePrompts =
    totals.positivePoints > 0 || totals.boxes > 0;

  useEffect(() => {
    if (!open || !activeFileId) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent): void => {
      if ((event.key === "Delete" || event.key === "Backspace") && selectedAnnotation) {
        event.preventDefault();
        setAnnotationsByFileId((previous) => {
          const current = previous[activeFileId] ?? { points: [], boxes: [] };
          if (selectedAnnotation.kind === "point") {
            return {
              ...previous,
              [activeFileId]: {
                ...current,
                points: current.points.filter(
                  (point) => point.id !== selectedAnnotation.id
                ),
              },
            };
          }
          return {
            ...previous,
            [activeFileId]: {
              ...current,
              boxes: current.boxes.filter((box) => box.id !== selectedAnnotation.id),
            },
          };
        });
        setSelectedAnnotation(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeFileId, open, selectedAnnotation]);

  const nextAnnotationOrder = (): number => {
    annotationCounterRef.current += 1;
    return annotationCounterRef.current;
  };

  const addPointAnnotation = (
    coords: { x: number; y: number },
    label: PromptLabel
  ): void => {
    if (!activeFileId) {
      return;
    }
    const nextPoint: AnnotationPoint = {
      id: makeAnnotationId(),
      x: coords.x,
      y: coords.y,
      label,
      order: nextAnnotationOrder(),
    };
    setAnnotationsByFileId((previous) => {
      const current = previous[activeFileId] ?? { points: [], boxes: [] };
      return {
        ...previous,
        [activeFileId]: {
          ...current,
          points: [...current.points, nextPoint],
        },
      };
    });
    setSelectedAnnotation({ kind: "point", id: nextPoint.id });
  };

  const handleStageClick = (
    event: ReactMouseEvent<HTMLDivElement>
  ): void => {
    if (!canPlacePrompts || !activeFileId || interactionMode !== "point") {
      return;
    }
    const coords = getSourceCoordinatesFromEvent(
      event,
      imageRef.current,
      activeViewerState.sourceSize
    );
    if (!coords) {
      return;
    }
    addPointAnnotation(coords, promptLabel);
  };

  const handleStagePointerDown = (
    event: ReactPointerEvent<HTMLDivElement>
  ): void => {
    if (!canPlacePrompts || !activeFileId) {
      return;
    }
    if (event.button !== 0) {
      return;
    }
    if (interactionMode !== "box") {
      return;
    }
    const coords = getSourceCoordinatesFromEvent(
      event,
      imageRef.current,
      activeViewerState.sourceSize
    );
    if (!coords) {
      return;
    }
    setDraftBox({
      pointerId: event.pointerId,
      x1: coords.x,
      y1: coords.y,
      x2: coords.x,
      y2: coords.y,
    });
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelectedAnnotation(null);
  };

  const handleStagePointerMove = (
    event: ReactPointerEvent<HTMLDivElement>
  ): void => {
    if (interactionMode !== "box" || !draftBox || draftBox.pointerId !== event.pointerId) {
      return;
    }
    const coords = getSourceCoordinatesFromEvent(
      event,
      imageRef.current,
      activeViewerState.sourceSize
    );
    if (!coords) {
      return;
    }
    setDraftBox((previous) =>
      previous
        ? {
            ...previous,
            x2: coords.x,
            y2: coords.y,
          }
        : previous
    );
  };

  const handleStagePointerUp = (
    event: ReactPointerEvent<HTMLDivElement>
  ): void => {
    if (!draftBox || !activeFileId || draftBox.pointerId !== event.pointerId) {
      return;
    }
    const normalized = normalizeBox({
      id: makeAnnotationId(),
      x1: draftBox.x1,
      y1: draftBox.y1,
      x2: draftBox.x2,
      y2: draftBox.y2,
      label: promptLabel,
      order: nextAnnotationOrder(),
    });
    setDraftBox(null);
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (!normalized) {
      return;
    }
    setAnnotationsByFileId((previous) => {
      const current = previous[activeFileId] ?? { points: [], boxes: [] };
      return {
        ...previous,
        [activeFileId]: {
          ...current,
          boxes: [...current.boxes, normalized],
        },
      };
    });
    setSelectedAnnotation({ kind: "box", id: normalized.id });
  };

  const clearActiveFileAnnotations = (): void => {
    if (!activeFileId || busy) {
      return;
    }
    setAnnotationsByFileId((previous) => ({
      ...previous,
      [activeFileId]: {
        points: [],
        boxes: [],
      },
    }));
    setSelectedAnnotation(null);
    setDraftBox(null);
  };

  const deleteSelectedAnnotation = (): void => {
    if (!activeFileId || !selectedAnnotation || busy) {
      return;
    }
    setAnnotationsByFileId((previous) => {
      const current = previous[activeFileId] ?? { points: [], boxes: [] };
      if (selectedAnnotation.kind === "point") {
        return {
          ...previous,
          [activeFileId]: {
            ...current,
            points: current.points.filter(
              (point) => point.id !== selectedAnnotation.id
            ),
          },
        };
      }
      return {
        ...previous,
        [activeFileId]: {
          ...current,
          boxes: current.boxes.filter((box) => box.id !== selectedAnnotation.id),
        },
      };
    });
    setSelectedAnnotation(null);
  };

  const undoLastAnnotation = (): void => {
    if (!activeFileId || busy) {
      return;
    }
    setAnnotationsByFileId((previous) => {
      const current = previous[activeFileId] ?? { points: [], boxes: [] };
      const lastPoint = current.points[current.points.length - 1] ?? null;
      const lastBox = current.boxes[current.boxes.length - 1] ?? null;
      if (!lastPoint && !lastBox) {
        return previous;
      }
      if (!lastBox || (lastPoint && lastPoint.order > lastBox.order)) {
        return {
          ...previous,
          [activeFileId]: {
            ...current,
            points: current.points.slice(0, -1),
          },
        };
      }
      return {
        ...previous,
        [activeFileId]: {
          ...current,
          boxes: current.boxes.slice(0, -1),
        },
      };
    });
    setSelectedAnnotation(null);
  };

  const convertToRequestAnnotations = (): Sam3ImageAnnotation[] => {
    return files.map((file) => {
      const item = annotationsByFileId[file.file_id] ?? { points: [], boxes: [] };
      return {
        file_id: file.file_id,
        points: item.points.map((point) => ({
          x: point.x,
          y: point.y,
          label: point.label,
        })),
        boxes: item.boxes.map((box) => {
          const normalized = normalizeBox(box) ?? box;
          return {
            x1: normalized.x1,
            y1: normalized.y1,
            x2: normalized.x2,
            y2: normalized.y2,
            label: normalized.label,
          };
        }),
      };
    });
  };

  const submitAnnotations = async (): Promise<void> => {
    if (busy || files.length === 0 || !hasSubmittablePrompts) {
      return;
    }
    const payload: Sam3InteractiveRequest = {
      file_ids: files.map((file) => file.file_id),
      annotations: convertToRequestAnnotations(),
      model: segmentationModel,
      conversation_id: conversationId ?? null,
      concept_prompt: null,
      save_visualizations: true,
      preset: "balanced",
      tracker_prompt_mode: trackerPromptMode,
      force_rerun: true,
    };
    await onSubmit(payload);
  };

  const pointHelpText =
    interactionMode === "box"
      ? "Drag a local box only when a single point is too ambiguous."
      : promptLabel === 1
        ? trackerPromptMode === "single_object_refine"
          ? "Click multiple times inside the same target to refine one MedSAM2 mask."
          : "Click once inside each target structure. Each positive point seeds a separate object with its own preview color."
        : "Click background regions to suppress accidental mask growth around your positive points.";

  const overlayCursorClass = !canPlacePrompts
    ? "cursor-not-allowed"
    : interactionMode === "box"
      ? "cursor-crosshair"
      : "cursor-cell";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        portalContainer={portalContainer}
        overlayClassName="pointer-events-none bg-transparent"
        className="absolute left-1/2 top-auto bottom-2 z-50 h-[min(92svh,860px)] w-[calc(100%-0.75rem)] max-w-[calc(100%-0.75rem)] -translate-x-1/2 translate-y-0 rounded-2xl border-border/70 bg-background/95 p-0 shadow-[0_26px_78px_rgba(0,0,0,0.32)] dark:border-white/10 dark:bg-zinc-950/96 dark:shadow-[0_30px_90px_rgba(0,0,0,0.58)] supports-[backdrop-filter]:bg-background/90 dark:supports-[backdrop-filter]:bg-zinc-950/92 md:top-1/2 md:bottom-auto md:h-[min(88svh,860px)] md:w-[min(calc(100%-2rem),var(--user-chat-width))] md:max-w-[var(--user-chat-width)] md:-translate-y-1/2 grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden gap-0"
      >
        <DialogHeader className="border-b px-4 pt-4 pb-3 sm:px-6 sm:pt-5 sm:pb-4">
          <DialogTitle className="text-base tracking-tight">
            Interactive segmentation
          </DialogTitle>
          <DialogDescription className="pt-0.5 text-sm leading-relaxed">
            Interactive point prompts now run on MedSAM2 by default. Use one
            positive point per target for separate objects, or switch to
            single-object refinement in Advanced when you want one mask.
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 space-y-4 overflow-y-auto px-3 py-3 sm:px-6 sm:py-4">
          <Card className="border-dashed">
            <CardContent className="space-y-4 p-3">
              <div className="space-y-2">
                <Label>Segmentation model</Label>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="secondary">{selectedModelCopy.label}</Badge>
                  <span className="text-xs text-muted-foreground">
                    SAM3 is currently disabled for this interactive workflow.
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  {selectedModelCopy.note}
                </p>
              </div>

              <div className="space-y-2">
                <Label>Point prompt mode</Label>
                <div className="flex flex-wrap items-center gap-2">
                  <ToggleGroup
                    type="single"
                    value={String(promptLabel)}
                    onValueChange={(value) => {
                      if (!value) {
                        return;
                      }
                      setPromptLabel(toPromptLabel(value));
                    }}
                    variant="outline"
                    size="sm"
                  >
                    <ToggleGroupItem
                      value="1"
                      className="data-[state=on]:bg-emerald-500/20 data-[state=on]:text-emerald-700 dark:data-[state=on]:text-emerald-300"
                    >
                      + Include
                    </ToggleGroupItem>
                    <ToggleGroupItem
                      value="0"
                      className="data-[state=on]:bg-rose-500/20 data-[state=on]:text-rose-700 dark:data-[state=on]:text-rose-300"
                    >
                      − Exclude
                    </ToggleGroupItem>
                  </ToggleGroup>
                  <Badge variant="secondary">
                    Points: {totals.positivePoints + totals.negativePoints}
                  </Badge>
                  <Badge variant="secondary">
                    Boxes: {totals.boxes}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">{pointHelpText}</p>
              </div>

              <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
                <div className="rounded-xl border border-border/70 bg-muted/20">
                  <CollapsibleTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      className="flex h-auto w-full items-center justify-between rounded-xl px-3 py-2 text-sm"
                    >
                      <span>Advanced</span>
                      <ChevronDown
                        className={cn(
                          "size-4 transition-transform",
                          advancedOpen && "rotate-180"
                        )}
                      />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-3 px-3 pb-3">
                    <p className="text-xs text-muted-foreground">
                      MedSAM2 uses explicit prompts only. Add positive and
                      negative points directly on the image; no concept text is
                      sent to the model.
                    </p>

                    <div className="space-y-2">
                      <Label>Advanced annotation mode</Label>
                      <ToggleGroup
                        type="single"
                        value={interactionMode}
                        onValueChange={(value) => {
                          if (!value) {
                            return;
                          }
                          setInteractionMode(value === "box" ? "box" : "point");
                        }}
                        variant="outline"
                        size="sm"
                        className="justify-start"
                      >
                        <ToggleGroupItem value="point">Point prompts</ToggleGroupItem>
                        <ToggleGroupItem value="box">
                          <MoveDiagonal2 className="mr-1 size-4" />
                          Draw boxes
                        </ToggleGroupItem>
                      </ToggleGroup>
                      <p className="text-xs text-muted-foreground">
                        Boxes stay available as a fallback, but point prompts
                        remain the default interaction.
                      </p>
                    </div>

                    {interactionMode === "point" ? (
                      <div className="space-y-2">
                        <Label>Point grouping</Label>
                        <ToggleGroup
                          type="single"
                          value={trackerPromptMode}
                          onValueChange={(value) => {
                            if (!value) {
                              return;
                            }
                            setTrackerPromptMode(
                              value === "per_positive_point_instance"
                                ? "per_positive_point_instance"
                                : "single_object_refine"
                            );
                          }}
                          variant="outline"
                          size="sm"
                          className="justify-start"
                        >
                          <ToggleGroupItem value="per_positive_point_instance">
                            Separate instances
                          </ToggleGroupItem>
                          <ToggleGroupItem value="single_object_refine">
                            Refine one object
                          </ToggleGroupItem>
                        </ToggleGroup>
                        <p className="text-xs text-muted-foreground">
                          “Separate instances” treats each positive click as its
                          own object prompt and gives each object a distinct
                          color in the preview. “Refine one object” groups all
                          positive and negative clicks into a single mask.
                        </p>
                      </div>
                    ) : null}
                  </CollapsibleContent>
                </div>
              </Collapsible>
            </CardContent>
          </Card>

          <Card className="min-h-0">
            <CardContent className="space-y-3 p-3">
              {activeFile ? (
                <>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">
                        {activeFile.original_name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Image {activeIndex + 1} of {files.length}
                        {activeViewerState.loading
                          ? " · loading source size…"
                          : ` · ${activeViewerState.sourceSize.width}×${activeViewerState.sourceSize.height} source px`}
                      </p>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={undoLastAnnotation}
                        disabled={
                          busy ||
                          (activeAnnotations.points.length === 0 &&
                            activeAnnotations.boxes.length === 0)
                        }
                      >
                        <Undo2 className="mr-1 size-4" />
                        Undo last
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={deleteSelectedAnnotation}
                        disabled={busy || !selectedAnnotation}
                      >
                        <X className="mr-1 size-4" />
                        Delete selected
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={clearActiveFileAnnotations}
                        disabled={busy}
                      >
                        <Trash2 className="mr-1 size-4" />
                        Clear
                      </Button>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary">
                      Include points: {activeBreakdown.positivePoints}
                    </Badge>
                    <Badge variant="secondary">
                      Exclude points: {activeBreakdown.negativePoints}
                    </Badge>
                    <Badge variant="secondary">
                      Advanced boxes: {activeBreakdown.boxes}
                    </Badge>
                    {selectedDetails ? (
                      <span className="text-xs text-muted-foreground">
                        Selected: {selectedDetails.label} ({selectedDetails.detail})
                      </span>
                    ) : null}
                  </div>

                  <div className="flex min-h-[320px] items-center justify-center rounded-xl bg-muted/25 p-2">
                    <div className="relative inline-block max-w-full select-none">
                      <img
                        ref={imageRef}
                        data-testid="sam-interactive-image"
                        src={activeViewerState.imageUrl}
                        alt={activeFile.original_name}
                        className="block max-h-[56vh] max-w-full rounded-xl border bg-background"
                        onError={() => {
                          setViewerStateByFileId((previous) => ({
                            ...previous,
                            [activeFile.file_id]: {
                              sourceSize:
                                previous[activeFile.file_id]?.sourceSize ??
                                DEFAULT_IMAGE_SIZE,
                              imageUrl:
                                previous[activeFile.file_id]?.imageUrl ??
                                apiClient.uploadPreviewUrl(activeFile.file_id),
                              loading: false,
                              error: "Preview unavailable for this file.",
                            },
                          }));
                        }}
                        draggable={false}
                      />
                      <div
                        data-testid="sam-interactive-overlay"
                        className={cn(
                          "absolute inset-0 rounded-xl",
                          overlayCursorClass
                        )}
                        onClick={handleStageClick}
                        onPointerDown={handleStagePointerDown}
                        onPointerMove={handleStagePointerMove}
                        onPointerUp={handleStagePointerUp}
                        onPointerCancel={(event) => {
                          if (
                            draftBox &&
                            draftBox.pointerId === event.pointerId &&
                            event.currentTarget.hasPointerCapture(event.pointerId)
                          ) {
                            event.currentTarget.releasePointerCapture(event.pointerId);
                          }
                          setDraftBox(null);
                        }}
                      >
                        {activeAnnotations.boxes.map((box) => {
                          const sourceWidth = activeViewerState.sourceSize.width || 1;
                          const sourceHeight = activeViewerState.sourceSize.height || 1;
                          const left = (box.x1 / sourceWidth) * 100;
                          const top = (box.y1 / sourceHeight) * 100;
                          const boxWidth = ((box.x2 - box.x1) / sourceWidth) * 100;
                          const boxHeight = ((box.y2 - box.y1) / sourceHeight) * 100;
                          const selected =
                            selectedAnnotation?.kind === "box" &&
                            selectedAnnotation.id === box.id;
                          return (
                            <button
                              key={box.id}
                              type="button"
                              className={cn(
                                "absolute rounded-sm border-2 bg-transparent",
                                box.label === 1
                                  ? "border-emerald-500/90"
                                  : "border-rose-500/90",
                                selected && "ring-2 ring-background ring-offset-2"
                              )}
                              style={{
                                left: `${left}%`,
                                top: `${top}%`,
                                width: `${boxWidth}%`,
                                height: `${boxHeight}%`,
                              }}
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                setSelectedAnnotation({ kind: "box", id: box.id });
                              }}
                              aria-label="Select box prompt"
                            />
                          );
                        })}

                        {activeAnnotations.points.map((point) => {
                          const sourceWidth = activeViewerState.sourceSize.width || 1;
                          const sourceHeight = activeViewerState.sourceSize.height || 1;
                          const left = (point.x / sourceWidth) * 100;
                          const top = (point.y / sourceHeight) * 100;
                          const selected =
                            selectedAnnotation?.kind === "point" &&
                            selectedAnnotation.id === point.id;
                          const pointIndex =
                            point.label === 1
                              ? activeAnnotations.points
                                  .filter((candidate) => candidate.label === 1)
                                  .findIndex((candidate) => candidate.id === point.id) + 1
                              : activeAnnotations.points
                                  .filter((candidate) => candidate.label === 0)
                                  .findIndex((candidate) => candidate.id === point.id) + 1;
                          return (
                            <button
                              key={point.id}
                              type="button"
                              className={cn(
                                "absolute flex size-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 text-[10px] font-semibold shadow-sm",
                                point.label === 1
                                  ? "border-emerald-500 bg-emerald-500/90 text-emerald-50"
                                  : "border-rose-500 bg-rose-500/90 text-rose-50",
                                selected && "ring-2 ring-background ring-offset-2"
                              )}
                              style={{
                                left: `${left}%`,
                                top: `${top}%`,
                              }}
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                setSelectedAnnotation({ kind: "point", id: point.id });
                              }}
                              aria-label="Select point prompt"
                            >
                              {point.label === 1 ? pointIndex : "−"}
                            </button>
                          );
                        })}

                        {draftBox ? (
                          <div
                            className="pointer-events-none absolute rounded-sm border-2 border-primary/90 border-dashed"
                            style={{
                              left: `${(Math.min(draftBox.x1, draftBox.x2) / activeViewerState.sourceSize.width) * 100}%`,
                              top: `${(Math.min(draftBox.y1, draftBox.y2) / activeViewerState.sourceSize.height) * 100}%`,
                              width: `${(Math.abs(draftBox.x2 - draftBox.x1) / activeViewerState.sourceSize.width) * 100}%`,
                              height: `${(Math.abs(draftBox.y2 - draftBox.y1) / activeViewerState.sourceSize.height) * 100}%`,
                            }}
                          />
                        ) : null}
                      </div>
                    </div>
                  </div>

                  {activeViewerState.error ? (
                    <p className="text-xs text-destructive">
                      {activeViewerState.error}
                    </p>
                  ) : null}
                  {!activeViewerState.error && !canPlacePrompts ? (
                    <p className="text-xs text-muted-foreground">
                      Prompt placement unlocks once the source image dimensions
                      are loaded.
                    </p>
                  ) : null}
                </>
              ) : (
                <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground">
                  No files selected.
                </div>
              )}
            </CardContent>
          </Card>

          {(files.length > 1 || activeFile) && (
            <Card>
              <CardContent className="space-y-2 p-3">
                <p className="text-xs font-medium text-muted-foreground">
                  Per-image summary
                </p>
                <div className="grid gap-1 sm:grid-cols-2">
                  {files.map((file, index) => {
                    const entry = annotationsByFileId[file.file_id] ?? {
                      points: [],
                      boxes: [],
                    };
                    const breakdown = countPromptBreakdown(entry);
                    const active = index === activeIndex;
                    return (
                      <button
                        key={file.file_id}
                        type="button"
                        className={cn(
                          "w-full rounded-md border p-2 text-left text-xs transition",
                          active
                            ? "border-primary/60 bg-primary/10"
                            : "border-border/60 hover:bg-muted/40"
                        )}
                        onClick={() => setActiveIndex(index)}
                        disabled={busy}
                      >
                        <p className="truncate font-medium">{file.original_name}</p>
                        <p className="text-muted-foreground">
                          {breakdown.positivePoints} include ·{" "}
                          {breakdown.negativePoints} exclude · {breakdown.boxes} box
                          {breakdown.boxes === 1 ? "" : "es"}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <DialogFooter className="border-t px-3 pt-3 pb-[calc(0.9rem+env(safe-area-inset-bottom))] sm:px-6 sm:py-4">
          <div className="flex w-full flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setActiveIndex((index) => Math.max(0, index - 1))}
                disabled={busy || activeIndex <= 0}
              >
                <ChevronLeft className="mr-1 size-4" />
                Previous
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() =>
                  setActiveIndex((index) => Math.min(files.length - 1, index + 1))
                }
                disabled={busy || activeIndex >= files.length - 1}
              >
                Next
                <ChevronRight className="ml-1 size-4" />
              </Button>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                onClick={() => onOpenChange(false)}
                disabled={busy}
              >
                Cancel
              </Button>
              <Button
                type="button"
                onClick={() => void submitAnnotations()}
                disabled={busy || !hasSubmittablePrompts}
              >
                {busy
                  ? selectedModelCopy.running
                  : `Finish and run ${selectedModelCopy.label}`}
              </Button>
            </div>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
