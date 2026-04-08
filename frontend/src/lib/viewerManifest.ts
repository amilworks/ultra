import type { Hdf5ViewerTreeNode, UploadViewerInfo } from "@/types";

type ViewerAxis = "z" | "y" | "x";
type UnknownRecord = Record<string, unknown>;

const DEFAULT_AXIS_SIZES: UploadViewerInfo["axis_sizes"] = {
  T: 1,
  C: 1,
  Z: 1,
  Y: 1,
  X: 1,
};

const DEFAULT_CHANNEL_COLORS = ["#ffffff", "#ff0000", "#00ff00", "#0000ff"];

const toRecord = (value: unknown): UnknownRecord =>
  value && typeof value === "object" ? (value as UnknownRecord) : {};

const toFiniteNumber = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const toPositiveInt = (value: unknown, fallback: number): number => {
  const numeric = Math.max(1, Math.round(toFiniteNumber(value, fallback)));
  return Number.isFinite(numeric) ? numeric : Math.max(1, fallback);
};

const clampNonNegativeInt = (value: unknown, fallback: number): number => {
  const numeric = Math.max(0, Math.round(toFiniteNumber(value, fallback)));
  return Number.isFinite(numeric) ? numeric : Math.max(0, fallback);
};

const normalizeDimsOrder = (value: unknown): string => {
  const raw = String(value ?? "").toUpperCase();
  const seen = new Set<string>();
  const ordered = Array.from(raw).filter((axis) => {
    if (!["T", "C", "Z", "Y", "X"].includes(axis) || seen.has(axis)) {
      return false;
    }
    seen.add(axis);
    return true;
  });
  return ordered.length > 0 ? ordered.join("") : "TCZYX";
};

const normalizeAxisSizes = (value: unknown): UploadViewerInfo["axis_sizes"] => {
  const source = toRecord(value);
  return {
    T: toPositiveInt(source.T, 1),
    C: toPositiveInt(source.C, 1),
    Z: toPositiveInt(source.Z, 1),
    Y: toPositiveInt(source.Y, 1),
    X: toPositiveInt(source.X, 1),
  };
};

const normalizeSelectedIndices = (
  value: unknown,
  axisSizes: UploadViewerInfo["axis_sizes"]
): UploadViewerInfo["selected_indices"] => {
  const source = toRecord(value);
  return {
    T: Math.min(axisSizes.T - 1, clampNonNegativeInt(source.T, 0)),
    C: Math.min(axisSizes.C - 1, clampNonNegativeInt(source.C, 0)),
    Z: Math.min(axisSizes.Z - 1, clampNonNegativeInt(source.Z, 0)),
  };
};

const normalizePhysicalSpacing = (
  metadata: UnknownRecord,
  source: UnknownRecord
): NonNullable<UploadViewerInfo["metadata"]["physical_spacing"]> | null => {
  const spacing = toRecord(metadata.physical_spacing ?? source.physical_spacing);
  if (Object.keys(spacing).length === 0) {
    return null;
  }
  const z = toFiniteNumber(spacing.z, NaN);
  const y = toFiniteNumber(spacing.y, NaN);
  const x = toFiniteNumber(spacing.x, NaN);
  return {
    z: Number.isFinite(z) && z > 0 ? z : null,
    y: Number.isFinite(y) && y > 0 ? y : null,
    x: Number.isFinite(x) && x > 0 ? x : null,
  };
};

const buildPlaneDescriptor = (
  axis: ViewerAxis,
  axisSizes: UploadViewerInfo["axis_sizes"],
  spacing: NonNullable<UploadViewerInfo["metadata"]["physical_spacing"]> | null
): UploadViewerInfo["viewer"]["default_plane"] => {
  const zSpacing = Math.max(1, Number(spacing?.z ?? 1));
  const ySpacing = Math.max(1, Number(spacing?.y ?? 1));
  const xSpacing = Math.max(1, Number(spacing?.x ?? 1));

  if (axis === "x") {
    const width = Math.max(1, axisSizes.Y);
    const height = Math.max(1, axisSizes.Z);
    const worldWidth = width * ySpacing;
    const worldHeight = height * zSpacing;
    return {
      axis,
      label: "YZ plane",
      axes: ["Z", "Y"],
      pixel_size: { width, height },
      spacing: { row: zSpacing, col: ySpacing },
      world_size: { width: worldWidth, height: worldHeight },
      aspect_ratio: worldWidth / Math.max(1e-9, worldHeight),
    };
  }
  if (axis === "y") {
    const width = Math.max(1, axisSizes.X);
    const height = Math.max(1, axisSizes.Z);
    const worldWidth = width * xSpacing;
    const worldHeight = height * zSpacing;
    return {
      axis,
      label: "XZ plane",
      axes: ["Z", "X"],
      pixel_size: { width, height },
      spacing: { row: zSpacing, col: xSpacing },
      world_size: { width: worldWidth, height: worldHeight },
      aspect_ratio: worldWidth / Math.max(1e-9, worldHeight),
    };
  }

  const width = Math.max(1, axisSizes.X);
  const height = Math.max(1, axisSizes.Y);
  const worldWidth = width * xSpacing;
  const worldHeight = height * ySpacing;
  return {
    axis: "z",
    label: "XY plane",
    axes: ["Y", "X"],
    pixel_size: { width, height },
    spacing: { row: ySpacing, col: xSpacing },
    world_size: { width: worldWidth, height: worldHeight },
    aspect_ratio: worldWidth / Math.max(1e-9, worldHeight),
  };
};

const normalizePlaneDescriptor = (
  value: unknown,
  axis: ViewerAxis,
  axisSizes: UploadViewerInfo["axis_sizes"],
  spacing: NonNullable<UploadViewerInfo["metadata"]["physical_spacing"]> | null
): UploadViewerInfo["viewer"]["default_plane"] => {
  const fallback = buildPlaneDescriptor(axis, axisSizes, spacing);
  const source = toRecord(value);
  const pixel = toRecord(source.pixel_size);
  const planeSpacing = toRecord(source.spacing);
  const world = toRecord(source.world_size);

  const pixelWidth = toPositiveInt(pixel.width, fallback.pixel_size.width);
  const pixelHeight = toPositiveInt(pixel.height, fallback.pixel_size.height);
  const rowSpacing = Math.max(1e-9, toFiniteNumber(planeSpacing.row, fallback.spacing.row));
  const colSpacing = Math.max(1e-9, toFiniteNumber(planeSpacing.col, fallback.spacing.col));
  const worldWidth = Math.max(1e-9, toFiniteNumber(world.width, pixelWidth * colSpacing));
  const worldHeight = Math.max(1e-9, toFiniteNumber(world.height, pixelHeight * rowSpacing));

  return {
    axis,
    label: String(source.label ?? fallback.label),
    axes: Array.isArray(source.axes) ? source.axes.map((item) => String(item)) : fallback.axes,
    pixel_size: {
      width: pixelWidth,
      height: pixelHeight,
    },
    spacing: {
      row: rowSpacing,
      col: colSpacing,
    },
    world_size: {
      width: worldWidth,
      height: worldHeight,
    },
    aspect_ratio: Math.max(1e-9, toFiniteNumber(source.aspect_ratio, worldWidth / worldHeight)),
  };
};

const buildTileLevels = (
  width: number,
  height: number,
  tileSize: number
): UploadViewerInfo["viewer"]["tile_scheme"]["levels"] => {
  const safeTileSize = Math.max(64, tileSize);
  const levels: UploadViewerInfo["viewer"]["tile_scheme"]["levels"] = [];
  let currentWidth = Math.max(1, width);
  let currentHeight = Math.max(1, height);
  let downsample = 1;

  while (true) {
    levels.unshift({
      level: levels.length,
      width: currentWidth,
      height: currentHeight,
      columns: Math.max(1, Math.ceil(currentWidth / safeTileSize)),
      rows: Math.max(1, Math.ceil(currentHeight / safeTileSize)),
      downsample,
    });
    if (currentWidth <= safeTileSize && currentHeight <= safeTileSize) {
      break;
    }
    currentWidth = Math.max(1, Math.ceil(currentWidth / 2));
    currentHeight = Math.max(1, Math.ceil(currentHeight / 2));
    downsample *= 2;
  }

  return levels.map((level, index) => ({ ...level, level: index }));
};

const buildAtlasScheme = (
  axisSizes: UploadViewerInfo["axis_sizes"],
  defaultPlane: UploadViewerInfo["viewer"]["default_plane"]
): NonNullable<UploadViewerInfo["viewer"]["atlas_scheme"]> => {
  const sliceCount = Math.max(1, axisSizes.Z);
  const baseWidth = Math.max(1, defaultPlane.pixel_size.width);
  const baseHeight = Math.max(1, defaultPlane.pixel_size.height);
  const columns = Math.max(1, Math.ceil(Math.sqrt(sliceCount)));
  const rows = Math.max(1, Math.ceil(sliceCount / columns));
  return {
    slice_count: sliceCount,
    columns,
    rows,
    slice_width: baseWidth,
    slice_height: baseHeight,
    atlas_width: baseWidth * columns,
    atlas_height: baseHeight * rows,
    downsample: 1,
    format: "png",
  };
};

const inferModality = (source: UnknownRecord, originalName: string): UploadViewerInfo["modality"] => {
  const explicit = String(source.modality ?? "").trim().toLowerCase();
  if (explicit) {
    return explicit;
  }
  const reader = String(source.reader ?? "").toLowerCase();
  const lowerName = originalName.toLowerCase();
  const microscopyMetadata = toRecord(toRecord(source.metadata).microscopy);
  if (lowerName.endsWith(".nii") || lowerName.endsWith(".nii.gz") || reader.includes("nibabel")) {
    return "medical";
  }
  if (lowerName.endsWith(".png") || lowerName.endsWith(".jpg") || lowerName.endsWith(".jpeg") || lowerName.endsWith(".webp") || lowerName.endsWith(".bmp") || lowerName.endsWith(".gif")) {
    return "image";
  }
  if (reader.includes("bioio")) {
    if (Object.keys(microscopyMetadata).length > 0) {
      return "microscopy";
    }
    if (
      lowerName.endsWith(".ome.tif") ||
      lowerName.endsWith(".ome.tiff") ||
      lowerName.endsWith(".ome.zarr") ||
      lowerName.endsWith(".czi") ||
      lowerName.endsWith(".nd2") ||
      lowerName.endsWith(".lif") ||
      lowerName.endsWith(".dv") ||
      lowerName.endsWith(".tif") ||
      lowerName.endsWith(".tiff")
    ) {
      return "microscopy";
    }
    return "image";
  }
  return "image";
};

const normalizeServiceUrls = (source: UnknownRecord, fileId: string) => {
  const fileSegment = encodeURIComponent(fileId);
  return {
    preview: String(source.preview ?? `/v1/uploads/${fileSegment}/preview`),
    display: source.display == null ? undefined : String(source.display),
    slice: String(source.slice ?? `/v1/uploads/${fileSegment}/slice`),
    tile: String(source.tile ?? `/v1/uploads/${fileSegment}/tiles`),
    atlas: String(source.atlas ?? `/v1/uploads/${fileSegment}/atlas`),
    scalar_volume:
      source.scalar_volume == null ? undefined : String(source.scalar_volume),
    histogram: String(source.histogram ?? `/v1/uploads/${fileSegment}/histogram`),
  };
};

const normalizeHdf5ServiceUrls = (source: UnknownRecord, fileId: string) => {
  const fileSegment = encodeURIComponent(fileId);
  return {
    dataset: String(source.dataset ?? `/v1/uploads/${fileSegment}/hdf5/dataset`),
    slice: String(source.slice ?? `/v1/uploads/${fileSegment}/hdf5/preview/slice`),
    atlas: String(source.atlas ?? `/v1/uploads/${fileSegment}/hdf5/preview/atlas`),
    histogram: String(source.histogram ?? `/v1/uploads/${fileSegment}/hdf5/preview/histogram`),
    table: String(source.table ?? `/v1/uploads/${fileSegment}/hdf5/preview/table`),
  };
};

const normalizeHdf5Tree = (value: unknown): Hdf5ViewerTreeNode[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((entry, index) => {
    const source = toRecord(entry);
    return {
      path: String(source.path ?? `/${index}`),
      name: String(source.name ?? source.path ?? `node-${index}`),
      node_type: String(source.node_type ?? "group"),
      child_count: clampNonNegativeInt(source.child_count, 0),
      attributes_count: clampNonNegativeInt(source.attributes_count, 0),
      shape: Array.isArray(source.shape)
        ? source.shape.map((item) => clampNonNegativeInt(item, 0))
        : null,
      dtype: source.dtype == null ? null : String(source.dtype),
      preview_kind: source.preview_kind == null ? null : String(source.preview_kind),
      children: normalizeHdf5Tree(source.children),
    };
  });
};

const normalizePhys = (
  source: UnknownRecord,
  metadataSource: UnknownRecord,
  axisSizes: UploadViewerInfo["axis_sizes"],
  fileId: string,
  originalName: string,
  modality: string
): NonNullable<UploadViewerInfo["phys"]> => {
  const physSource = toRecord(source.phys);
  const channelCount = Math.max(1, axisSizes.C);
  const channelColors = Array.isArray(physSource.channel_colors)
    ? physSource.channel_colors.map((item, index) => {
      const entry = toRecord(item);
      const rgb = Array.isArray(entry.rgb) ? entry.rgb.map((value) => clampNonNegativeInt(value, 255)).slice(0, 3) : [255, 255, 255];
      return {
        index: clampNonNegativeInt(entry.index, index),
        hex: String(entry.hex ?? DEFAULT_CHANNEL_COLORS[index] ?? "#ffffff"),
        rgb: rgb.length === 3 ? rgb : [255, 255, 255],
      };
    })
    : Array.from({ length: channelCount }, (_, index) => ({
      index,
      hex: DEFAULT_CHANNEL_COLORS[index] ?? "#ffffff",
      rgb: DEFAULT_CHANNEL_COLORS[index]
        ? [
          parseInt(DEFAULT_CHANNEL_COLORS[index].slice(1, 3), 16),
          parseInt(DEFAULT_CHANNEL_COLORS[index].slice(3, 5), 16),
          parseInt(DEFAULT_CHANNEL_COLORS[index].slice(5, 7), 16),
        ]
        : [255, 255, 255],
    }));
  const dicom = toRecord(physSource.dicom ?? metadataSource.dicom);
  return {
    resource_uniq: String(physSource.resource_uniq ?? fileId),
    name: String(physSource.name ?? originalName),
    x: toPositiveInt(physSource.x, axisSizes.X),
    y: toPositiveInt(physSource.y, axisSizes.Y),
    z: toPositiveInt(physSource.z, axisSizes.Z),
    t: toPositiveInt(physSource.t, axisSizes.T),
    ch: toPositiveInt(physSource.ch, axisSizes.C),
    pixel_depth: toPositiveInt(physSource.pixel_depth, 8),
    pixel_format: String(physSource.pixel_format ?? "u"),
    pixel_size: Array.isArray(physSource.pixel_size)
      ? physSource.pixel_size.map((value) => toFiniteNumber(value, 1))
      : [1, 1, 1, 1],
    pixel_units: Array.isArray(physSource.pixel_units)
      ? physSource.pixel_units.map((value) => String(value))
      : ["px", "px", "px", "frame"],
    channel_names: Array.isArray(physSource.channel_names)
      ? physSource.channel_names.map((value) => String(value))
      : Array.from({ length: channelCount }, (_, index) => (channelCount === 1 ? "Intensity" : `Ch${index + 1}`)),
    display_channels: Array.isArray(physSource.display_channels)
      ? physSource.display_channels.map((value) => Math.round(Number(value) || 0))
      : channelCount === 1
        ? [0, 0, 0]
        : channelCount === 2
          ? [0, 1, -1]
          : [0, 1, 2],
    channel_colors: channelColors,
    units: String(physSource.units ?? (modality === "medical" ? "physical" : "pixel")),
    dicom: {
      modality: dicom.modality == null ? null : String(dicom.modality),
      wnd_center: dicom.wnd_center == null ? null : toFiniteNumber(dicom.wnd_center, 0),
      wnd_width: dicom.wnd_width == null ? null : toFiniteNumber(dicom.wnd_width, 0),
    },
    geo: Object.keys(toRecord(physSource.geo ?? metadataSource.geo)).length > 0 ? toRecord(physSource.geo ?? metadataSource.geo) : null,
    coordinates:
      Object.keys(toRecord(physSource.coordinates)).length > 0 ? toRecord(physSource.coordinates) : null,
  };
};

const normalizeDisplayDefaults = (
  source: UnknownRecord,
  viewerSource: UnknownRecord,
  phys: NonNullable<UploadViewerInfo["phys"]>,
  selectedIndices: UploadViewerInfo["selected_indices"]
): NonNullable<UploadViewerInfo["display_defaults"]> => {
  const defaultsSource = toRecord(source.display_defaults);
  const viewerDefaults = toRecord(viewerSource.display_defaults);
  const merged = { ...viewerDefaults, ...defaultsSource };
  return {
    enhancement: String(merged.enhancement ?? "d"),
    negative: Boolean(merged.negative ?? false),
    rotate: Math.round(toFiniteNumber(merged.rotate, 0)),
    fusion_method: String(merged.fusion_method ?? "m"),
    channel_mode: String(merged.channel_mode ?? "composite"),
    channels: Array.isArray(merged.channels)
      ? merged.channels.map((value) => clampNonNegativeInt(value, 0))
      : (phys.display_channels ?? [0, 1, 2]).filter((value) => value >= 0),
    channel_colors: Array.isArray(merged.channel_colors)
      ? merged.channel_colors.map((value) => String(value))
      : (phys.channel_colors ?? []).map((entry) => entry.hex),
    time_index: clampNonNegativeInt(merged.time_index, selectedIndices.T),
    z_index: clampNonNegativeInt(merged.z_index, selectedIndices.Z),
    volume_channel:
      merged.volume_channel == null ? clampNonNegativeInt(selectedIndices.C, 0) : clampNonNegativeInt(merged.volume_channel, selectedIndices.C),
  };
};

const hasPositiveSpacing = (
  spacing: NonNullable<UploadViewerInfo["metadata"]["physical_spacing"]> | null
): boolean =>
  Boolean(
    spacing &&
      ["z", "y", "x"].some((axis) => {
        const numeric = Number(spacing[axis as keyof typeof spacing]);
        return Number.isFinite(numeric) && numeric > 0;
      })
  );

const normalizeMeasurementPolicy = (
  value: unknown,
  options: {
    orientationFrame: string;
    physicalSpacing: NonNullable<UploadViewerInfo["metadata"]["physical_spacing"]> | null;
  }
): "pixel-only" | "spacing-aware" | "orientation-aware" => {
  const explicit = String(value ?? "").trim().toLowerCase();
  if (explicit === "pixel-only" || explicit === "spacing-aware" || explicit === "orientation-aware") {
    return explicit;
  }
  const safeFrame = String(options.orientationFrame ?? "").trim().toLowerCase();
  if (hasPositiveSpacing(options.physicalSpacing) && (safeFrame === "patient" || safeFrame === "geospatial")) {
    return "orientation-aware";
  }
  if (hasPositiveSpacing(options.physicalSpacing)) {
    return "spacing-aware";
  }
  return "pixel-only";
};

const buildOrientationLabels = (
  rowAxis: string,
  colAxis: string,
  sliceAxis: string | null
): NonNullable<NonNullable<UploadViewerInfo["viewer"]["orientation"]>["labels"]> => ({
  top: ["R", "L", "A", "P", "S", "I", "H", "F"].includes(String(rowAxis || "").toUpperCase())
    ? ({ R: "L", L: "R", A: "P", P: "A", S: "I", I: "S", H: "F", F: "H" } as Record<string, string>)[String(rowAxis || "").toUpperCase()]
    : `-${String(rowAxis || "Y").toUpperCase()}`,
  bottom: ["R", "L", "A", "P", "S", "I", "H", "F"].includes(String(rowAxis || "").toUpperCase())
    ? String(rowAxis || "Y").toUpperCase()
    : `+${String(rowAxis || "Y").toUpperCase()}`,
  left: ["R", "L", "A", "P", "S", "I", "H", "F"].includes(String(colAxis || "").toUpperCase())
    ? ({ R: "L", L: "R", A: "P", P: "A", S: "I", I: "S", H: "F", F: "H" } as Record<string, string>)[String(colAxis || "").toUpperCase()]
    : `-${String(colAxis || "X").toUpperCase()}`,
  right: ["R", "L", "A", "P", "S", "I", "H", "F"].includes(String(colAxis || "").toUpperCase())
    ? String(colAxis || "X").toUpperCase()
    : `+${String(colAxis || "X").toUpperCase()}`,
  front: sliceAxis
    ? (["R", "L", "A", "P", "S", "I", "H", "F"].includes(String(sliceAxis).toUpperCase())
      ? ({ R: "L", L: "R", A: "P", P: "A", S: "I", I: "S", H: "F", F: "H" } as Record<string, string>)[String(sliceAxis).toUpperCase()]
      : `-${String(sliceAxis).toUpperCase()}`)
    : null,
  back: sliceAxis
    ? (["R", "L", "A", "P", "S", "I", "H", "F"].includes(String(sliceAxis).toUpperCase())
      ? String(sliceAxis).toUpperCase()
      : `+${String(sliceAxis).toUpperCase()}`)
    : null,
});

const normalizeOrientationAxisLabels = (
  value: unknown
): NonNullable<NonNullable<UploadViewerInfo["viewer"]["orientation"]>["axis_labels"]> => {
  const source = toRecord(value);
  const normalizeEntry = (axis: "x" | "y" | "z", fallback: string) => {
    const entry = toRecord(source[axis]);
    const positive = String(entry.positive ?? fallback).trim().toUpperCase() || fallback;
    const negative = String(entry.negative ?? `-${positive}`).trim().toUpperCase() || `-${positive}`;
    return { positive, negative };
  };
  return {
    x: normalizeEntry("x", "X"),
    y: normalizeEntry("y", "Y"),
    z: normalizeEntry("z", "Z"),
  };
};

const inferImageRenderPolicy = (
  source: UnknownRecord,
  options: {
    modality: string;
    axisSizes: UploadViewerInfo["axis_sizes"];
  }
): "scalar" | "categorical" | "display" | "analysis" => {
  const explicit = String(toRecord(source.viewer).render_policy ?? "").trim().toLowerCase();
  if (explicit === "scalar" || explicit === "categorical" || explicit === "display" || explicit === "analysis") {
    return explicit;
  }
  const semanticKind = String(source.semantic_kind ?? "").trim().toLowerCase();
  if (semanticKind === "label") {
    return "categorical";
  }
  if (semanticKind === "display" || semanticKind === "rgb") {
    return "display";
  }
  if (semanticKind === "vector" || semanticKind === "analysis") {
    return "analysis";
  }
  const isVolume = Boolean(source.is_volume) || options.axisSizes.Z > 1;
  const dtypeName = String(source.array_dtype ?? toRecord(source.metadata).array_dtype ?? "uint8").toLowerCase();
  if (!isVolume && options.axisSizes.C >= 3 && options.axisSizes.C <= 4 && !dtypeName.includes("float")) {
    return "display";
  }
  if (options.modality === "medical" || options.modality === "microscopy") {
    return "scalar";
  }
  if (options.axisSizes.C === 1) {
    return "scalar";
  }
  return options.axisSizes.C <= 4 ? "display" : "analysis";
};

const inferHdf5RenderPolicy = (
  previewKind: string | null | undefined
): "scalar" | "categorical" | "display" | "analysis" => {
  const safeKind = String(previewKind ?? "").trim().toLowerCase();
  if (safeKind === "scalar_volume") {
    return "scalar";
  }
  if (safeKind === "label_volume") {
    return "categorical";
  }
  if (safeKind === "rgb_volume") {
    return "display";
  }
  return "analysis";
};

const normalizeRenderPolicy = (
  value: unknown,
  fallback: "scalar" | "categorical" | "display" | "analysis"
): "scalar" | "categorical" | "display" | "analysis" => {
  const explicit = String(value ?? "").trim().toLowerCase();
  if (explicit === "scalar" || explicit === "categorical" || explicit === "display" || explicit === "analysis") {
    return explicit;
  }
  return fallback;
};

const normalizeDiagnosticSurface = (
  value: unknown,
  fallback: "mpr" | "none"
): "mpr" | "none" => {
  const explicit = String(value ?? "").trim().toLowerCase();
  if (explicit === "mpr" || explicit === "none") {
    return explicit;
  }
  return fallback;
};

const normalizeDisplayCapabilities = (value: unknown, fallback: string[]): string[] => {
  if (Array.isArray(value)) {
    const deduped = value.map((item) => String(item).trim()).filter(Boolean);
    return Array.from(new Set(deduped));
  }
  return Array.from(new Set(fallback.map((item) => String(item).trim()).filter(Boolean)));
};

const normalizeDeliveryMode = (
  value: unknown,
  fallback: "direct" | "scalar" | "atlas" | "deferred_multiscale"
): "direct" | "scalar" | "atlas" | "deferred_multiscale" => {
  const explicit = String(value ?? "").trim().toLowerCase();
  if (explicit === "direct" || explicit === "scalar" || explicit === "atlas" || explicit === "deferred_multiscale") {
    return explicit;
  }
  return fallback;
};

const inferTexturePolicy = (
  renderPolicy: "scalar" | "categorical" | "display" | "analysis"
): "linear" | "nearest" => (renderPolicy === "categorical" || renderPolicy === "analysis" ? "nearest" : "linear");

const normalizeFirstPaintMode = (
  value: unknown,
  fallback: "image" | "webgl"
): "image" | "webgl" => {
  const explicit = String(value ?? "").trim().toLowerCase();
  if (explicit === "image" || explicit === "webgl") {
    return explicit;
  }
  return fallback;
};

const normalizeTexturePolicy = (
  value: unknown,
  fallback: "linear" | "nearest"
): "linear" | "nearest" => {
  const explicit = String(value ?? "").trim().toLowerCase();
  if (explicit === "linear" || explicit === "nearest") {
    return explicit;
  }
  return fallback;
};

const normalizeViewerCapabilities = (value: unknown, fallback: string[]): string[] => {
  if (Array.isArray(value)) {
    return Array.from(new Set(value.map((item) => String(item).trim()).filter(Boolean)));
  }
  return Array.from(new Set(fallback.map((item) => String(item).trim()).filter(Boolean)));
};

const normalizeHdf5ViewerInfo = (source: UnknownRecord): UploadViewerInfo => {
  const metadataSource = toRecord(source.metadata);
  const viewerSource = toRecord(source.viewer);
  const hdf5Source = toRecord(source.hdf5);
  const fileId = String(source.file_id ?? "");
  const originalName = String(source.original_name ?? "resource");
  const modality = String(source.modality ?? "unknown");
  const axisSizes = normalizeAxisSizes(source.axis_sizes ?? viewerSource.axis_sizes ?? DEFAULT_AXIS_SIZES);
  const selectedIndices = normalizeSelectedIndices(source.selected_indices ?? viewerSource.selected_indices, axisSizes);
  const physicalSpacing = normalizePhysicalSpacing(metadataSource, source);
  const hdf5ServiceUrls = normalizeHdf5ServiceUrls(
    { ...toRecord(viewerSource.service_urls), ...toRecord(source.service_urls) },
    fileId
  );
  const sliceAxes: ViewerAxis[] = Array.isArray(viewerSource.slice_axes)
    ? viewerSource.slice_axes
      .map((item) => String(item).toLowerCase())
      .filter((axis): axis is ViewerAxis => axis === "z" || axis === "y" || axis === "x")
    : ["z"];
  const defaultAxisCandidate = String(viewerSource.default_axis ?? sliceAxes[0] ?? "z").toLowerCase();
  const defaultAxis: ViewerAxis =
    defaultAxisCandidate === "x" || defaultAxisCandidate === "y" || defaultAxisCandidate === "z"
      ? defaultAxisCandidate
      : "z";
  const defaultPlane = normalizePlaneDescriptor(
    viewerSource.default_plane,
    defaultAxis,
    axisSizes,
    physicalSpacing
  );
  const planesSource = toRecord(viewerSource.planes);
  const planes = Object.fromEntries(
    sliceAxes.map((axis) => [
      axis,
      normalizePlaneDescriptor(planesSource[axis], axis, axisSizes, physicalSpacing),
    ])
  );
  const atlasSource = toRecord(viewerSource.atlas_scheme);
  const atlasScheme =
    Object.keys(atlasSource).length > 0
      ? {
        slice_count: toPositiveInt(atlasSource.slice_count, Math.max(1, axisSizes.Z)),
        columns: toPositiveInt(atlasSource.columns, 1),
        rows: toPositiveInt(atlasSource.rows, 1),
        slice_width: toPositiveInt(atlasSource.slice_width, defaultPlane.pixel_size.width),
        slice_height: toPositiveInt(atlasSource.slice_height, defaultPlane.pixel_size.height),
        atlas_width: toPositiveInt(atlasSource.atlas_width, defaultPlane.pixel_size.width),
        atlas_height: toPositiveInt(atlasSource.atlas_height, defaultPlane.pixel_size.height),
        downsample: Math.max(1, toFiniteNumber(atlasSource.downsample, 1)),
        format: String(atlasSource.format ?? "png"),
      }
      : undefined;
  const availableSurfaces = Array.isArray(viewerSource.available_surfaces)
    ? viewerSource.available_surfaces.map((item) => String(item))
    : [String(viewerSource.default_surface ?? "metadata")];
  const defaultSurfaceCandidate = String(viewerSource.default_surface ?? availableSurfaces[0] ?? "metadata");
  const defaultSurface = availableSurfaces.includes(defaultSurfaceCandidate)
    ? defaultSurfaceCandidate
    : (availableSurfaces[0] ?? "metadata");
  const volumeMode = String(viewerSource.volume_mode ?? "none");
  const orientationFrame = String(toRecord(viewerSource.orientation).frame ?? "voxel");
  const rowAxis = String(toRecord(viewerSource.orientation).row_axis ?? defaultPlane.axes[0] ?? "Y");
  const colAxis = String(toRecord(viewerSource.orientation).col_axis ?? defaultPlane.axes[1] ?? "X");
  const sliceAxis =
    toRecord(viewerSource.orientation).slice_axis == null
      ? null
      : String(toRecord(viewerSource.orientation).slice_axis);
  const axisLabels = normalizeOrientationAxisLabels(toRecord(viewerSource.orientation).axis_labels);
  const renderPolicy = normalizeRenderPolicy(
    viewerSource.render_policy,
    "analysis"
  );
  const measurementPolicy = normalizeMeasurementPolicy(viewerSource.measurement_policy, {
    orientationFrame,
    physicalSpacing,
  });
  const diagnosticSurface = normalizeDiagnosticSurface(viewerSource.diagnostic_surface, "none");
  const displayCapabilities = normalizeDisplayCapabilities(
    viewerSource.display_capabilities,
    toRecord(hdf5Source.materials).detected ? ["dataset_explorer", "materials_dashboard"] : ["dataset_explorer"]
  );
  const deliveryMode = normalizeDeliveryMode(
    viewerSource.delivery_mode,
    volumeMode === "scalar" ? "scalar" : volumeMode === "atlas" ? "atlas" : "direct"
  );
  const firstPaintMode = normalizeFirstPaintMode(
    viewerSource.first_paint_mode,
    defaultSurface === "volume" ? "webgl" : "image"
  );
  const texturePolicy = normalizeTexturePolicy(
    viewerSource.texture_policy,
    inferTexturePolicy(renderPolicy)
  );
  const viewerCapabilities = normalizeViewerCapabilities(
    viewerSource.viewer_capabilities,
    [
      firstPaintMode === "webgl" ? "webgl_first_paint" : "image_first_paint",
      deliveryMode === "scalar"
        ? "scalar_volume_delivery"
        : deliveryMode === "atlas"
          ? "atlas_volume_delivery"
          : deliveryMode === "deferred_multiscale"
            ? "deferred_multiscale"
            : "direct_delivery",
      texturePolicy === "nearest" ? "nearest_sampling" : "linear_sampling",
      ...(diagnosticSurface === "mpr" ? ["mpr_truth_surface"] : []),
      ...displayCapabilities,
    ]
  );
  const hasPhysSource = Object.keys(toRecord(source.phys)).length > 0;
  const hasDisplayDefaults =
    Object.keys(toRecord(source.display_defaults)).length > 0 ||
    Object.keys(toRecord(viewerSource.display_defaults)).length > 0;
  const phys =
    hasPhysSource || hasDisplayDefaults
      ? normalizePhys(source, metadataSource, axisSizes, fileId, originalName, modality)
      : undefined;
  const displayDefaults =
    hasDisplayDefaults && phys
      ? normalizeDisplayDefaults(source, viewerSource, phys, selectedIndices)
      : undefined;
  const warningsSource = Array.isArray(metadataSource.warnings)
    ? metadataSource.warnings
    : Array.isArray(source.warnings)
      ? source.warnings
      : [];

  return {
    kind: "hdf5",
    file_id: fileId,
    original_name: originalName,
    modality,
    backend_mode: String(source.backend_mode ?? viewerSource.backend_mode ?? "hdf5"),
    dims_order: String(source.dims_order ?? ""),
    axis_sizes: axisSizes,
    selected_indices: selectedIndices,
    is_volume:
      Boolean(source.is_volume) ||
      defaultSurface === "volume" ||
      availableSurfaces.includes("volume") ||
      String(volumeMode).toLowerCase() !== "none",
    is_timeseries: false,
    is_multichannel: Boolean(source.is_multichannel) || axisSizes.C > 1,
    phys,
    display_defaults: displayDefaults,
    service_urls: hdf5ServiceUrls,
    metadata: {
      reader: String(metadataSource.reader ?? "h5py"),
      dims_order: String(metadataSource.dims_order ?? ""),
      array_shape: Array.isArray(metadataSource.array_shape)
        ? metadataSource.array_shape.map((item) => clampNonNegativeInt(item, 0))
        : [],
      array_dtype: String(metadataSource.array_dtype ?? "hdf5"),
      scene: metadataSource.scene == null ? null : String(metadataSource.scene),
      scene_count: toPositiveInt(metadataSource.scene_count ?? 1, 1),
      header: Object.fromEntries(
        Object.entries(toRecord(metadataSource.header)).map(([key, value]) => [key, String(value)])
      ),
      filename_hints: toRecord(metadataSource.filename_hints),
      physical_spacing: physicalSpacing,
      exif: {},
      geo: null,
      dicom: null,
      microscopy: null,
      warnings: warningsSource.map((item: unknown) => String(item)),
    },
    viewer: {
      status: String(
        viewerSource.status ?? (Boolean(hdf5Source.supported ?? true) ? "ready" : "degraded-fallback")
      ),
      warmup_mode: String(viewerSource.warmup_mode ?? "lazy"),
      backend_mode: String(viewerSource.backend_mode ?? source.backend_mode ?? "hdf5"),
      default_surface: defaultSurface,
      available_surfaces: availableSurfaces,
      default_axis: defaultAxis,
      slice_axes: sliceAxes,
      channel_mode: String(viewerSource.channel_mode ?? "single"),
      tile_scheme: {
        tile_size: toPositiveInt(toRecord(viewerSource.tile_scheme).tile_size, 256),
        format: String(toRecord(viewerSource.tile_scheme).format ?? "png"),
        levels: Array.isArray(toRecord(viewerSource.tile_scheme).levels)
          ? (toRecord(viewerSource.tile_scheme).levels as Array<unknown>).map((entry, index) => {
            const levelSource = toRecord(entry);
            return {
              level: clampNonNegativeInt(levelSource.level, index),
              width: toPositiveInt(levelSource.width, defaultPlane.pixel_size.width),
              height: toPositiveInt(levelSource.height, defaultPlane.pixel_size.height),
              columns: toPositiveInt(levelSource.columns, 1),
              rows: toPositiveInt(levelSource.rows, 1),
              downsample: Math.max(1, toFiniteNumber(levelSource.downsample, 1)),
            };
          })
          : buildTileLevels(defaultPlane.pixel_size.width, defaultPlane.pixel_size.height, 256),
      },
      atlas_scheme: atlasScheme,
      default_plane: defaultPlane,
      planes,
      volume_mode: volumeMode,
      render_policy: renderPolicy,
      delivery_mode: deliveryMode,
      diagnostic_surface: diagnosticSurface,
      first_paint_mode: firstPaintMode,
      measurement_policy: measurementPolicy,
      texture_policy: texturePolicy,
      display_capabilities: displayCapabilities,
      viewer_capabilities: viewerCapabilities,
      orientation: {
        frame: orientationFrame,
        row_axis: rowAxis,
        col_axis: colAxis,
        slice_axis: sliceAxis,
        axis_labels: axisLabels,
        labels: buildOrientationLabels(rowAxis, colAxis, sliceAxis),
      },
      asset_preparation: {
        status: String(
          toRecord(viewerSource.asset_preparation).status ??
            (Boolean(hdf5Source.supported ?? true) ? "ready" : "degraded-fallback")
        ),
        native_supported: Boolean(
          toRecord(viewerSource.asset_preparation).native_supported ?? Boolean(hdf5Source.supported ?? true)
        ),
        tile_pyramid: String(toRecord(viewerSource.asset_preparation).tile_pyramid ?? "none"),
        volume_representation: String(
          toRecord(viewerSource.asset_preparation).volume_representation ?? "none"
        ),
      },
      chunk_scheme: {
        mode: String(toRecord(viewerSource.chunk_scheme).mode ?? "none"),
        axis: String(toRecord(viewerSource.chunk_scheme).axis ?? "z") as "z" | "y" | "x",
        sample_count: toPositiveInt(toRecord(viewerSource.chunk_scheme).sample_count, 1),
      },
      display_defaults: displayDefaults,
      service_urls: hdf5ServiceUrls,
      fallback_urls: {
        preview:
          toRecord(viewerSource.fallback_urls).preview == null
            ? undefined
            : String(toRecord(viewerSource.fallback_urls).preview),
        slice:
          toRecord(viewerSource.fallback_urls).slice == null
            ? undefined
            : String(toRecord(viewerSource.fallback_urls).slice),
      },
    },
    hdf5: {
      enabled: Boolean(hdf5Source.enabled ?? true),
      supported: Boolean(hdf5Source.supported ?? true),
      status: String(hdf5Source.status ?? (Boolean(hdf5Source.supported ?? true) ? "ready" : "unsupported")),
      error: hdf5Source.error == null ? null : String(hdf5Source.error),
      root_keys: Array.isArray(hdf5Source.root_keys) ? hdf5Source.root_keys.map((item) => String(item)) : [],
      root_attributes: toRecord(hdf5Source.root_attributes),
      summary: {
        group_count: clampNonNegativeInt(toRecord(hdf5Source.summary).group_count, 0),
        dataset_count: clampNonNegativeInt(toRecord(hdf5Source.summary).dataset_count, 0),
        dataset_kinds: Object.fromEntries(
          Object.entries(toRecord(toRecord(hdf5Source.summary).dataset_kinds)).map(([key, value]) => [
            String(key),
            clampNonNegativeInt(value, 0),
          ])
        ),
        truncated: Boolean(toRecord(hdf5Source.summary).truncated ?? false),
        geometry:
          Object.keys(toRecord(toRecord(hdf5Source.summary).geometry)).length > 0
            ? {
              path:
                toRecord(toRecord(hdf5Source.summary).geometry).path == null
                  ? null
                  : String(toRecord(toRecord(hdf5Source.summary).geometry).path),
              dimensions: Array.isArray(toRecord(toRecord(hdf5Source.summary).geometry).dimensions)
                ? (toRecord(toRecord(hdf5Source.summary).geometry).dimensions as Array<unknown>).map((item) =>
                  clampNonNegativeInt(item, 0)
                )
                : null,
              spacing: Array.isArray(toRecord(toRecord(hdf5Source.summary).geometry).spacing)
                ? (toRecord(toRecord(hdf5Source.summary).geometry).spacing as Array<unknown>).map((item) =>
                  toFiniteNumber(item, 0)
                )
                : null,
              origin: Array.isArray(toRecord(toRecord(hdf5Source.summary).geometry).origin)
                ? (toRecord(toRecord(hdf5Source.summary).geometry).origin as Array<unknown>).map((item) =>
                  toFiniteNumber(item, 0)
                )
                : null,
            }
            : null,
      },
      tree: normalizeHdf5Tree(hdf5Source.tree),
      limitations: Array.isArray(hdf5Source.limitations)
        ? hdf5Source.limitations.map((item) => String(item))
        : [],
      selected_dataset_path:
        hdf5Source.selected_dataset_path == null ? null : String(hdf5Source.selected_dataset_path),
      default_dataset_path:
        hdf5Source.default_dataset_path == null ? null : String(hdf5Source.default_dataset_path),
      materials:
        Object.keys(toRecord(hdf5Source.materials)).length > 0
          ? {
              detected: Boolean(toRecord(hdf5Source.materials).detected ?? false),
              schema:
                toRecord(hdf5Source.materials).schema == null
                  ? null
                  : String(toRecord(hdf5Source.materials).schema) === "dream3d"
                    ? "dream3d"
                    : null,
              capabilities: Array.isArray(toRecord(hdf5Source.materials).capabilities)
                ? (toRecord(hdf5Source.materials).capabilities as Array<unknown>).map((item) => String(item))
                : [],
              roles: Object.fromEntries(
                Object.entries(toRecord(toRecord(hdf5Source.materials).roles)).map(([key, value]) => [
                  String(key),
                  String(value),
                ])
              ),
              phase_names: Array.isArray(toRecord(hdf5Source.materials).phase_names)
                ? (toRecord(hdf5Source.materials).phase_names as Array<unknown>).map((item) => String(item))
                : [],
              feature_count:
                toRecord(hdf5Source.materials).feature_count == null
                  ? null
                  : clampNonNegativeInt(toRecord(hdf5Source.materials).feature_count, 0),
              grain_count:
                toRecord(hdf5Source.materials).grain_count == null
                  ? null
                  : clampNonNegativeInt(toRecord(hdf5Source.materials).grain_count, 0),
              recommended_view:
                String(toRecord(hdf5Source.materials).recommended_view ?? "explorer") === "materials"
                  ? "materials"
                  : "explorer",
            }
          : null,
    },
  };
};

export const normalizeUploadViewerInfo = (raw: unknown): UploadViewerInfo => {
  const source = toRecord(raw);
  if (String(source.kind ?? "").trim().toLowerCase() === "hdf5") {
    return normalizeHdf5ViewerInfo(source);
  }
  const metadataSource = toRecord(source.metadata);
  const viewerSource = toRecord(source.viewer);
  const axisSizes = normalizeAxisSizes(source.axis_sizes ?? DEFAULT_AXIS_SIZES);
  const selectedIndices = normalizeSelectedIndices(source.selected_indices, axisSizes);
  const physicalSpacing = normalizePhysicalSpacing(metadataSource, source);
  const backendMode = String(source.backend_mode ?? viewerSource.backend_mode ?? "").trim().toLowerCase();
  const isVolume = Boolean(source.is_volume) || axisSizes.Z > 1 || backendMode === "atlas";
  const planesSource = toRecord(viewerSource.planes);
  const warningsSource = Array.isArray(metadataSource.warnings)
    ? metadataSource.warnings
    : Array.isArray(source.warnings)
      ? source.warnings
      : [];
  const defaultPlane = normalizePlaneDescriptor(viewerSource.default_plane, "z", axisSizes, physicalSpacing);
  const sliceAxes: ViewerAxis[] = isVolume ? ["z", "y", "x"] : ["z"];
  const planes = Object.fromEntries(
    sliceAxes.map((axis) => [axis, normalizePlaneDescriptor(planesSource[axis], axis, axisSizes, physicalSpacing)])
  );
  const availableSurfaces = Array.isArray(viewerSource.available_surfaces)
    ? viewerSource.available_surfaces.map((item) => String(item))
    : isVolume
      ? ["2d", "mpr", "volume", "metadata"]
      : ["2d", "metadata"];
  const defaultSurfaceCandidate = String(viewerSource.default_surface ?? (isVolume ? "volume" : "2d"));
  const defaultSurface = availableSurfaces.includes(defaultSurfaceCandidate)
    ? defaultSurfaceCandidate
    : availableSurfaces[0] ?? "2d";
  const tileSize = toPositiveInt(toRecord(viewerSource.tile_scheme).tile_size, 256);
  const defaultShape =
    axisSizes.T > 1 || axisSizes.C > 1 || axisSizes.Z > 1
      ? [axisSizes.T, axisSizes.C, axisSizes.Z, axisSizes.Y, axisSizes.X]
      : [axisSizes.Y, axisSizes.X];
  const originalName = String(source.original_name ?? "resource");
  const modality = String(inferModality(source, originalName));
  const phys = normalizePhys(source, metadataSource, axisSizes, String(source.file_id ?? ""), originalName, modality);
  const displayDefaults = normalizeDisplayDefaults(source, viewerSource, phys, selectedIndices);
  const orientationFrame = String(toRecord(viewerSource.orientation).frame ?? (isVolume ? "voxel" : "pixel"));
  const rowAxis = String(toRecord(viewerSource.orientation).row_axis ?? defaultPlane.axes[0] ?? "Y");
  const colAxis = String(toRecord(viewerSource.orientation).col_axis ?? defaultPlane.axes[1] ?? "X");
  const sliceAxis = String(toRecord(viewerSource.orientation).slice_axis ?? (isVolume ? "Z" : "")) || null;
  const axisLabels = normalizeOrientationAxisLabels(toRecord(viewerSource.orientation).axis_labels);
  const renderPolicy = normalizeRenderPolicy(
    viewerSource.render_policy,
    inferImageRenderPolicy(source, { modality, axisSizes })
  );
  const measurementPolicy = normalizeMeasurementPolicy(viewerSource.measurement_policy, {
    orientationFrame,
    physicalSpacing,
  });
  const diagnosticSurface = normalizeDiagnosticSurface(
    viewerSource.diagnostic_surface,
    renderPolicy === "scalar" && modality === "medical" && isVolume ? "mpr" : "none"
  );
  const defaultDisplayCapabilities =
    renderPolicy === "scalar"
      ? [
          "slice_navigation",
          "histogram",
          ...(isVolume ? ["volume_context"] : []),
          ...(measurementPolicy !== "pixel-only" ? ["physical_scale"] : []),
          ...(modality === "medical"
            ? ["window_level", "scalar_probe", ...(isVolume ? ["diagnostic_mpr"] : [])]
            : modality === "microscopy"
              ? axisSizes.C > 1
                ? ["channel_mix", "channel_visibility", "channel_color"]
                : ["intensity_window"]
              : ["intensity_window"]),
        ]
      : renderPolicy === "categorical"
        ? ["slice_navigation", "palette", ...(isVolume ? ["volume_context"] : [])]
        : renderPolicy === "display"
          ? ["slice_navigation", "display_composite"]
          : ["slice_navigation", "analysis"];
  const displayCapabilities = normalizeDisplayCapabilities(
    viewerSource.display_capabilities,
    defaultDisplayCapabilities
  );
  const deliveryMode = normalizeDeliveryMode(
    viewerSource.delivery_mode,
    String(viewerSource.volume_mode ?? "").toLowerCase() === "scalar"
      ? "scalar"
      : isVolume
        ? "atlas"
        : String(toRecord(viewerSource.asset_preparation).tile_pyramid ?? "").toLowerCase() === "deferred"
          ? "deferred_multiscale"
          : "direct"
  );
  const firstPaintMode = normalizeFirstPaintMode(
    viewerSource.first_paint_mode,
    defaultSurface === "volume" ? "webgl" : "image"
  );
  const texturePolicy = normalizeTexturePolicy(
    viewerSource.texture_policy,
    inferTexturePolicy(renderPolicy)
  );
  const viewerCapabilities = normalizeViewerCapabilities(
    viewerSource.viewer_capabilities,
    [
      firstPaintMode === "webgl" ? "webgl_first_paint" : "image_first_paint",
      deliveryMode === "scalar"
        ? "scalar_volume_delivery"
        : deliveryMode === "atlas"
          ? "atlas_volume_delivery"
          : deliveryMode === "deferred_multiscale"
            ? "deferred_multiscale"
            : "direct_delivery",
      texturePolicy === "nearest" ? "nearest_sampling" : "linear_sampling",
      ...(diagnosticSurface === "mpr" ? ["mpr_truth_surface"] : []),
      ...displayCapabilities,
    ]
  );
  const serviceUrls = normalizeServiceUrls(toRecord(source.service_urls), String(source.file_id ?? ""));
  const atlasSource = toRecord(viewerSource.atlas_scheme);
  const atlasScheme = Object.keys(atlasSource).length > 0
    ? {
      slice_count: toPositiveInt(atlasSource.slice_count, axisSizes.Z),
      columns: toPositiveInt(atlasSource.columns, 1),
      rows: toPositiveInt(atlasSource.rows, 1),
      slice_width: toPositiveInt(atlasSource.slice_width, defaultPlane.pixel_size.width),
      slice_height: toPositiveInt(atlasSource.slice_height, defaultPlane.pixel_size.height),
      atlas_width: toPositiveInt(atlasSource.atlas_width, defaultPlane.pixel_size.width),
      atlas_height: toPositiveInt(atlasSource.atlas_height, defaultPlane.pixel_size.height),
      downsample: Math.max(1, toFiniteNumber(atlasSource.downsample, 1)),
      format: String(atlasSource.format ?? "png"),
    }
    : buildAtlasScheme(axisSizes, defaultPlane);

  return {
    kind: "image",
    file_id: String(source.file_id ?? ""),
    original_name: originalName,
    modality,
    backend_mode: String(source.backend_mode ?? viewerSource.backend_mode ?? (isVolume ? "atlas" : "direct")),
    dims_order: normalizeDimsOrder(source.dims_order),
    axis_sizes: axisSizes,
    selected_indices: selectedIndices,
    is_volume: isVolume,
    is_timeseries: Boolean(source.is_timeseries) || axisSizes.T > 1,
    is_multichannel: Boolean(source.is_multichannel) || axisSizes.C > 1,
    phys,
    display_defaults: displayDefaults,
    service_urls: serviceUrls,
    metadata: {
      reader: String(metadataSource.reader ?? source.reader ?? "unknown"),
      dims_order: normalizeDimsOrder(metadataSource.dims_order ?? source.dims_order),
      array_shape: Array.isArray(metadataSource.array_shape)
        ? metadataSource.array_shape.map((item) => Math.max(0, Math.round(Number(item) || 0)))
        : defaultShape,
      array_dtype: String(metadataSource.array_dtype ?? source.array_dtype ?? "unknown"),
      array_min: toFiniteNumber(metadataSource.array_min ?? source.array_min, 0),
      array_max: toFiniteNumber(metadataSource.array_max ?? source.array_max, 0),
      intensity_stats: {
        min: toFiniteNumber(metadataSource.array_min ?? source.array_min, 0),
        max: toFiniteNumber(metadataSource.array_max ?? source.array_max, 0),
      },
      physical_spacing: physicalSpacing,
      scene: metadataSource.scene == null ? null : String(metadataSource.scene),
      scene_count: toPositiveInt(metadataSource.scene_count ?? source.scene_count, 1),
      header: Object.fromEntries(Object.entries(toRecord(metadataSource.header)).map(([key, value]) => [key, String(value)])),
      filename_hints: toRecord(metadataSource.filename_hints),
      exif: Object.fromEntries(Object.entries(toRecord(metadataSource.exif)).map(([key, value]) => [key, String(value)])),
      geo: Object.keys(toRecord(metadataSource.geo)).length > 0 ? toRecord(metadataSource.geo) : null,
      dicom: phys.dicom ?? null,
      microscopy:
        Object.keys(toRecord(metadataSource.microscopy)).length > 0
          ? {
              channel_names: Array.isArray(toRecord(metadataSource.microscopy).channel_names)
                ? (toRecord(metadataSource.microscopy).channel_names as Array<unknown>).map((item) => String(item))
                : undefined,
              dimensions_present:
                toRecord(metadataSource.microscopy).dimensions_present == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).dimensions_present),
              objective:
                toRecord(metadataSource.microscopy).objective == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).objective),
              imaging_datetime:
                toRecord(metadataSource.microscopy).imaging_datetime == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).imaging_datetime),
              binning:
                toRecord(metadataSource.microscopy).binning == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).binning),
              position_index:
                toRecord(metadataSource.microscopy).position_index == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).position_index),
              row:
                toRecord(metadataSource.microscopy).row == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).row),
              column:
                toRecord(metadataSource.microscopy).column == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).column),
              timelapse_interval:
                toRecord(metadataSource.microscopy).timelapse_interval == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).timelapse_interval),
              total_time_duration:
                toRecord(metadataSource.microscopy).total_time_duration == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).total_time_duration),
              current_scene:
                toRecord(metadataSource.microscopy).current_scene == null
                  ? undefined
                  : String(toRecord(metadataSource.microscopy).current_scene),
              scene_names: Array.isArray(toRecord(metadataSource.microscopy).scene_names)
                ? (toRecord(metadataSource.microscopy).scene_names as Array<unknown>).map((item) => String(item))
                : undefined,
            }
          : null,
      warnings: warningsSource.map((item: unknown) => String(item)),
    },
    viewer: {
      status: String(viewerSource.status ?? "ready"),
      warmup_mode: String(viewerSource.warmup_mode ?? "lazy"),
      backend_mode: String(
        viewerSource.backend_mode ??
          source.backend_mode ??
          (String(viewerSource.volume_mode ?? "").toLowerCase() === "scalar"
            ? "scalar"
            : isVolume
              ? "atlas"
              : "direct")
      ),
      default_surface: defaultSurface,
      available_surfaces: availableSurfaces,
      default_axis: String(viewerSource.default_axis ?? "z").toLowerCase() as ViewerAxis,
      slice_axes: sliceAxes,
      channel_mode: String(viewerSource.channel_mode ?? displayDefaults.channel_mode ?? "composite"),
      tile_scheme: {
        tile_size: tileSize,
        format: String(toRecord(viewerSource.tile_scheme).format ?? "png"),
        levels: Array.isArray(toRecord(viewerSource.tile_scheme).levels)
          ? (toRecord(viewerSource.tile_scheme).levels as Array<unknown>).map((item, index) => {
            const level = toRecord(item);
            return {
              level: clampNonNegativeInt(level.level, index),
              width: toPositiveInt(level.width, defaultPlane.pixel_size.width),
              height: toPositiveInt(level.height, defaultPlane.pixel_size.height),
              columns: toPositiveInt(level.columns, Math.ceil(defaultPlane.pixel_size.width / tileSize)),
              rows: toPositiveInt(level.rows, Math.ceil(defaultPlane.pixel_size.height / tileSize)),
              downsample: Math.max(1, toFiniteNumber(level.downsample, 1)),
            };
          })
          : buildTileLevels(defaultPlane.pixel_size.width, defaultPlane.pixel_size.height, tileSize),
      },
      atlas_scheme: atlasScheme,
      default_plane: defaultPlane,
      planes,
      volume_mode: String(viewerSource.volume_mode ?? (isVolume ? "atlas" : "none")),
      render_policy: renderPolicy,
      delivery_mode: deliveryMode,
      diagnostic_surface: diagnosticSurface,
      first_paint_mode: firstPaintMode,
      measurement_policy: measurementPolicy,
      texture_policy: texturePolicy,
      display_capabilities: displayCapabilities,
      viewer_capabilities: viewerCapabilities,
      orientation: {
        frame: orientationFrame,
        row_axis: rowAxis,
        col_axis: colAxis,
        slice_axis: sliceAxis,
        axis_labels: axisLabels,
        labels: buildOrientationLabels(rowAxis, colAxis, sliceAxis),
      },
      asset_preparation: {
        status: String(toRecord(viewerSource.asset_preparation).status ?? viewerSource.status ?? "ready"),
        native_supported: Boolean(toRecord(viewerSource.asset_preparation).native_supported ?? true),
        tile_pyramid: String(toRecord(viewerSource.asset_preparation).tile_pyramid ?? viewerSource.warmup_mode ?? "lazy"),
        volume_representation: String(
          toRecord(viewerSource.asset_preparation).volume_representation ?? (isVolume ? "atlas" : "none")
        ),
      },
      chunk_scheme: {
        mode: String(toRecord(viewerSource.chunk_scheme).mode ?? (isVolume ? "atlas" : "none")),
        axis: String(toRecord(viewerSource.chunk_scheme).axis ?? "z").toLowerCase() as ViewerAxis,
        sample_count: toPositiveInt(toRecord(viewerSource.chunk_scheme).sample_count, axisSizes.Z),
      },
      display_defaults: displayDefaults,
      service_urls: normalizeServiceUrls(toRecord(viewerSource.service_urls), String(source.file_id ?? "")),
      fallback_urls: {
        preview: String(toRecord(viewerSource.fallback_urls).preview ?? serviceUrls.preview),
        slice: String(toRecord(viewerSource.fallback_urls).slice ?? serviceUrls.slice),
      },
    },
    hdf5: null,
  };
};
