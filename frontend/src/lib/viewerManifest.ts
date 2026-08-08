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

const CT_VOLUME_DISPLAY_DEFAULTS = {
  // Diagnostic brain window (WC 40 / WW 80 HU); kept in sync with the backend
  // niftiScalarDisplayDefaults. 350/1800 was not a clinical window.
  enhancement: "hounsfield:40.000:80.000",
  fusionMethod: "a",
  volumeSignalFloor: 0.12,
  volumeDensity: 1.75,
  volumeLighting: true,
  volumeLightingStrength: 0.72,
  volumeViewPreset: "iso",
  volumeCameraMode: "orthographic",
} as const;

type DisplayDefaultsContext = {
  backendMode?: string;
  isVolume?: boolean;
  metadataSource?: UnknownRecord;
  modality?: string;
  source?: UnknownRecord;
};

const toRecord = (value: unknown): UnknownRecord =>
  value && typeof value === "object" ? (value as UnknownRecord) : {};

const toFiniteNumber = (value: unknown, fallback: number): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const toBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true") {
      return true;
    }
    if (normalized === "false") {
      return false;
    }
  }
  return fallback;
};

const normalizedString = (value: unknown): string =>
  String(value ?? "").trim().toLowerCase();

const toPositiveInt = (value: unknown, fallback: number): number => {
  const numeric = Math.max(1, Math.round(toFiniteNumber(value, fallback)));
  return Number.isFinite(numeric) ? numeric : Math.max(1, fallback);
};

const clampNonNegativeInt = (value: unknown, fallback: number): number => {
  const numeric = Math.max(0, Math.round(toFiniteNumber(value, fallback)));
  return Number.isFinite(numeric) ? numeric : Math.max(0, fallback);
};

const normalizeExactChannelIndices = (value: unknown, channelCount: number): number[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: number[] = [];
  const seen = new Set<number>();
  for (const entry of value) {
    if (
      typeof entry !== "number" ||
      !Number.isSafeInteger(entry) ||
      entry < 0 ||
      entry >= channelCount ||
      seen.has(entry)
    ) {
      continue;
    }
    normalized.push(entry);
    seen.add(entry);
  }
  return normalized;
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

const normalizePhysicalSpacingUnit = (
  metadata: UnknownRecord,
  viewer: UnknownRecord,
  source: UnknownRecord
): string | null => {
  const unit = String(
    metadata.physical_spacing_unit ??
      viewer.physical_spacing_unit ??
      source.physical_spacing_unit ??
      ""
  ).trim();
  return unit || null;
};

const positiveSpacing = (value: unknown): number => {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 1;
};

const buildPlaneDescriptor = (
  axis: ViewerAxis,
  axisSizes: UploadViewerInfo["axis_sizes"],
  spacing: NonNullable<UploadViewerInfo["metadata"]["physical_spacing"]> | null
): UploadViewerInfo["viewer"]["default_plane"] => {
  // Physical spacing must keep its true magnitude: clamping to >=1mm squashed
  // sub-millimeter anisotropic CT/MRI in-plane (e.g. 0.439mm read as 1mm). Only
  // guard against non-positive/non-finite values.
  const zSpacing = positiveSpacing(spacing?.z);
  const ySpacing = positiveSpacing(spacing?.y);
  const xSpacing = positiveSpacing(spacing?.x);

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

const scalarRangeLooksCTLike = (source: UnknownRecord, metadataSource: UnknownRecord): boolean => {
  const stats = toRecord(metadataSource.intensity_stats ?? source.intensity_stats);
  const rawMin = toFiniteNumber(metadataSource.array_min ?? source.array_min ?? stats.min, NaN);
  const rawMax = toFiniteNumber(metadataSource.array_max ?? source.array_max ?? stats.max, NaN);
  return Number.isFinite(rawMin) && Number.isFinite(rawMax) && rawMin <= -900 && rawMax >= 500;
};

const isScalarVolumeSource = (
  source: UnknownRecord,
  viewerSource: UnknownRecord,
  context?: DisplayDefaultsContext
): boolean => {
  const backendMode = normalizedString(
    context?.backendMode ?? source.backend_mode ?? viewerSource.backend_mode
  );
  return (
    backendMode === "scalar" ||
    normalizedString(viewerSource.backend_mode) === "scalar" ||
    normalizedString(viewerSource.volume_mode) === "scalar" ||
    normalizedString(viewerSource.render_policy) === "scalar" ||
    normalizedString(viewerSource.delivery_mode) === "scalar"
  );
};

const shouldUseCTVolumeDisplayDefaults = (
  source: UnknownRecord,
  viewerSource: UnknownRecord,
  context?: DisplayDefaultsContext
): boolean => {
  const metadataSource = context?.metadataSource ?? toRecord(source.metadata);
  return (
    normalizedString(context?.modality ?? source.modality) === "medical" &&
    Boolean(context?.isVolume ?? source.is_volume) &&
    isScalarVolumeSource(source, viewerSource, context) &&
    scalarRangeLooksCTLike(context?.source ?? source, metadataSource)
  );
};

const normalizeServiceUrls = (source: UnknownRecord, fileId: string) => {
  const fileSegment = encodeURIComponent(fileId);
  return {
    preview: String(source.preview ?? `/v2/uploads/${fileSegment}/preview`),
    display: source.display == null ? undefined : String(source.display),
    slice: String(source.slice ?? `/v2/uploads/${fileSegment}/slice`),
    tile: String(source.tile ?? `/v2/uploads/${fileSegment}/tiles`),
    atlas: String(source.atlas ?? `/v2/uploads/${fileSegment}/atlas`),
    scalar_volume:
      source.scalar_volume == null ? undefined : String(source.scalar_volume),
    histogram: String(source.histogram ?? `/v2/uploads/${fileSegment}/histogram`),
  };
};

const normalizeHdf5ServiceUrls = (source: UnknownRecord, fileId: string) => {
  const fileSegment = encodeURIComponent(fileId);
  return {
    dataset: String(source.dataset ?? `/v2/uploads/${fileSegment}/hdf5/dataset`),
    slice: String(source.slice ?? `/v2/uploads/${fileSegment}/hdf5/preview/slice`),
    atlas: String(source.atlas ?? `/v2/uploads/${fileSegment}/hdf5/preview/atlas`),
    histogram: String(source.histogram ?? `/v2/uploads/${fileSegment}/hdf5/preview/histogram`),
    table: String(source.table ?? `/v2/uploads/${fileSegment}/hdf5/preview/table`),
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
  const physicalSpacingSource = toRecord(
    metadataSource.physical_spacing ?? source.physical_spacing
  );
  const spacingUnitsSource = toRecord(metadataSource.spacing_units);
  const sourceChannelNames = Array.isArray(source.channel_names)
    ? source.channel_names.map((value) => String(value))
    : [];
  const sourceDisplayDefaults = toRecord(source.display_defaults);
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
    pixel_depth:
      physSource.pixel_depth == null ? undefined : toPositiveInt(physSource.pixel_depth, 1),
    pixel_format:
      physSource.pixel_format == null ? undefined : String(physSource.pixel_format),
    pixel_size: Array.isArray(physSource.pixel_size)
      ? physSource.pixel_size.map((value) => toFiniteNumber(value, 1))
      : [
        positiveSpacing(physicalSpacingSource.x),
        positiveSpacing(physicalSpacingSource.y),
        positiveSpacing(physicalSpacingSource.z),
        1,
      ],
    pixel_units: Array.isArray(physSource.pixel_units)
      ? physSource.pixel_units.map((value) => String(value))
      : [
        String(spacingUnitsSource.x ?? "px"),
        String(spacingUnitsSource.y ?? "px"),
        String(spacingUnitsSource.z ?? "px"),
        "frame",
      ],
    channel_names: Array.isArray(physSource.channel_names)
      ? physSource.channel_names.map((value) => String(value))
      : sourceChannelNames.length === channelCount
        ? sourceChannelNames
        : Array.from({ length: channelCount }, (_, index) => (channelCount === 1 ? "Intensity" : `Ch${index + 1}`)),
    display_channels: Array.isArray(physSource.display_channels)
      ? normalizeExactChannelIndices(physSource.display_channels, channelCount)
      : Array.isArray(sourceDisplayDefaults.channels)
        ? normalizeExactChannelIndices(sourceDisplayDefaults.channels, channelCount)
      : channelCount === 1
        ? [0]
        : channelCount === 2
          ? [0, 1]
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
  selectedIndices: UploadViewerInfo["selected_indices"],
  context?: DisplayDefaultsContext
): NonNullable<UploadViewerInfo["display_defaults"]> => {
  const defaultsSource = toRecord(source.display_defaults);
  const viewerDefaults = toRecord(viewerSource.display_defaults);
  const merged = { ...viewerDefaults, ...defaultsSource };
  const useCTVolumeDefaults = shouldUseCTVolumeDisplayDefaults(source, viewerSource, context);
  return {
    enhancement: String(
      merged.enhancement ?? (useCTVolumeDefaults ? CT_VOLUME_DISPLAY_DEFAULTS.enhancement : "d")
    ),
    negative: Boolean(merged.negative ?? false),
    rotate: Math.round(toFiniteNumber(merged.rotate, 0)),
    fusion_method: String(
      merged.fusion_method ?? (useCTVolumeDefaults ? CT_VOLUME_DISPLAY_DEFAULTS.fusionMethod : "m")
    ),
    channel_mode: String(merged.channel_mode ?? "composite"),
    channels: Array.isArray(merged.channels)
      ? normalizeExactChannelIndices(merged.channels, Math.max(1, phys.ch ?? 1))
      : normalizeExactChannelIndices(
          phys.display_channels ?? [0, 1, 2],
          Math.max(1, phys.ch ?? 1)
        ),
    channel_colors: Array.isArray(merged.channel_colors)
      ? merged.channel_colors.map((value) => String(value))
      : (phys.channel_colors ?? []).map((entry) => entry.hex),
    time_index: clampNonNegativeInt(merged.time_index, selectedIndices.T),
    z_index: clampNonNegativeInt(merged.z_index, selectedIndices.Z),
    scalar_colormap: String(merged.scalar_colormap ?? "grayscale"),
    volume_signal_floor:
      merged.volume_signal_floor == null
        ? useCTVolumeDefaults
          ? CT_VOLUME_DISPLAY_DEFAULTS.volumeSignalFloor
          : 0
        : Math.max(0, Math.min(0.95, toFiniteNumber(merged.volume_signal_floor, 0))),
    volume_density:
      merged.volume_density == null
        ? useCTVolumeDefaults
          ? CT_VOLUME_DISPLAY_DEFAULTS.volumeDensity
          : 1
        : Math.max(0.1, Math.min(3, toFiniteNumber(merged.volume_density, 1))),
    volume_lighting: toBoolean(
      merged.volume_lighting,
      useCTVolumeDefaults ? CT_VOLUME_DISPLAY_DEFAULTS.volumeLighting : false
    ),
    volume_lighting_strength:
      merged.volume_lighting_strength == null
        ? useCTVolumeDefaults
          ? CT_VOLUME_DISPLAY_DEFAULTS.volumeLightingStrength
          : 0.65
        : Math.max(0, Math.min(1, toFiniteNumber(merged.volume_lighting_strength, 0.65))),
    volume_channel:
      merged.volume_channel == null ? clampNonNegativeInt(selectedIndices.C, 0) : clampNonNegativeInt(merged.volume_channel, selectedIndices.C),
    volume_view_preset:
      merged.volume_view_preset == null && !useCTVolumeDefaults
        ? undefined
        : String(merged.volume_view_preset ?? CT_VOLUME_DISPLAY_DEFAULTS.volumeViewPreset),
    volume_camera_mode:
      merged.volume_camera_mode == null && !useCTVolumeDefaults
        ? undefined
        : String(merged.volume_camera_mode ?? CT_VOLUME_DISPLAY_DEFAULTS.volumeCameraMode),
    scalar_render_mode:
      merged.scalar_render_mode === "intensity" || merged.scalar_render_mode === "mask"
        ? merged.scalar_render_mode
        : "auto",
    scalar_threshold_method:
      merged.scalar_threshold_method === "manual" ? "manual" : "otsu-256-v1",
    scalar_threshold_value:
      Number.isFinite(Number(merged.scalar_threshold_value))
        ? Number(merged.scalar_threshold_value)
        : null,
    scalar_threshold_foreground: "above",
  };
};

const normalizeDataSemantics = (
  value: unknown
): UploadViewerInfo["data_semantics"] | undefined => {
  const source = toRecord(value);
  const kind = String(source.kind ?? "");
  if (kind !== "intensity" && kind !== "binary_mask" && kind !== "probability_mask") {
    return undefined;
  }
  const modes: Array<"intensity" | "mask"> = Array.isArray(source.supported_modes)
    ? source.supported_modes
        .map((mode) => String(mode))
        .filter((mode): mode is "intensity" | "mask" => mode === "intensity" || mode === "mask")
    : ["intensity"];
  const thresholdSource = toRecord(source.threshold);
  const thresholdValue = Number(thresholdSource.value);
  const thresholdChannel = Number(thresholdSource.channel);
  const thresholdTime = Number(thresholdSource.t);
  const thresholdSampleCount = Number(thresholdSource.sample_count);
  const thresholdSampleScope = String(thresholdSource.sample_scope ?? "");
  const thresholdSamplingAlgorithm = String(
    thresholdSource.sampling_algorithm ?? ""
  ).trim();
  return {
    kind,
    basis: String(source.basis ?? "unknown"),
    strength: String(source.strength ?? "unknown"),
    supported_modes: modes.length > 0 ? modes : ["intensity"],
    recommended_view: source.recommended_view === "mask" ? "mask" : "intensity",
    threshold:
      Number.isFinite(thresholdValue) &&
      String(thresholdSource.method) === "otsu-256-v1" &&
      String(thresholdSource.domain) === "raw" &&
      String(thresholdSource.foreground) === "above" &&
      Number.isSafeInteger(thresholdChannel) &&
      thresholdChannel >= 0 &&
      Number.isSafeInteger(thresholdTime) &&
      thresholdTime >= 0 &&
      Number.isSafeInteger(thresholdSampleCount) &&
      thresholdSampleCount > 0 &&
      (thresholdSampleScope === "volume" ||
        thresholdSampleScope === "stratified_z") &&
      Boolean(thresholdSamplingAlgorithm)
        ? {
            method: "otsu-256-v1",
            value: thresholdValue,
            domain: "raw",
            foreground: "above",
            sample_scope: thresholdSampleScope,
            sample_count: thresholdSampleCount,
            z_samples: Array.isArray(thresholdSource.z_samples)
              ? thresholdSource.z_samples.map((entry) => clampNonNegativeInt(entry, 0))
              : [],
            channel: thresholdChannel,
            t: thresholdTime,
            sampling_algorithm: thresholdSamplingAlgorithm,
          }
        : undefined,
  };
};

const normalizeScalarMaskCapability = (
  value: unknown,
  sourceSha256: string,
  arrayDtype: string,
  availableSurfaces: string[]
): UploadViewerInfo["scalar_mask_capability"] | undefined => {
  const source = toRecord(value);
  const dtype = String(source.dtype ?? "").trim().toLowerCase();
  const format = String(source.source_format ?? "");
  const capabilitySha = String(source.source_sha256 ?? "").trim();
  const surfaces = Array.isArray(source.surfaces)
    ? source.surfaces.map((surface) => String(surface))
    : [];
  const expectedSurfaces = availableSurfaces.filter(
    (surface) => surface === "2d" || surface === "mpr" || surface === "volume"
  );
  const requiredSurfaces = ["2d", "mpr", "volume"];
  if (
    Number(source.version) !== 1 ||
    source.source_authority !== "original" ||
    (format !== "tiff" && format !== "ome-tiff") ||
    !["uint8", "uint16", "int16"].includes(dtype) ||
    dtype !== arrayDtype.trim().toLowerCase() ||
    !sourceSha256 ||
    capabilitySha !== sourceSha256 ||
    source.threshold_domain !== "raw" ||
    source.threshold_foreground !== "above" ||
    source.slice_delivery !== "thresholded_png" ||
    source.volume_delivery !== "raw_scalar" ||
    source.volume_sampling !== "nearest" ||
    source.channel_selection !== "single" ||
    source.time_selection !== "single" ||
    expectedSurfaces.length !== requiredSurfaces.length ||
    expectedSurfaces.some(
      (surface, index) => surface !== requiredSurfaces[index]
    ) ||
    surfaces.length !== requiredSurfaces.length ||
    surfaces.some((surface, index) => surface !== requiredSurfaces[index])
  ) {
    return undefined;
  }
  return {
    version: 1,
    source_authority: "original",
    source_format: format,
    source_sha256: capabilitySha,
    dtype: dtype as "uint8" | "uint16" | "int16",
    threshold_domain: "raw",
    threshold_foreground: "above",
    slice_delivery: "thresholded_png",
    volume_delivery: "raw_scalar",
    volume_sampling: "nearest",
    channel_selection: "single",
    time_selection: "single",
    surfaces: surfaces as Array<"2d" | "mpr" | "volume">,
  };
};

export const normalizeViewerCalibrations = (
  value: unknown,
  sourceSha256: string
): UploadViewerInfo["viewer_calibrations"] | undefined => {
  const source = toRecord(value);
  const expectedSourceSha = String(sourceSha256 ?? "").trim();
  if (
    !expectedSourceSha ||
    Number(source.version) !== 1 ||
    String(source.source_sha256 ?? "").trim() !== expectedSourceSha ||
    !source.selections ||
    typeof source.selections !== "object"
  ) {
    return undefined;
  }
  const selections: NonNullable<
    UploadViewerInfo["viewer_calibrations"]
  >["selections"] = {};
  Object.entries(toRecord(source.selections)).forEach(([key, rawSelection]) => {
    const selection = toRecord(rawSelection);
    const provenance = toRecord(selection.threshold_provenance);
    const channel = Number(selection.channel);
    const time = Number(selection.t);
    const revision = Number(selection.revision);
    const threshold = Number(selection.threshold_value);
    const provenanceValue = Number(provenance.value);
    const renderMode = String(selection.render_mode);
    const thresholdMethod = String(selection.threshold_method);
    const sampleCount = clampNonNegativeInt(provenance.sample_count, 0);
    const samplingAlgorithm = String(provenance.sampling_algorithm ?? "").trim();
    const samplingStrategy = String(provenance.sampling_strategy ?? "");
    const sampleScope = String(provenance.sample_scope ?? "");
    const provenanceSourceSha = String(provenance.source_sha256 ?? "").trim();
    const bins = Number(provenance.bins);
    const zSamples = Array.isArray(provenance.z_samples)
      ? provenance.z_samples.map((value) => Number(value))
      : [];
    if (
      !Number.isSafeInteger(channel) ||
      channel < 0 ||
      !Number.isSafeInteger(time) ||
      time < 0 ||
      !Number.isSafeInteger(revision) ||
      revision <= 0 ||
      key !== `c${channel}:t${time}` ||
      (renderMode !== "auto" && renderMode !== "intensity" && renderMode !== "mask") ||
      (thresholdMethod !== "manual" && thresholdMethod !== "otsu-256-v1") ||
      !Number.isFinite(threshold) ||
      (thresholdMethod === "otsu-256-v1" && threshold !== provenanceValue) ||
      String(selection.threshold_foreground) !== "above" ||
      String(provenance.method) !== "otsu-256-v1" ||
      !Number.isFinite(provenanceValue) ||
      String(provenance.domain) !== "raw" ||
      String(provenance.foreground) !== "above" ||
      Number(provenance.channel) !== channel ||
      Number(provenance.t) !== time ||
      (samplingStrategy !== "exact" &&
        samplingStrategy !== "stratified-z-spatial") ||
      sampleScope !==
        (samplingStrategy === "exact" ? "volume" : "stratified_z") ||
      sampleCount <= 0 ||
      !samplingAlgorithm ||
      provenanceSourceSha !== expectedSourceSha ||
      !Number.isSafeInteger(bins) ||
      bins < 8 ||
      zSamples.length === 0 ||
      zSamples.some(
        (value, index) =>
          !Number.isSafeInteger(value) ||
          value < 0 ||
          (index > 0 && value <= (zSamples[index - 1] as number))
      )
    ) {
      return;
    }
    selections[key] = {
      revision,
      channel,
      t: time,
      render_mode: renderMode,
      threshold_method: thresholdMethod,
      threshold_value: threshold,
      threshold_foreground: "above",
      threshold_provenance: {
        method: "otsu-256-v1",
        value: provenanceValue,
        domain: "raw",
        foreground: "above",
        channel,
        t: time,
        sample_scope: sampleScope as "volume" | "stratified_z",
        sample_count: sampleCount,
        sampling_algorithm: samplingAlgorithm,
        sampling_strategy: samplingStrategy as
          | "exact"
          | "stratified-z-spatial",
        z_samples: zSamples,
        source_sha256: provenanceSourceSha,
        bins,
      },
    };
  });
  return Object.keys(selections).length > 0
    ? { version: 1, source_sha256: expectedSourceSha, selections }
    : undefined;
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
  const hdf5Supported = hdf5Source.supported !== false;
  const hdf5Enabled = hdf5Source.enabled !== false;
  const fileId = String(source.file_id ?? "");
  const originalName = String(source.original_name ?? "resource");
  const modality = String(source.modality ?? "unknown");
  const axisSizes = normalizeAxisSizes(source.axis_sizes ?? viewerSource.axis_sizes ?? DEFAULT_AXIS_SIZES);
  const selectedIndices = normalizeSelectedIndices(source.selected_indices ?? viewerSource.selected_indices, axisSizes);
  const physicalSpacing = normalizePhysicalSpacing(metadataSource, source);
  const physicalSpacingUnit = normalizePhysicalSpacingUnit(metadataSource, viewerSource, source);
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
    ["dataset_explorer"]
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
      physical_spacing_unit: physicalSpacingUnit,
      spacing_units: (() => {
        const units = toRecord(metadataSource.spacing_units);
        if (Object.keys(units).length === 0) {
          return undefined;
        }
        return {
          x: units.x == null ? null : String(units.x),
          y: units.y == null ? null : String(units.y),
          z: units.z == null ? null : String(units.z),
        };
      })(),
      exif: {},
      geo: null,
      dicom: null,
      microscopy: null,
      warnings: warningsSource.map((item: unknown) => String(item)),
    },
    viewer: {
      status: String(
        viewerSource.status ?? (hdf5Supported ? "ready" : "degraded-fallback")
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
            (hdf5Supported ? "ready" : "degraded-fallback")
        ),
        native_supported:
          toRecord(viewerSource.asset_preparation).native_supported !== false &&
          hdf5Supported,
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
      enabled: hdf5Enabled,
      supported: hdf5Supported,
      status: String(hdf5Source.status ?? (hdf5Supported ? "ready" : "unsupported")),
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
              cell_data_path:
                toRecord(toRecord(hdf5Source.summary).geometry).cell_data_path == null
                  ? null
                  : String(toRecord(toRecord(hdf5Source.summary).geometry).cell_data_path),
              cell_data_consistent:
                toRecord(toRecord(hdf5Source.summary).geometry).cell_data_consistent == null
                  ? null
                  : Boolean(toRecord(toRecord(hdf5Source.summary).geometry).cell_data_consistent),
              complete:
                toRecord(toRecord(hdf5Source.summary).geometry).complete == null
                  ? null
                  : Boolean(toRecord(toRecord(hdf5Source.summary).geometry).complete),
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
    },
  };
};

const normalizeCiftiViewerInfo = (source: UnknownRecord): UploadViewerInfo => {
  const fileId = String(source.file_id ?? "");
  const seg = encodeURIComponent(fileId);
  const urls = toRecord(source.service_urls);
  const rows = clampNonNegativeInt(source.rows, 0);
  const cols = clampNonNegativeInt(source.cols, 0);
  const views = (Array.isArray(source.views) ? source.views : [])
    .map((v) => String(v))
    .filter((v) => v === "carpet" || v === "connectivity");
  const structures = (Array.isArray(source.structures) ? source.structures : [])
    .map((s) => toRecord(s))
    .map((s) => ({ name: String(s.name ?? ""), count: clampNonNegativeInt(s.count, 0) }));
  const colAxis = toRecord(source.column_axis);
  // Reuse the generic builder for a type-complete metadata/viewer skeleton, then
  // stamp the CIFTI identity + payload on top. The matrix shape rides in axis_sizes
  // (Y=rows/grayordinates, X=cols) so downstream size reads stay sane.
  const base = normalizeUploadViewerInfo({
    kind: "image",
    file_id: fileId,
    original_name: String(source.original_name ?? "resource"),
    decodable: true,
    axis_sizes: { T: 1, C: 1, Z: 1, Y: rows, X: cols },
  });
  return {
    ...base,
    kind: "cifti",
    modality: "connectivity",
    is_volume: false,
    is_timeseries: colAxis.role === "series",
    message: typeof source.message === "string" ? source.message : undefined,
    cifti: {
      cifti_type: String(source.cifti_type ?? "connectivity"),
      views: views.length > 0 ? views : ["carpet"],
      rows,
      cols,
      structures,
      column_axis: {
        role: colAxis.role == null ? undefined : String(colAxis.role),
        size: colAxis.size == null ? undefined : clampNonNegativeInt(colAxis.size, 0),
        step: colAxis.step == null ? undefined : Number(colAxis.step),
        unit: colAxis.unit == null ? undefined : String(colAxis.unit),
      },
      service_urls: {
        carpet: String(urls.carpet ?? `/v2/uploads/${seg}/cifti/carpet`),
        connectivity: String(urls.connectivity ?? `/v2/uploads/${seg}/cifti/connectivity`),
        download: String(urls.download ?? `/v2/resources/${seg}/download`),
      },
    },
  };
};

const SCENE3D_STATUSES = new Set(["ready", "deriving", "failed"]);
const SCENE3D_KINDS = new Set(["splat", "pointcloud", "colmap"]);

const normalizeScene3dViewerInfo = (source: UnknownRecord): UploadViewerInfo => {
  const fileId = String(source.file_id ?? "");
  const seg = encodeURIComponent(fileId);
  const urls = toRecord(source.service_urls);
  const status = normalizedString(source.status);
  const sceneKind = normalizedString(source.scene_kind);
  // Same trick as CIFTI: build a type-complete skeleton with the generic
  // normalizer, then stamp the scene identity on top. A 3D scene has no
  // T/C/Z/Y/X grid at all, so the axis sizes stay at their 1×1 defaults and
  // nothing downstream tries to drive a slice slider from them.
  const base = normalizeUploadViewerInfo({
    kind: "image",
    file_id: fileId,
    original_name: String(source.original_name ?? "resource"),
    decodable: true,
  });
  return {
    ...base,
    kind: "scene3d",
    modality: "unknown",
    is_volume: false,
    is_timeseries: false,
    is_multichannel: false,
    message: typeof source.message === "string" ? source.message : undefined,
    scene3d: {
      status: SCENE3D_STATUSES.has(status) ? status : "ready",
      scene_kind: SCENE3D_KINDS.has(sceneKind) ? sceneKind : "pointcloud",
      element_count: clampNonNegativeInt(source.element_count, 0),
      message: source.message == null ? null : String(source.message),
      service_urls: {
        manifest: String(urls.manifest ?? `/v2/uploads/${seg}/scene3d/manifest`),
        chunk: String(urls.chunk ?? `/v2/uploads/${seg}/scene3d/chunk`),
        download: String(urls.download ?? `/v2/resources/${seg}/download`),
      },
    },
  };
};

export const normalizeUploadViewerInfo = (raw: unknown): UploadViewerInfo => {
  const source = toRecord(raw);
  if (String(source.kind ?? "").trim().toLowerCase() === "hdf5") {
    return normalizeHdf5ViewerInfo(source);
  }
  if (String(source.kind ?? "").trim().toLowerCase() === "cifti") {
    return normalizeCiftiViewerInfo(source);
  }
  if (String(source.kind ?? "").trim().toLowerCase() === "scene3d") {
    return normalizeScene3dViewerInfo(source);
  }
  const metadataSource = toRecord(source.metadata);
  const viewerSource = toRecord(source.viewer);
  const axisSizes = normalizeAxisSizes(source.axis_sizes ?? DEFAULT_AXIS_SIZES);
  const selectedIndices = normalizeSelectedIndices(source.selected_indices, axisSizes);
  const physicalSpacing = normalizePhysicalSpacing(metadataSource, source);
  const physicalSpacingUnit = normalizePhysicalSpacingUnit(metadataSource, viewerSource, source);
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
  const displayDefaults = normalizeDisplayDefaults(source, viewerSource, phys, selectedIndices, {
    backendMode,
    isVolume,
    metadataSource,
    modality,
    source,
  });
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
  const dataSemantics = normalizeDataSemantics(source.data_semantics);
  const sourceSha256 =
    metadataSource.sha256 == null ? "" : String(metadataSource.sha256);
  const arrayDtype = String(metadataSource.array_dtype ?? source.array_dtype ?? "unknown");
  const scalarMaskCapability = normalizeScalarMaskCapability(
    source.scalar_mask_capability,
    sourceSha256,
    arrayDtype,
    availableSurfaces
  );
  const viewerCalibrations = normalizeViewerCalibrations(
    source.viewer_calibrations,
    sourceSha256
  );
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

  // The engine can recognize a container but not decode it (a Leica .lif, etc.); the
  // control plane then sends kind:"unsupported" + decodable:false. Preserve that so the
  // viewer can show a "preview unavailable, download instead" card instead of a broken
  // 1×1 canvas. Everything else is filled with the usual (zeroed) image shape so any
  // consumer reading axis_sizes/service_urls unconditionally still works.
  const undecodable = source.decodable === false || String(source.kind ?? "").trim().toLowerCase() === "unsupported";
  return {
    kind: undecodable ? "unsupported" : "image",
    decodable: undecodable ? false : undefined,
    message: typeof source.message === "string" ? source.message : undefined,
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
    data_semantics: dataSemantics,
    scalar_mask_capability: scalarMaskCapability,
    viewer_calibrations: viewerCalibrations,
    phys,
    display_defaults: displayDefaults,
    service_urls: serviceUrls,
    metadata: {
      reader: String(metadataSource.reader ?? source.reader ?? "unknown"),
      // The real container format ("OME-TIFF"/"BigTIFF"/...), distinct from the reader.
      format: String(metadataSource.format ?? source.format ?? ""),
      dims_order: normalizeDimsOrder(metadataSource.dims_order ?? source.dims_order),
      array_shape: Array.isArray(metadataSource.array_shape)
        ? metadataSource.array_shape.map((item) => Math.max(0, Math.round(Number(item) || 0)))
        : defaultShape,
      array_dtype: arrayDtype,
      sha256: sourceSha256 || undefined,
      size_bytes:
        Number.isFinite(Number(metadataSource.size_bytes))
          ? Number(metadataSource.size_bytes)
          : undefined,
      // NaN (not 0) when the backend never computed an intensity range, so the viewer
      // hides the row instead of showing a meaningless "0 → 0".
      array_min: toFiniteNumber(metadataSource.array_min ?? source.array_min, Number.NaN),
      array_max: toFiniteNumber(metadataSource.array_max ?? source.array_max, Number.NaN),
      intensity_stats: {
        min: toFiniteNumber(metadataSource.array_min ?? source.array_min, Number.NaN),
        max: toFiniteNumber(metadataSource.array_max ?? source.array_max, Number.NaN),
      },
      // Curated provenance + instrument facts (software, capture date, acquisition
      // mode, objective, detector, experimenter). Empty object when the file carries
      // none, so the viewer's Acquisition group is hidden.
      acquisition: Object.fromEntries(
        Object.entries(toRecord(metadataSource.acquisition)).map(([key, value]) => [
          key,
          typeof value === "number" ? value : String(value),
        ])
      ),
      physical_spacing: physicalSpacing,
      physical_spacing_unit: physicalSpacingUnit,
      spacing_units: (() => {
        const units = toRecord(metadataSource.spacing_units);
        if (Object.keys(units).length === 0) {
          return undefined;
        }
        return {
          x: units.x == null ? null : String(units.x),
          y: units.y == null ? null : String(units.y),
          z: units.z == null ? null : String(units.z),
        };
      })(),
      scene: metadataSource.scene == null ? null : String(metadataSource.scene),
      scene_count: toPositiveInt(metadataSource.scene_count ?? source.scene_count, 1),
      header: Object.fromEntries(Object.entries(toRecord(metadataSource.header)).map(([key, value]) => [key, String(value)])),
      // Tiled-mosaic acquisition (>1 stitched/unstitched field). Null for a normal
      // single-field image. An UNstitched mosaic shows per-field illumination seams
      // that look like a render bug but are the raw data — the viewer labels it.
      mosaic: (() => {
        const source_ = toRecord(metadataSource.mosaic);
        const tiles = toPositiveInt(source_.tiles, 0);
        if (tiles <= 1) {
          return null;
        }
        const overlap = toFiniteNumber(source_.overlap, Number.NaN);
        return {
          tiles,
          stitched: typeof source_.stitched === "boolean" ? source_.stitched : undefined,
          overlap: Number.isFinite(overlap) ? overlap : undefined,
        };
      })(),
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
