import { DataUtils } from "three";

import type { ScalarVolumePayload } from "@/lib/api";

export type ScalarVolumeVoxelIndex = {
  x: number;
  y: number;
  z: number;
};

export type ScalarVolumeRescale = {
  slope: number;
  intercept: number;
};

export type ScalarVolumeWindow = {
  low: number;
  high: number;
  invert: boolean;
};

export type ScalarVolumeDataPayload = Pick<
  ScalarVolumePayload,
  | "data"
  | "width"
  | "height"
  | "depth"
  | "dtype"
  | "bytesPerVoxel"
  | "rawMin"
  | "rawMax"
  | "channel"
  | "sclSlope"
  | "sclInter"
>;

export type PreparedScalarVolume = Omit<ScalarVolumePayload, "data"> & {
  textureData: Uint16Array;
  autoWindow: { low: number; high: number };
};

export const MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES = 128 * 1024 * 1024;
export const MAX_ACTIVE_SCALAR_VOLUME_GPU_BYTES = 128 * 1024 * 1024;

const apiClientNamespaceIds = new WeakMap<object, number>();
let nextApiClientNamespaceId = 1;

export const scalarVolumeApiNamespace = (
  client: object | undefined,
  serviceUrl: string
): string => {
  if (!client) {
    return `external:${serviceUrl}`;
  }
  let id = apiClientNamespaceIds.get(client);
  if (id == null) {
    id = nextApiClientNamespaceId;
    nextApiClientNamespaceId += 1;
    apiClientNamespaceIds.set(client, id);
  }
  return `api-${id}:${serviceUrl}`;
};

export const scalarVolumeSourceIdentity = (identity: {
  fileId: string;
  sha256?: string | null;
  sizeBytes?: number | null;
  dtype?: string | null;
  sourceGrid: { width: number; height: number; depth: number };
}): string => {
  const digest = String(identity.sha256 ?? "").trim();
  if (digest) {
    return `sha256:${digest}`;
  }
  return JSON.stringify({
    file: identity.fileId,
    size: identity.sizeBytes ?? null,
    dtype: identity.dtype ?? null,
    sourceGrid: [
      identity.sourceGrid.width,
      identity.sourceGrid.height,
      identity.sourceGrid.depth,
    ],
  });
};

export const scalarVolumeIdentityKey = (identity: {
  apiNamespace: string;
  fileId: string;
  sourceIdentity: string;
  sourceGrid: { width: number; height: number; depth: number };
  channel: number;
  time: number;
  channelCount: number;
  timeCount: number;
  policy: string;
}): string => JSON.stringify(identity);

const checkedPositiveSafeInteger = (value: number, field: string): number => {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${field} must be a positive safe integer.`);
  }
  return value;
};

export const preparedScalarVolumeByteLength = (grid: {
  width: number;
  height: number;
  depth: number;
}): number => {
  const width = checkedPositiveSafeInteger(grid.width, "Scalar volume width");
  const height = checkedPositiveSafeInteger(grid.height, "Scalar volume height");
  const depth = checkedPositiveSafeInteger(grid.depth, "Scalar volume depth");
  const voxelCount = width * height * depth;
  const bytes = voxelCount * Uint16Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(voxelCount) || !Number.isSafeInteger(bytes)) {
    throw new RangeError("Scalar volume prepared geometry overflows safe integer bounds.");
  }
  return bytes;
};

export type PreparedScalarVolumeLease = {
  readonly value: PreparedScalarVolume;
  release(): void;
};

export type PreparedScalarVolumeReservation = {
  readonly keys: readonly string[];
  readonly bytesPerMissingEntry: number;
  release(): void;
};

type ResidencyEntry = {
  value: PreparedScalarVolume;
  bytes: number;
  pins: number;
};

type ResidencyReservation = {
  keys: Set<string>;
  missingKeys: Set<string>;
  bytesPerMissingEntry: number;
};

/**
 * Process-wide prepared-volume residency. Cached and active textures share the
 * same Uint16Array, so a pinned cache entry is counted once. Missing entries are
 * reserved before conversion; only unpinned LRU entries may be evicted.
 */
export class PreparedScalarVolumeResidencyManager {
  readonly maxBytes: number;
  private readonly entries = new Map<string, ResidencyEntry>();
  private readonly reservations = new Map<symbol, ResidencyReservation>();
  private readonly reservationOwners = new WeakMap<PreparedScalarVolumeReservation, symbol>();

  constructor(maxBytes = MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new RangeError("Prepared scalar volume residency must be a positive safe integer.");
    }
    this.maxBytes = maxBytes;
  }

  get byteSize(): number {
    let bytes = 0;
    this.entries.forEach((entry) => {
      bytes += entry.bytes;
    });
    this.reservations.forEach((reservation) => {
      bytes += reservation.missingKeys.size * reservation.bytesPerMissingEntry;
    });
    return bytes;
  }

  get(key: string): PreparedScalarVolume | undefined {
    const entry = this.entries.get(key);
    if (!entry) {
      return undefined;
    }
    this.entries.delete(key);
    this.entries.set(key, entry);
    return entry.value;
  }

  reserve(keys: readonly string[], bytesPerMissingEntry: number): PreparedScalarVolumeReservation {
    checkedPositiveSafeInteger(bytesPerMissingEntry, "Prepared scalar volume reservation");
    const uniqueKeys = Array.from(new Set(keys));
    if (uniqueKeys.length === 0) {
      throw new RangeError("Prepared scalar volume reservation requires at least one key.");
    }
    const owner = Symbol("prepared-volume-reservation");
    // Each owner that observes a cache miss may allocate a distinct staged array.
    // Count every one of those arrays even when another viewer is preparing the
    // same identity concurrently; publication will collapse them to one cache entry.
    const missingKeys = uniqueKeys.filter((key) => !this.entries.has(key));
    const additionalBytes = missingKeys.length * bytesPerMissingEntry;
    if (!Number.isSafeInteger(additionalBytes)) {
      throw new RangeError("Prepared scalar volume reservation overflows safe integer bounds.");
    }
    const protectedKeys = new Set(uniqueKeys);
    this.evictUnpinnedUntil(this.byteSize + additionalBytes, protectedKeys);
    if (this.byteSize + additionalBytes > this.maxBytes) {
      throw new RangeError(
        `Prepared scalar volume residency requires ${this.byteSize + additionalBytes} bytes, exceeding the ${this.maxBytes} byte limit.`
      );
    }
    this.reservations.set(owner, {
      keys: new Set(uniqueKeys),
      missingKeys: new Set(missingKeys),
      bytesPerMissingEntry,
    });
    let released = false;
    const reservation: PreparedScalarVolumeReservation = {
      keys: uniqueKeys,
      bytesPerMissingEntry,
      release: () => {
        if (released) {
          return;
        }
        released = true;
        this.reservations.delete(owner);
      },
    };
    this.reservationOwners.set(reservation, owner);
    return reservation;
  }

  acquire(key: string): PreparedScalarVolumeLease | undefined {
    const value = this.get(key);
    const entry = this.entries.get(key);
    if (!value || !entry) {
      return undefined;
    }
    entry.pins += 1;
    let released = false;
    return {
      value,
      release: () => {
        if (released) {
          return;
        }
        released = true;
        entry.pins = Math.max(0, entry.pins - 1);
      },
    };
  }

  publishAndAcquire(
    reservation: PreparedScalarVolumeReservation,
    key: string,
    value: PreparedScalarVolume
  ): PreparedScalarVolumeLease {
    if (!reservation.keys.includes(key)) {
      throw new RangeError("Prepared scalar volume publication is outside its reservation.");
    }
    const bytes = value.textureData.byteLength;
    if (bytes !== reservation.bytesPerMissingEntry) {
      throw new RangeError(
        `Prepared scalar volume produced ${bytes} bytes, not its reserved ${reservation.bytesPerMissingEntry} bytes.`
      );
    }
    const owner = this.reservationOwners.get(reservation);
    const activeReservation = owner == null ? undefined : this.reservations.get(owner);
    if (!activeReservation || !activeReservation.keys.has(key)) {
      throw new Error("Prepared scalar volume reservation is no longer active.");
    }
    const existing = this.entries.get(key);
    if (existing && existing.bytes !== bytes) {
      throw new RangeError("Prepared scalar volume cache entry disagrees with its reservation size.");
    }
    if (!existing && !activeReservation.missingKeys.has(key)) {
      throw new Error("Prepared scalar volume publication has no staged-byte reservation.");
    }
    activeReservation.missingKeys.delete(key);
    if (!existing) {
      this.entries.set(key, { value, bytes, pins: 0 });
    }
    const lease = this.acquire(key);
    if (!lease) {
      throw new Error("Prepared scalar volume publication failed.");
    }
    return lease;
  }

  clear(): void {
    if (this.reservations.size > 0) {
      throw new Error("Cannot clear prepared scalar volume residency while preparations are active.");
    }
    for (const entry of this.entries.values()) {
      if (entry.pins > 0) {
        throw new Error("Cannot clear prepared scalar volume residency while entries are active.");
      }
    }
    this.entries.clear();
  }

  private isReservationProtected(key: string): boolean {
    for (const reservation of this.reservations.values()) {
      if (reservation.keys.has(key)) {
        return true;
      }
    }
    return false;
  }

  private evictUnpinnedUntil(targetBytes: number, protectedKeys: Set<string>): void {
    if (targetBytes <= this.maxBytes) {
      return;
    }
    for (const [key, entry] of this.entries) {
      if (entry.pins > 0 || protectedKeys.has(key) || this.isReservationProtected(key)) {
        continue;
      }
      this.entries.delete(key);
      targetBytes -= entry.bytes;
      if (targetBytes <= this.maxBytes) {
        return;
      }
    }
  }
}

export class PreparedScalarVolumeCache {
  readonly maxBytes: number;
  private readonly entries = new Map<string, PreparedScalarVolume>();
  private usedBytes = 0;

  constructor(maxBytes = MAX_PREPARED_SCALAR_VOLUME_CACHE_BYTES) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
      throw new RangeError("Prepared scalar volume cache size must be a positive safe integer.");
    }
    this.maxBytes = maxBytes;
  }

  get byteSize(): number {
    return this.usedBytes;
  }

  get(key: string): PreparedScalarVolume | undefined {
    const value = this.entries.get(key);
    if (!value) {
      return undefined;
    }
    this.entries.delete(key);
    this.entries.set(key, value);
    return value;
  }

  peek(key: string): PreparedScalarVolume | undefined {
    return this.entries.get(key);
  }

  set(key: string, value: PreparedScalarVolume): void {
    const bytes = value.textureData.byteLength;
    if (bytes > this.maxBytes) {
      throw new RangeError(
        `Prepared scalar volume requires ${bytes} bytes, exceeding the ${this.maxBytes} byte cache limit.`
      );
    }
    const previous = this.entries.get(key);
    if (previous) {
      this.usedBytes -= previous.textureData.byteLength;
      this.entries.delete(key);
    }
    this.entries.set(key, value);
    this.usedBytes += bytes;
    while (this.usedBytes > this.maxBytes) {
      const oldestKey = this.entries.keys().next().value as string | undefined;
      if (oldestKey == null) {
        break;
      }
      const oldest = this.entries.get(oldestKey);
      this.entries.delete(oldestKey);
      this.usedBytes -= oldest?.textureData.byteLength ?? 0;
    }
  }

  clear(): void {
    this.entries.clear();
    this.usedBytes = 0;
  }
}

export const validateScalarVolumeIdentity = (
  payload: Pick<
    ScalarVolumePayload,
    | "channel"
    | "time"
    | "sourceWidth"
    | "sourceHeight"
    | "sourceDepth"
    | "width"
    | "height"
    | "depth"
    | "downsampleX"
    | "downsampleY"
    | "downsampleZ"
    | "previewPolicy"
  >,
  expected: {
    channel: number;
    time: number;
    sourceGrid: { width: number; height: number; depth: number };
    policy: string;
  }
): void => {
  if (payload.channel !== expected.channel) {
    throw new RangeError(
      `Scalar volume channel mismatch: requested ${expected.channel}, received ${String(payload.channel)}.`
    );
  }
  if (payload.time !== expected.time) {
    throw new RangeError(
      `Scalar volume time mismatch: requested ${expected.time}, received ${String(payload.time)}.`
    );
  }
  if (payload.previewPolicy !== expected.policy) {
    throw new RangeError(
      `Scalar volume preview policy mismatch: expected ${expected.policy}, received ${String(payload.previewPolicy)}.`
    );
  }
  const actualSource = [payload.sourceWidth, payload.sourceHeight, payload.sourceDepth] as const;
  const expectedSource = [
    expected.sourceGrid.width,
    expected.sourceGrid.height,
    expected.sourceGrid.depth,
  ] as const;
  if (actualSource.some((value, index) => value !== expectedSource[index])) {
    throw new RangeError(
      `Scalar volume source grid mismatch: expected ${expectedSource.join("x")}, received ${actualSource.join("x")}.`
    );
  }
  const previewAxes = [
    [actualSource[0], payload.downsampleX, payload.width],
    [actualSource[1], payload.downsampleY, payload.height],
    [actualSource[2], payload.downsampleZ, payload.depth],
  ] as const;
  if (
    previewAxes.some(
      ([sourceSize, factor, deliveredSize]) =>
        !Number.isSafeInteger(sourceSize) ||
        !Number.isSafeInteger(factor) ||
        Number(sourceSize) <= 0 ||
        Number(factor) <= 0 ||
        Math.ceil(Number(sourceSize) / Number(factor)) !== deliveredSize
    )
  ) {
    throw new RangeError("Scalar volume delivered grid does not match its preview provenance.");
  }
};

export const resolveScalarVolumeRescale = (
  payload: Pick<ScalarVolumePayload, "sclSlope" | "sclInter">
): ScalarVolumeRescale => {
  const slope = Number(payload.sclSlope);
  const intercept = Number(payload.sclInter);
  if (!Number.isFinite(slope) || slope === 0 || !Number.isFinite(intercept)) {
    throw new RangeError("Scalar volume rescale metadata must be finite with a non-zero slope.");
  }
  return { slope, intercept };
};

const positiveSafeIntegerOrNull = (value: number): number | null =>
  Number.isSafeInteger(value) && value > 0 ? value : null;

const nonNegativeSafeIntegerOrNull = (value: number): number | null =>
  Number.isSafeInteger(value) && value >= 0 ? value : null;

const scalarVolumeOffset = (
  payload: ScalarVolumeDataPayload,
  index: ScalarVolumeVoxelIndex
): { offset: number; bytesPerVoxel: number } | null => {
  const width = positiveSafeIntegerOrNull(payload.width);
  const height = positiveSafeIntegerOrNull(payload.height);
  const depth = positiveSafeIntegerOrNull(payload.depth);
  const x = nonNegativeSafeIntegerOrNull(index.x);
  const y = nonNegativeSafeIntegerOrNull(index.y);
  const z = nonNegativeSafeIntegerOrNull(index.z);
  const bytesPerVoxel = positiveSafeIntegerOrNull(payload.bytesPerVoxel);

  if (
    width == null ||
    height == null ||
    depth == null ||
    x == null ||
    y == null ||
    z == null ||
    bytesPerVoxel == null ||
    x >= width ||
    y >= height ||
    z >= depth
  ) {
    return null;
  }

  const voxelIndex = (z * height + y) * width + x;
  const offset = voxelIndex * bytesPerVoxel;
  if (offset + bytesPerVoxel > payload.data.byteLength) {
    return null;
  }
  return { offset, bytesPerVoxel };
};

const scalarVolumeSampleValue = (view: DataView, offset: number, payload: ScalarVolumeDataPayload): number => {
  const dtype = String(payload.dtype ?? "").trim().toLowerCase();
  const bytesPerVoxel = checkedPositiveSafeInteger(
    payload.bytesPerVoxel,
    "Scalar volume bytes per voxel"
  );
  if (dtype === "float32" || bytesPerVoxel === 4) {
    return view.getFloat32(offset, true);
  }
  if (dtype === "int16") {
    return view.getInt16(offset, true);
  }
  if (dtype === "uint16" || bytesPerVoxel === 2) {
    return view.getUint16(offset, true);
  }
  return view.getUint8(offset);
};

export const scalarVolumePayloadValueAt = (
  payload: ScalarVolumeDataPayload,
  index: ScalarVolumeVoxelIndex
): number | null => {
  const location = scalarVolumeOffset(payload, index);
  if (!location) {
    return null;
  }
  const view = new DataView(payload.data);
  const storedValue = scalarVolumeSampleValue(view, location.offset, payload);
  if (!Number.isFinite(storedValue)) {
    return null;
  }
  const { slope, intercept } = resolveScalarVolumeRescale(payload);
  const physicalValue = storedValue * slope + intercept;
  return Number.isFinite(physicalValue) ? physicalValue : null;
};

const scalarVolumeNormalizationRange = (
  payload: Pick<ScalarVolumePayload, "rawMin" | "rawMax">
): { rawMin: number; range: number } => {
  const rawMin = Number.isFinite(payload.rawMin) ? payload.rawMin : 0;
  const rawMax = Number.isFinite(payload.rawMax) && payload.rawMax > rawMin ? payload.rawMax : rawMin + 1;
  return { rawMin, range: rawMax - rawMin };
};

const parsePhysicalWindow = (
  enhancement: string | undefined,
  fallbackLow: number,
  fallbackHigh: number
): { low: number; high: number } => {
  const safeEnhancement = String(enhancement || "").trim();
  if (safeEnhancement.startsWith("hounsfield:")) {
    const parts = safeEnhancement.split(":");
    const center = Number(parts[1]);
    const width = Number(parts[2]);
    if (Number.isFinite(center) && Number.isFinite(width) && width > 0) {
      return { low: center - width / 2, high: center + width / 2 };
    }
  }
  return { low: fallbackLow, high: fallbackHigh };
};

/**
 * Map an ordered physical-intensity display window into stored voxel codes.
 * Negative NIfTI slopes reverse stored-code polarity, so the returned inversion
 * flag XORs that reversal with the user's explicit display inversion.
 */
export const resolveScalarStoredWindow = (
  payload: Pick<ScalarVolumePayload, "rawMin" | "rawMax" | "sclSlope" | "sclInter">,
  physicalLow: number,
  physicalHigh: number,
  userInvert = false
): ScalarVolumeWindow => {
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const rawMax = rawMin + range;
  const { slope, intercept } = resolveScalarVolumeRescale(payload);
  const mappedA = Number.isFinite(physicalLow) ? (physicalLow - intercept) / slope : rawMin;
  const mappedB = Number.isFinite(physicalHigh) ? (physicalHigh - intercept) / slope : rawMax;
  const endpointA = Number.isFinite(mappedA) ? mappedA : rawMin;
  const endpointB = Number.isFinite(mappedB) ? mappedB : rawMax;
  const storedLow = Math.min(endpointA, endpointB);
  const storedHigh = Math.max(endpointA, endpointB);
  return {
    low: storedLow,
    high: storedHigh > storedLow ? storedHigh : storedLow + Math.max(1e-6, range * 1e-4),
    invert: Boolean(userInvert) !== (slope < 0),
  };
};

/** Window uniforms in the normalized stored-code space uploaded to the GPU. */
export const resolveScalarVolumeWindow = (
  payload: Pick<ScalarVolumePayload, "rawMin" | "rawMax" | "sclSlope" | "sclInter">,
  enhancement: string | undefined,
  userInvert = false
): ScalarVolumeWindow => {
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const rawMax = rawMin + range;
  const { slope, intercept } = resolveScalarVolumeRescale(payload);
  const physicalA = rawMin * slope + intercept;
  const physicalB = rawMax * slope + intercept;
  const physicalWindow = parsePhysicalWindow(
    enhancement,
    Math.min(physicalA, physicalB),
    Math.max(physicalA, physicalB)
  );
  const storedWindow = resolveScalarStoredWindow(
    payload,
    physicalWindow.low,
    physicalWindow.high,
    userInvert
  );
  const normalizedLow = (storedWindow.low - rawMin) / range;
  const normalizedHigh = (storedWindow.high - rawMin) / range;
  const low = Number.isFinite(normalizedLow) ? normalizedLow : 0;
  const finiteHigh = Number.isFinite(normalizedHigh) ? normalizedHigh : 1;
  const high = finiteHigh > low ? finiteHigh : low + 1e-4;
  return { low, high, invert: storedWindow.invert };
};

const scalarVolumeOutputLength = (payload: ScalarVolumeDataPayload): number => {
  const voxelCount = preparedScalarVolumeByteLength(payload) / Uint16Array.BYTES_PER_ELEMENT;
  const bytesPerVoxel = checkedPositiveSafeInteger(
    payload.bytesPerVoxel,
    "Scalar volume bytes per voxel"
  );
  const availableVoxels = Math.floor(payload.data.byteLength / bytesPerVoxel);
  if (availableVoxels !== voxelCount) {
    throw new RangeError("Scalar volume body length does not match its delivered grid.");
  }
  return voxelCount;
};

/**
 * Auto-contrast window for a scalar volume in volume-rendering use, returned in
 * the normalized [0,1] space the GPU texture stores. Unlike a 2D display
 * auto-contrast (which brackets the full data range), a *volume* needs the low
 * cutoff ABOVE the background bulk: even a small per-voxel background opacity
 * accumulates into a flat fog over a deep z-stack. So we take a high low-percentile
 * (the background of fluorescence is the dominant low-intensity mode) and a high
 * high-percentile, computed from a subsampled histogram.
 */
export const computeScalarVolumeAutoContrast = (
  payload: ScalarVolumeDataPayload,
  lowPercentile = 0.78,
  highPercentile = 0.995
): { low: number; high: number } => {
  const bytesPerVoxel = checkedPositiveSafeInteger(
    payload.bytesPerVoxel,
    "Scalar volume bytes per voxel"
  );
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  const voxelCount = scalarVolumeOutputLength(payload);
  const BINS = 512;
  const histogram = new Float64Array(BINS);
  // Subsample for speed; a histogram is robust to it (cap ~2M samples).
  const stride = Math.max(1, Math.floor(voxelCount / 2_000_000));
  let sampled = 0;
  for (let voxelIndex = 0; voxelIndex < voxelCount; voxelIndex += stride) {
    const value = scalarVolumeSampleValue(view, voxelIndex * bytesPerVoxel, payload);
    const finite = Number.isFinite(value) ? value : rawMin;
    const normalized = Math.max(0, Math.min(0.999999, (finite - rawMin) / range));
    histogram[Math.floor(normalized * BINS)] += 1;
    sampled += 1;
  }
  if (sampled === 0) {
    return { low: 0, high: 1 };
  }
  const percentileBin = (p: number): number => {
    const target = sampled * Math.max(0, Math.min(1, p));
    let cumulative = 0;
    for (let i = 0; i < BINS; i += 1) {
      cumulative += histogram[i];
      if (cumulative >= target) {
        return i;
      }
    }
    return BINS - 1;
  };
  const low = percentileBin(lowPercentile) / BINS;
  const highBin = percentileBin(highPercentile);
  const high = Math.min(1, (highBin + 1) / BINS);
  // Guarantee a usable, non-degenerate window.
  if (high <= low) {
    return { low: Math.min(low, 0.95), high: 1 };
  }
  return { low, high };
};

export const scalarVolumePayloadToTextureBytes = (payload: ScalarVolumeDataPayload): Uint8Array => {
  const bytesPerVoxel = checkedPositiveSafeInteger(
    payload.bytesPerVoxel,
    "Scalar volume bytes per voxel"
  );
  const output = new Uint8Array(scalarVolumeOutputLength(payload));
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  for (let voxelIndex = 0; voxelIndex < output.length; voxelIndex += 1) {
    const offset = voxelIndex * bytesPerVoxel;
    const value = scalarVolumeSampleValue(view, offset, payload);
    const finiteValue = Number.isFinite(value) ? value : rawMin;
    const normalized = Math.max(0, Math.min(1, (finiteValue - rawMin) / range));
    output[voxelIndex] = Math.max(0, Math.min(255, Math.round(normalized * 255)));
  }
  return output;
};

/**
 * Convert a scalar volume to normalized 16-bit half-float samples for GPU upload.
 *
 * Medical CT/MRI volumes span a huge raw range (air ~-1024 HU through bone/metal
 * >3000 HU). Quantizing that to 8 bits collapses the entire brain-tissue band
 * (CSF ~0-15 HU, parenchyma ~20-45 HU) into ~3 code values, erasing the
 * ventricle/parenchyma boundary before windowing can ever stretch it. Storing the
 * normalized value at half-float precision keeps ~11 bits of mantissa, i.e.
 * sub-HU resolution across the full range, so soft-tissue contrast survives and
 * window/level stays a cheap GPU uniform. R16F linear filtering is core in WebGL2.
 */
export const scalarVolumePayloadToHalfFloat = (payload: ScalarVolumeDataPayload): Uint16Array => {
  const bytesPerVoxel = checkedPositiveSafeInteger(
    payload.bytesPerVoxel,
    "Scalar volume bytes per voxel"
  );
  const output = new Uint16Array(scalarVolumeOutputLength(payload));
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  for (let voxelIndex = 0; voxelIndex < output.length; voxelIndex += 1) {
    const offset = voxelIndex * bytesPerVoxel;
    const value = scalarVolumeSampleValue(view, offset, payload);
    const finiteValue = Number.isFinite(value) ? value : rawMin;
    const normalized = Math.max(0, Math.min(1, (finiteValue - rawMin) / range));
    output[voxelIndex] = DataUtils.toHalfFloat(normalized);
  }
  return output;
};

/**
 * Abortable, cooperatively-yielding half-float conversion for renderer loads.
 * Yielding between bounded chunks lets a superseded React generation cancel
 * before it allocates/uploads a complete stale texture.
 */
export const scalarVolumePayloadToHalfFloatAsync = async (
  payload: ScalarVolumeDataPayload,
  signal?: AbortSignal,
  chunkVoxels = 262_144
): Promise<Uint16Array> => {
  const throwIfAborted = () => {
    if (signal?.aborted) {
      throw new DOMException("The volume conversion was aborted.", "AbortError");
    }
  };
  throwIfAborted();
  const bytesPerVoxel = checkedPositiveSafeInteger(
    payload.bytesPerVoxel,
    "Scalar volume bytes per voxel"
  );
  const output = new Uint16Array(scalarVolumeOutputLength(payload));
  const { rawMin, range } = scalarVolumeNormalizationRange(payload);
  const view = new DataView(payload.data);
  const safeChunk = Math.max(1, Math.floor(Number(chunkVoxels) || 1));
  for (let chunkStart = 0; chunkStart < output.length; chunkStart += safeChunk) {
    const chunkEnd = Math.min(output.length, chunkStart + safeChunk);
    for (let voxelIndex = chunkStart; voxelIndex < chunkEnd; voxelIndex += 1) {
      const offset = voxelIndex * bytesPerVoxel;
      const value = scalarVolumeSampleValue(view, offset, payload);
      const finiteValue = Number.isFinite(value) ? value : rawMin;
      const normalized = Math.max(0, Math.min(1, (finiteValue - rawMin) / range));
      output[voxelIndex] = DataUtils.toHalfFloat(normalized);
    }
    throwIfAborted();
    if (chunkEnd < output.length) {
      await new Promise<void>((resolve) => globalThis.setTimeout(resolve, 0));
      throwIfAborted();
    }
  }
  return output;
};

export const prepareScalarVolume = async (
  payload: ScalarVolumePayload,
  signal?: AbortSignal
): Promise<PreparedScalarVolume> => {
  const autoWindow = computeScalarVolumeAutoContrast(payload);
  const textureData = await scalarVolumePayloadToHalfFloatAsync(payload, signal);
  return {
    width: payload.width,
    height: payload.height,
    depth: payload.depth,
    dtype: payload.dtype,
    bytesPerVoxel: payload.bytesPerVoxel,
    rawMin: payload.rawMin,
    rawMax: payload.rawMax,
    channel: payload.channel,
    time: payload.time,
    sourceWidth: payload.sourceWidth,
    sourceHeight: payload.sourceHeight,
    sourceDepth: payload.sourceDepth,
    downsampleX: payload.downsampleX,
    downsampleY: payload.downsampleY,
    downsampleZ: payload.downsampleZ,
    previewPolicy: payload.previewPolicy,
    sclSlope: payload.sclSlope,
    sclInter: payload.sclInter,
    textureData,
    autoWindow,
  };
};
