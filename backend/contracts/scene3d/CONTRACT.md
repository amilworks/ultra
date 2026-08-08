# Lens `scene3d` — frozen contract, v1

The wire contract between the Python derive worker, the Go control plane, and the
Lens renderer. Backend and frontend are built in parallel against this document;
anything not written here is not guaranteed.

Everything below is grounded in measurements of two real files. Those measurements
are reproduced in `Appendix A` because several of them contradict what the formats
nominally claim.

---

## 1. Scope

| Species | Source | Rendered as |
| --- | --- | --- |
| 3D Gaussian splats | Binary INRIA / Postshot `.ply` | `@sparkjsdev/spark` `SplatMesh` |
| Dense point maps | COLMAP/HLOC `fused.ply`, generic binary `.ply` with `x,y,z[,rgb]` | `THREE.Points` |
| Sparse point maps + posed cameras | A ZIP containing a COLMAP/HLOC model (`cameras` + `images` + optional `points3D`) | `THREE.Points` + `THREE.LineSegments` frusta |
| NeRF | — | **Not rendered.** Only its export is. Stated explicitly in `limitations`. |

Compact splat containers (`.spz`, `.splat`, `.ksplat`, `.sog`) are deliberately not
advertised yet. Spark can consume several of them, but Ultra's server-side, source-bound
converter does not. Sending one to the PLY parser would be a false support claim and a
multi-gigabyte browser download. Each format becomes supported only with a bounded,
fixture-tested converter. A raw COLMAP directory is recognized so it never reaches the
image service, but it must be archived as ZIP to give the worker one immutable byte
stream and SHA-256 identity.

---

## 2. World frame — the one rule everything else depends on

**The source world frame is the canonical scene frame. It is never rotated.**

3DGS trains directly on COLMAP `points3D`, so a splat `.ply` and the sparse model it
came from are already in identical world coordinates. Keeping that frame means points,
cameras and splats co-register with zero registration math.

The RDF→RUB flip (`diag(1,−1,−1)`) is applied in **exactly one place**: constructing a
camera's orientation basis. It is never applied to a scene-graph node holding splats.

> **Why this is a hard rule.** Spherical-harmonic coefficients are defined in the
> asset's own frame. Rotating a node that holds an SH-bearing splat asset rotates the
> geometry but not the coefficients; correcting it requires a Wigner-D rotation of the
> SH bands. Skip that and every odd band (l=1, l=3) sign-flips: geometry looks perfect,
> view-dependent highlights appear on the wrong side of every surface, and the error is
> invisible in a screenshot — it only shows while orbiting.

"Up" is a **view hint** applied to the camera controls, carried in the manifest as
`world.up_axis` with a provenance string. It is never baked into geometry.

`up_axis` is intentionally an unsigned dominant-axis hint in v1. The renderer resolves
its direction from the source convention: a heuristic Y hint on legacy PLY/COLMAP data
means **−Y is up** (their common right/down/forward convention), while Z means +Z and
non-RDF Y-up formats retain +Y. A `declared` or `user` basis is authoritative and is not
overridden by that file-family fallback. Initial framing and Reset view place the camera
on the positive side of this resolved up direction; source positions, splat rotations,
and spherical-harmonic coefficients remain untouched.

---

## 3. Activation domains

PLY stores *raw model parameters*. Other formats store *post-activation* values. A
converter that copies fields across without checking produces a scene that is either
invisible or a field of giant blurs.

| Field | INRIA/Postshot `.ply` | Our wire format |
| --- | --- | --- |
| scale | `log` — apply `exp` | **`log`** (Spark re-logs it) |
| opacity | `logit` — apply `sigmoid` | **`[0,1]`** |
| rotation | unnormalized quat | **unit quat** |
| axes | source frame | **source frame, declared** |

Every layer therefore declares `source_frame` and `activation_domain`. Applying the
handedness flip twice is exactly as broken as never applying it.

---

## 4. Chunk wire formats

`UPC1` is the production point-cloud path. `USX1` remains the bounded legacy/preview
path and a fixture-tested fallback, but production Gaussian uploads use the native RAD
tree in §5.1. A raw uniform subset is not appearance-preserving for Gaussian splats.

Two formats, both **planar** (not interleaved) so the browser constructs typed-array
views with zero copying and zero per-element JS. Both use a 64-byte header, which keeps
every following array 4- and 8-byte aligned.

### 4.1 Common header (64 B)

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic — `USX1` or `UPC1` |
| 4 | 2 | `version` u16 = 1 |
| 6 | 2 | `flags` u16 |
| 8 | 4 | `count` u32 — elements in this chunk |
| 12 | 4 | `sh_degree` u32 — **measured**, not declared |
| 16 | 12 | `bbox_min` 3×f32 — chunk-local |
| 28 | 12 | `bbox_max` 3×f32 — chunk-local |
| 40 | 12 | `origin` 3×f32 — chunk origin in world coordinates |
| 52 | 12 | reserved, zero |

`origin` is the world-space translation to apply to the whole chunk. Coordinates inside
the chunk are **chunk-local**. See §5.

### 4.2 `USX1` — splats, Spark `ExtSplats` layout

Payload is literally what Spark's `encodeExtSplat` writes, so the browser hands the
buffers straight to `new ExtSplats({ extArrays: [a, b], numSplats })`.

```
64            + count*16   extA : Uint32Array, 4 words/splat
64+count*16   + count*16   extB : Uint32Array, 4 words/splat
```

Per splat, little-endian:

```
extA[0] = float32 bits of x        <- FULL f32, chunk-local
extA[1] = float32 bits of y
extA[2] = float32 bits of z
extA[3] = half(opacity)            <- post-sigmoid, [0,1]

extB[0] = half(r)        | half(g) << 16          <- DISPLAY-REFERRED, see below
extB[1] = half(b)        | half(ln scaleX) << 16  <- ln of POST-exp scale
extB[2] = half(ln scaleY)| half(ln scaleZ) << 16
extB[3] = encodeQuatOctXy1010R12(x,y,z,w)         <- unit quat, xyzw
```

**We target `ExtSplats`, not `PackedSplats`.** `PackedSplats` is 16 B/splat but stores
centres as float16; measured against the real 14.5M-splat file, 97.4% of splats would be
displaced by more than their own thin-axis extent — the surface-normal direction — which
detaches splats from the surfaces they represent. `ExtSplats` costs 2× the bytes and
keeps centres exact. `PackedSplats` may later be offered as an explicit, labelled
"performance" mode; it is not the default and never silently substituted.

Splat colour is **display-referred and NOT linearised**: `0.5 + C0*f_dc`, with
`C0 = 0.28209479177387814`. Components outside [0,1] remain representable in float16 and
are preserved through Gaussian compositing, matching Spark's `encodeExtSplat`. Their
fraction is counted and reported; raster outputs clamp only at the final display boundary.

> **This is deliberately the opposite of the point path in §4.3, and the asymmetry is
> the correct answer, not an oversight.** The governing rule is that our derived path
> must be indistinguishable from Spark loading the same `.ply` directly. Every Spark
> input path does exactly this and no linearisation:
> ```
> PlyReader : r = item.f_dc_0 * SH_C0 + 0.5
> SPZ       : r = (byte/255 - 0.5) * (SH_C0/0.15) + 0.5
> SOG       : rLookup = codebook.map(x => SH_C0 * x + 0.5)
> ```
> Spark's shader therefore consumes display-referred values, matching INRIA's reference
> rasterizer. Linearising here would render every splat too dark — 0.5 would arrive as
> 0.214. Points go the other way (§4.3) because they are consumed by three.js's own
> `PointsMaterial` vertex-colour path, which assumes the linear working space.
>
> A regression test asserts our encoder reproduces `PlyReader`'s value bit-for-bit.

### 4.3 `UPC1` — point clouds

```
64            + count*12   positions : Float32Array, xyz, chunk-local
64+count*12   + count*4    colors    : Uint8Array,  rgba
```

Colours are **sRGB** here, source-faithful, converted to linear in the shader. Point
colours come from source photographs and are sRGB-encoded; three.js assumes vertex-colour
attributes are already in the linear working space and would otherwise double-encode
them — measured, sRGB 0.2 renders at ≈0.48. The renderer owns this conversion in one
documented place. `flags` bit 0 set means alpha is meaningful; otherwise alpha is 255.

---

## 5. Bounded streaming and level of detail

### 5.1 Gaussian splats: reconstructed spatial LoD

Production splats are converted offline with Spark 2.1.0's pinned `build-lod --quality
--rad-chunked` path. The quality builder merges nearby Gaussian distributions using a
Bhattacharyya-distance hierarchy. Coarse nodes therefore preserve aggregate coverage,
colour, opacity, scale, and orientation; they are not holes left by deleting source
rows. The output is one `scene-lod.rad` header plus relative `scene-lod-N.radc` pages.

The browser uses Spark's native view-adaptive traversal and a bounded page pool. It
chooses spatial nodes for the current camera, refines visible regions, and evicts pages
outside the working set. The readout says `adaptive LoD · N active · M source`; `N`
counts reconstructed active nodes and must never be described as `N of M` source rows.

Before invoking the native builder, the worker performs a sequential full-source scan of
every declared `f_rest_*` property. `measured_sh_degree` is therefore the highest band
with at least one non-zero coefficient anywhere in the immutable source, not a bounded
sample. Spark receives `--max-sh=measured_sh_degree`; only bands proven zero across every
source splat are omitted. This exactness is required because even one spatially-localized
coefficient is view-dependent signal. A source with a non-finite coordinate fails closed
before the native builder because Ultra cannot otherwise prove which rows the external
tree contains.

### 5.2 Points and the explicit legacy preview: additive density tiers

The derive makes two sequential passes over PLY: one to validate the declared record
count and measure exact/robust bounds, then one to encode. Resident memory is bounded by
one source read batch plus `tier_count × max_splats_per_chunk`; it never materializes a
whole-scene table or octree. Production chunks contain at most **50,000** elements.

For point clouds (and only for an explicitly requested legacy splat preview), every
finite source record is assigned exactly once to an additive density tier by a
stable SplitMix64 hash of its source row. Tier 0 is a deterministic, uniform sample of
the whole source (target 100k splats or 280k points), not the first N records or one
spatial corner. Later tiers refine it; the union of all tiers is the full finite source.
Non-finite coordinates are counted and reported because they cannot be placed in space.

`tiers[k]` lists the chunks added at level `k`. The browser loads only a **complete
cumulative tier** whose estimated GPU residency fits its device budget. It never clips
a chunk, loads half a tier, or calls a partial prefix “the scene.” A desktop budget is
108 MB estimated residency; a mobile/coarse-pointer budget is 24 MB. The provenance
readout states the displayed and source counts.

Each output chunk records a float32 world `origin`; encoded positions are chunk-local.
The origin is chosen on the source float32 grid and verified with the same float32
addition the shader performs. Axes that cannot round-trip exactly fall back to origin
zero. Float64 inputs report their measured maximum conversion error in `quantization`.

**No silent decimation, ever.** Every finite point record is present in the full tier
union, and a reduced-density display is labelled as such. Gaussian RAD displays report
their adaptive-node count separately, as described above.

---

## 6. Manifest

`GET /v2/uploads/{file_id}/scene3d/manifest` → `application/json`.

```jsonc
{
  "schema": "ultra.scene3d.v1",
  "generator_revision": "scene3d-rad-v3",
  "scene_kind": "splat" | "pointcloud" | "colmap",
  "source": {
    "format": "ply",
    "writer": "postshot",              // parsed from PLY comments when present
    "vertex_count": 14469103,
    "bytes": 3414709820,
    "sha256": "…",                   // catalog identity used by the derive
    "declared_sh_degree": 3,
    "measured_sh_degree": 0,           // MEASURED. See Appendix A.
    "stride_bytes": 236
  },
  "world": {
    "units": "arbitrary",              // never "meters" without a user reference
    "up_axis": "unknown" | "y" | "z",
    "up_axis_basis": "unknown" | "heuristic" | "declared" | "user",
    "frame": "source",
    "bbox": [minx, miny, minz, maxx, maxy, maxz],
    "bbox_robust": [ ... ]             // 1st..99th percentile. FRAME THE CAMERA ON THIS.
  },
  "layers": [{
    "type": "splats" | "points" | "cameras",
    "encoding": "spark-rad-v1" | "usx-v1" | "upc-v1" | "json",
    "total": 14469103,
    "chunks": [],                         // RAD pages are listed under lod.chunks
    "tiers": [],                          // reconstructed LoD is not an additive subset
    "activation_domain": "post",
    "source_frame": "rdf" | "rub" | "source",
    "quantization": {
      "center": "spark-rad-resolved-per-asset",
      "scale": "spark-rad-resolved-per-asset",
      "rotation": "spark-rad-resolved-per-asset",
      "color": "spark-rad-resolved-per-asset",
      "out_of_range_color_fraction": 0.0031 // preserved for splat compositing
    },
    "lod": {
      "format": "spark-rad-v1",
      "method": "bhatt-lod-quality",
      "builder_revision": "spark-build-lod-2.1.0-f22236f",
      "paged": true,
      "source_elements": 14469103,
      "max_sh_degree": 0,                // exact retained degree in the RAD artifact
      "header": {"name": "scene-lod.rad", "bytes": 1234},
      "chunks": [{"name": "scene-lod-0.radc", "bytes": 123456}]
    }
  }],
  "limitations": ["..."],              // rendered verbatim in the provenance panel
  "service_urls": {
    "chunk": "/v2/uploads/{id}/scene3d/chunk/{index}",
    "lod": "/v2/uploads/{id}/scene3d/lod/scene-lod.rad",
    "download": "/v2/resources/{id}/download"
  }
}
```

`limitations` is the CIFTI honesty field. It is a list of plain sentences, shown to the
user, stating what the viewer is *not* doing.

### Why `bbox_robust` exists

Cameras frame on `bbox_robust`, never on `bbox`. Dense reconstructions routinely carry a
handful of far-field outliers, and the effect is not marginal — measured on the COLMAP
corridor fixture:

```
full bbox diagonal      3453.6
middle-99% diagonal       32.8      <- 0.9% of it
```

A camera fitted to `bbox` renders the actual scene as a speck a few hundred pixels wide,
which reads as "the viewer is broken" rather than "this file has outliers". The derive
computes the percentile box (stride-sampled, so it costs nothing) because the viewer
cannot: it would have to download every chunk first. `bbox` is still reported, so the
outliers remain discoverable and are still drawn when in view.

The initial camera is fitted to the robust box, but its far plane still contains the
exact full box so outliers are genuinely discoverable. When that honest near/far ratio
reaches 10,000, the renderer enables the logarithmic-depth path shared by three.js and
Spark's splat shaders. Without it, every fragment on the measured corridor lands near
NDC `z = 1`, points occlude each other, and the scene collapses to a few hundred lit
pixels; clipping the outliers instead would only hide the error.

---

## 7. Derive job

Mirrors `image.derive_pyramid` exactly, on the same worker node.

- subject `ultra.image.jobs`, type `scene.derive`
- payload `{resource_id, src_path, dst_dir, source_sha256, source_size_bytes,
  splat_delivery, max_splats_per_chunk, tier_count, preview_splats, preview_points}`
- production splat output: `{dst_dir}/manifest.json` + `scene-lod.rad` +
  `scene-lod-N.radc` + `poster.png`
- point/legacy output: `{dst_dir}/manifest.json` + `chunk_{n:05d}.bin` + `poster.png`
- permanent failure writes `{dst_dir}.failed`; the control plane honours it as backoff
- redelivery capped by the existing `max_deliver`

`dst_dir` is exactly `{resource_id}__scene3d.v3.sha256-{source_sha256}`. The worker verifies
the catalog digest and byte count before reading, serializes expensive conversion with
the shared `scene3d.work.lock`, derives into a sibling temporary directory, revalidates
the source generation, then atomically renames the complete directory while holding the
resource lifecycle lock. A redelivery reuses a complete matching generation. Failure
markers contain stable codes and source identity, never local source paths.

The control plane **never parses a scene file in the request path.** It sniffs headers
(bounded read), enqueues, and serves derived bytes.

---

## 8. Go surface

| Route | Returns |
| --- | --- |
| `GET /v2/uploads/{id}/viewer` | existing ladder, new arm emitting `kind:"scene3d"` |
| `GET /v2/uploads/{id}/scene3d/manifest` | the manifest, `ETag` + `Cache-Control` |
| `GET /v2/uploads/{id}/scene3d/chunk/{n}` | chunk bytes, `Range`, strong `ETag`, short private revalidation |
| `GET /v2/uploads/{id}/scene3d/lod/{artifact}` | canonical RAD/RADC bytes, authenticated `Range`, strong `ETag` |

The arm must be added to **both** `handleGetUploadViewerService`
(`imageservice_viewer.go:296`) and `handleGetUploadViewer` (`handlers.go:8069`) — the
second runs when the image-service is unconfigured, and missing it means the modality
works locally and breaks in production.

Chunk and RAD responses are served with the same in-flight byte budget so concurrent
scene opens cannot exhaust the edge node. RAD names accept only `scene-lod.rad` and a
canonical `scene-lod-N.radc`; the route cannot address an arbitrary derived file.

---

## 9. Frontend surface

- `kind: "scene3d"` added to the `UploadViewerInfo.kind` union, `scene3d?: Scene3dViewerData` slot.
- `normalizeScene3dViewerInfo` in `viewerManifest.ts`, dispatched before the generic path.
- One branch in `UploadViewerSheet.tsx` beside `isHdf5Viewer` / `isCiftiViewer`, lazy-loaded.
- Spark goes in its own `vendor-spark` manual chunk, excluded from `modulePreload`,
  with its own `check-bundle-budgets.mjs` entry. It must not enter `vendor-three`.

### Pure modules (no WebGL, exhaustively unit-tested)

| Module | Responsibility |
| --- | --- |
| `sceneFrame.ts` | COLMAP world-to-camera inversion, `−Rᵀt`, wxyz→xyzw, RDF→RUB basis |
| `sceneIntrinsics.ts` | COLMAP camera models → projection matrix incl. principal-point offset |
| `sphericalHarmonics.ts` | INRIA's exact constant and sign table; planar↔interleaved |
| `splatCovariance.ts` | quat + log-scale → 3D covariance |
| `sceneColor.ts` | sRGB↔linear, DC→base colour, clamp accounting |
| `sceneChunks.ts` | header parse, typed-array views, tier selection |
| `sceneBudget.ts` | device-pixel-ratio cap and element ceilings |

### Rendering rules

- Splat pass: `depthTest: true`, `depthWrite: false`. 3DGS initialises Gaussians *at*
  the sparse points, so opaque depth-writing point sprites punch holes through the splat
  cloud at exactly the densest geometry. Points and splats are **mutually exclusive by
  default**, not overlaid.
- Sort key is **view-space z**, not Euclidean camera distance. The two diverge off-axis
  and produce stable, orientation-dependent seams near frustum edges under wide FOV.
- Camera frusta are built from an explicit projection matrix, never
  `THREE.PerspectiveCamera(fov, aspect)` — that is structurally a symmetric frustum and
  cannot represent a principal-point offset.
- `devicePixelRatio` capped at 2, mirroring `resolveVolumePixelRatio`.
- Rendering is invalidation-driven. Spark's asynchronous update is awaited at complete
  tier boundaries; camera, resize and dirty events request one frame. There is no
  permanent `requestAnimationFrame` loop while the scientist is idle.
- Production splats use one `PagedSplats` / `SplatMesh`, Spark's global compositing and
  native view-adaptive traversal. Request headers and credentials are supplied to every
  RAD range fetch; no API key is placed in a URL.

---

## Appendix A — measured ground truth

Both files probed directly; every number below is measured, not assumed.

### `fused_model1_superpoint.ply` — dense point cloud

```
format        binary_little_endian     vertices   2,068,089
properties    x y z nx ny nz red green blue
stride        27 B                     data off   248 B  (CRLF header)
size          55,838,651 B  (= 248 + 2068089*27, exact)
extent        1254.63 x 354.43 x 3198.07   diagonal 3453.6 (arbitrary units)
bbox          min [-39.30, -345.45, 1.04]  max [1215.33, 8.99, 3199.11]
```

A long corridor scan. The 3454-unit diagonal is why point positions stay f32 and why
chunking matters for this file too.

### `willaGlobalonlyDrone-deleted_env-1.ply` — Gaussian splats

```
format        binary_little_endian     vertices   14,469,103
properties    59  (x y z, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3)
stride        236 B                    data off   1512 B
size          3,414,709,820 B  (= 1512 + 14469103*236, exact)
comment       postshot.anti_aliasing=1
extent        122.20 x 27.10 x 121.09  diagonal   174.2
bbox          min [-61.28, -4.62, -50.35]  max [60.92, 22.48, 70.74]
```

Three findings that change the implementation:

1. **Stride is 236 B, not the canonical 248 B.** Postshot omits the `nx,ny,nz`
   properties that INRIA's writer emits. Property offsets **must** be derived from the
   header; a hardcoded layout silently misreads every field.

2. **Declared SH degree 3, measured SH degree 0.** A sequential scan of all 14,469,103
   source splats proves every one of the 45 `f_rest_*` coefficients is exactly `0.0`.
   The file allocates and zeroes the full degree-3 layout, so 180 of 236 bytes per splat
   (76% of 3.41 GB) is padding. The v3 RAD generation safely passes `--max-sh=0` and does
   not transmit those proven-empty planes. *Do not hardcode this conclusion* — the same
   full-source scan retains real degree-1, degree-2, and degree-3 signal, including a
   coefficient that occurs only in the final source record.

3. **Activation domains confirmed empirically.**
   ```
   opacity   min -4.1553  med  0.6668  max 13.1980   -> logit, sigmoid(med)=0.661
   scale_0   min -10.607  med -4.6392  max  0.5901   -> log,   exp(med)=0.00967
   |quat|    min  1.0000  med  1.0000  max  1.0000   -> already normalized
   f_dc_0 -> 0.5+C0*dc:  min -0.511  med 0.513  max 2.704  -> preserve through compositing
   ```
   Final framebuffer output clamps to its display gamut. Postshot normalises
   quaternions; INRIA's writer does not. Normalise defensively.

### Why `ExtSplats` and not `PackedSplats`

float16 ULP vs each splat's own thin-axis extent, 120,000 randomly sampled splats.
(The sample's coordinate range is narrower than the file's true bbox above, so the
world-coordinate row is if anything optimistic — the real conclusion is stronger.)

```
centre coords          med f16 ULP    p99 ULP   % splats with ULP > thin-axis scale
world (as-is)              0.01562    0.03125                                97.4%
chunk-local 16u            0.00781    0.00781                                90.2%
chunk-local  8u            0.00391    0.00391                                74.0%
chunk-local  4u            0.00195    0.00195                                51.0%
chunk-local  1u            0.00049    0.00049                                 5.4%

thin-axis scale       p05 0.00046   med 0.00169   p95 0.01041
```

f32 centres cost 16 extra bytes per splat and make the entire row irrelevant.

### Measured cost of the encodings we *do* accept

Independently reimplemented from Spark's `encodeQuatOctXy1010R12` / `decodeQuatOctXy1010R12`
and measured, rather than estimated:

```
quaternion oct-10-10-12, 60,000 random unit quats, geodesic rotation error
  median 0.1355 deg   p95 0.2799   p99.9 0.4044   max 0.4509

half(ln scale) -> linear scale, relative error over the file's real ln range [-10.7, 0.6]
  median 0.0636 %     p99 0.3743   max 0.3914
```

Both are well inside what the reconstruction itself resolves, and both are reported in
the manifest so the viewer can state them rather than imply exactness.
