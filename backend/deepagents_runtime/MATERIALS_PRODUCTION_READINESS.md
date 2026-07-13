# Ultra Materials Science Production-Readiness Contract

Date: 2026-07-12
Status: implementation in progress; **not production-promoted**
Release owner: Ultra Deep Agents runtime

## Decision

Ultra will keep the existing materials capability, but narrow it to the area it
actually supports well: microstructure, EBSD, DREAM.3D/HDF5, stereology, and
orientation analysis. Two focused companion skills will cover:

1. crystal structure, symmetry, defects, thermodynamics, and CALPHAD; and
2. characterization, including XRD and spectroscopy.

We will not create a monolithic materials agent or a separate materials
orchestrator. The general coordinator remains responsible for the workflow;
deterministic validation records and release gates provide the independent
checks. Heavy simulation (DFT, MD, phonons, and MPI/HPC solvers) remains
explicitly unavailable until the corresponding
engines, input validation, provenance, schedulers, and reference benchmarks are
present.

No release may describe the materials capability as production-ready until the
release/control-plane identity, live PostgreSQL CALPHAD ledger qualification,
full production sandbox, signed isolation evidence, designated live traces, and
MatTools results of at least **80% Function Runnable Rate** and **60% strict
scientific Task Success Rate** have all passed through the real Ultra runtime.
The readiness aggregator is not the final release authority: even when every
evidence gate passes, its JSON is only an **evidence-qualified candidate for
attestation**. A full production claim additionally requires a sanitized public
promotion envelope attested by GitHub/Sigstore and a final verifier that both
validates that attestation under the exact repository/workflow/ref/source-SHA
policy and revalidates the exact restricted evidence closure.

## Bounded first-class capabilities added on 2026-07-13

The product tool surface now exposes narrowly qualified operations instead of
asking the model to rediscover or reimplement their numerical APIs:

- crystal plasticity has typed canonical slip-family/resolved-shear/Schmid
  analysis and a separate CPFE input-contract validator. Geometry and contract
  readiness are not a constitutive integration, convergence result, stress-strain
  prediction, or qualified CPFE solver binding;
- degradation has five deterministic reducers: a Mode-I LEFM applicability
  screen, leakage-resistant Paris-law calibration with held-out interpolation,
  Norton/Arrhenius secondary-creep evaluation, linear/parabolic isothermal
  oxidation mass gain, and Faraday uniform-corrosion conversion. None predicts
  component failure, fatigue/creep/corrosion life, localized attack, oxide
  thickness, or ASTM compliance;
- advanced characterization has convention-explicit diffraction profile metrics
  and proper rigid registration with a fixed, complete held-out correspondence
  partition. These operations do not run Rietveld refinement, identify phases,
  discover feature correspondences, segment data, validate indexing, or establish
  physical/chemical identity;
- processing support discovery is a zero-argument typed boundary. It distinguishes
  the qualified Scheil and isolated diffusion/back-diffusion/KWN paths from phase
  field and coupled moving-interface work that still requires an external qualified
  HPC adapter. It never substitutes a toy solver; and
- selected sensor-series inspection now has a production-worker core dependency
  closure at NumPy 1.26.4, Zarr 3.1.5, and numcodecs 0.16.5. The worker image build
  opens and validates a real tiny ``ultra.sensor-series.v1`` Zarr, including values,
  calibration identity, quality flags, and a bounded min/max envelope. This does
  not upgrade declared cross-resource links or clock synchronization to verified
  lineage without the required out-of-band catalog evidence.

Routing is selective: a generic materials request carries none of these eight
degradation/characterization/processing schemas; an applicable request receives
only its five-tool, two-tool, or zero-argument group. In the focused qualification
observed for this change, 25 direct bounded-tool tests, 30 natural/adversarial
routing tests, 80 sensor/NGFF identity tests, and a 216-test cross-surface
agent/tool regression set passed. Aggregate gate totals and immutable release
hashes remain pending the final coordinated release run and must be updated from
that evidence rather than inferred from these focused counts.

The required three-trial MatTools campaign has not been run. No current report
establishes either promotion score; the thresholds remain at least **118/147
runnable parent attempts** and **249/414 strict scientific subtask attempts**,
with per-trial floors of **40/49 runnable parents** and **83/138 strict
scientific subtasks**. The separately retained official-upstream scientific
score must also reach **249/414**.

## Current assessed score

These are implementation-maturity scores, not release verdicts. They are
deliberately discounted for unrun independent promotion evidence and incomplete
release/security proof.

| Dimension | Weight | Score | Current evidence |
| --- | ---: | ---: | --- |
| Supported materials breadth | 15% | 7.5/10 | Strong microstructure/EBSD, crystallography, idealized XRD, defect geometry, and bounded CALPHAD; no DFT/MD/phonon/HPC engines |
| Skill structure and routing | 15% | 7.5/10 | Three focused skills, mandatory prompt-inferred routing, and a passing same-prompt live trace; no extra materials orchestrator |
| Deterministic correctness | 15% | 7.5/10 | Exact 13-check domain suite and live eight-validator CALPHAD record pass; clean release-image evidence is pending |
| CALPHAD backend | 20% | 8.0/10 | Typed pycalphad path, TDB/DAT support, immutable input/evidence retention, execute-only PostgreSQL writers, quotas, and exact replay |
| Independent benchmark strength | 20% | 3.5/10 | CALPHAD calibration and post-assessment holdout pass; the decisive MatTools campaign is unrun |
| Production operations and security | 15% | 3.0/10 | Promotion gates and typed BisQue intent/lease/receipt barriers exist, but clean release evidence, trust anchor, signed isolation/WORM proof, attestation, and atomic authorization/receipt claiming remain open |

The weighted platform score is therefore **6.1/10 for implementation maturity**
and approximately **4.5/10 for production readiness**. The CALPHAD subsystem
alone is **8.0/10 as a research/engineering implementation**, but only about
**6.5–7.0/10 as a production-qualified service** until a clean, attested
release run exists. The accurate product label remains **materials science
research preview**.

## Regression and performance baseline

The 2026-07-12 post-integration regression sweep passed the full Go control
plane (5.00 seconds), 1,167 Deep Agents tests with 68 intentional skips (56.46
seconds), 791 frontend unit tests plus lint/typecheck/build/bundle budgets and
four viewport smokes, and 323 materials readiness/promotion/MatTools contract
tests. The pinned hardened materials image passed all 38 non-skipped CALPHAD
runtime/CLI tests in 4.88 seconds; its real typed inspection-plus-equilibrium
test took 0.74 seconds. All 13 non-skipped materials scientific invariants
passed in 11.17 seconds; the Al-Co-W checkpoint took 1.45 seconds. The slowest
full-runtime tests remain unrelated imaging benchmarks at 19.92 and 13.60
seconds.

New test-only Go microbenchmarks establish a five-run Apple M4 Max baseline
after 501 retained validation events: append median 20,513 ns/op (5,616 B/op,
61 allocations), exact idempotent retry 18,713 ns/op (3,440 B/op, 56
allocations), a 500-row keyset page 699,449 ns/op (1,104,913 B/op, 5,053
allocations), and retained-evidence replay 6,256 ns/op (2,200 B/op, 8
allocations). These are comparative MemoryStore baselines, not PostgreSQL SLOs.
Production qualification still needs repeatable live-PostgreSQL p50/p95/p99
measurements for revision creation, validation append, evidence replay, and
keyset pagination on the target runner. CI must compare distributions on the
same runner/image rather than enforce a flaky single wall-clock assertion.

## Production trust boundary

The promotion path deliberately separates scientific evidence from its public
attestation:

1. The protected qualification job runs on a dedicated ephemeral Linux ARM64
   runner because the locked MatTools evaluator is `linux/arm64`. It produces an
   exact, content-addressed evidence-root manifest and a sanitized promotion
   envelope. The qualification job has no GitHub OIDC or repository-write
   permission and is capped at 23 hours so its GitHub token cannot expire before
   the sanitized envelope handoff. A campaign that cannot close within that
   budget remains blocked pending an explicit continuation design.
2. Raw MatTools prompts, submitted candidate code, evaluator logs/workbooks,
   generated artifacts, and Ultra traces remain in access-controlled,
   encrypted, write-once/read-many (WORM) storage. They must never be uploaded
   as public GitHub Actions artifacts or attached to a public release. The
   evidence-root manifest binds every retained regular file by relative path,
   SHA-256, and byte size; symlinks, traversal, missing files, extra files, and
   changed bytes fail verification.
3. The public envelope contains only reviewed, non-secret provenance and
   aggregates: source/workflow/run identities, release and readiness hashes,
   OCI manifest digest plus Docker config ID, evidence-root aggregate hash/file
   count/total size, benchmark counts/rates, the approved license basis, and
   hashes of the use purpose, any separate-license evidence, and the restricted-store
   locator. It contains no raw prompt, code, log,
   trace, credential, secret, or credential-bearing URI.
4. A fresh, short GitHub-hosted publish job receives only the sanitized
   envelope and uses least-privilege GitHub OIDC/attestation permissions to
   issue the keyless Sigstore attestation. Long-running qualification and
   attestation are separate jobs so the self-hosted runner never receives
   publication authority and an expired long-job token cannot weaken the
   provenance boundary.
5. A distinct protected ephemeral verifier mounts the sealed evidence store
   read-only, byte-regenerates the MatTools report without task execution,
   independently reruns readiness aggregation, rehashes every decisive role and
   envelope binding, and verifies the GitHub/Sigstore attestation against the
   exact trusted repository, signer workflow, source ref, source SHA, release
   digest, run ID, and run attempt. Only its final attestation-verification
   record may set `full_materials_production_ready=true`.
6. A second fresh GitHub-hosted job attests that sanitized final verdict. Neither
   hosted job can read the restricted raw closure, and no job deploys the release
   automatically.

Two prerequisites remain explicit blockers. The reviewed sandbox-isolation
trust anchor must exist as tracked file
`security/release-operator-public.pem`; it is currently absent, and no campaign
may execute generated candidate code until its signed isolation evidence
verifies under that key. The signing private key must never be committed or
placed in the benchmark job. Separately, the MatTools snapshot's allowed-use
and license basis require manual approval through the protected qualification
environment before the campaign starts; a schedule or release event alone is
not authorization.

## Why this structure

The local evidence began uneven and has since been hardened, but it is still not
an independent production qualification:

- The full code-execution image contains a useful pinned Python materials stack.
  The release gate has expanded from six probes to 13 executable invariants; the
  current host-pinned/WIP run passes 13/13 with zero skips. Every check emits structured observed,
  expected, tolerance, units/convention, version, and outcome evidence; missing
  evidence fails the gate. No current clean full-image 13-check promotion report
  exists, so the local/WIP result remains promotion-ineligible until rerun from
  clean Git against an immutable release image instance.
  A separate required CALPHAD experimental benchmark now runs in the same gate
  without changing the historical 13-check denominator: its assessment-basis
  calibration lane and independent post-assessment thermometric lane must both
  pass, and the exact benchmark JSON is retained and hashed.
- Production qualification binds and tests the resulting immutable Docker image
  ID; it is image-instance qualification, not yet a bit-reproducible build claim.
  The full Dockerfile now pins the multi-architecture Python base index by digest,
  but it still has unlocked apt packages and non-materials Python transitive
  dependencies, so rebuilding the same Git SHA can resolve different bytes.
  Complete OS/Python locks remain release-engineering work.
- A stored Ultra run produced a scientifically wrong IPF key by hand-rolling
  `R=x, G=y, B=z`; it marked [001] blue, contradicted its own prose, and still
  ended with run status `succeeded`. A corrected run used `orix` and a red-[001]
  assertion, but required 1,004.6 seconds, 3,038,408 tokens, and 38 execution
  calls, 16 of which exposed failure signals.
- A real 7.39 GB DREAM.3D file exposed an empty-StatsGenerator selection bug,
  invisible `Grain Data`, duplicate phase metadata, and missing charts. The
  corrected bounded reader now selects the complete, CellData-consistent 512^3
  SyntheticVolume, performs a complete chunked FeatureIds relationship scan,
  proves 56,978 declared tuples equal 56,978 referenced positive identities,
  reports 56,978 grains and stored phase `Primary`, and emits six maps plus two
  grain and two orientation charts in about 0.61 seconds. The source hash is
  `053e72e19b86757a1c7c4416c30512e699d6ea609f1bbab25fc9d75698efbc87`;
  this validates bounded viewer semantics, not segmentation truth.
- The image has no LAMMPS, Quantum ESPRESSO, CP2K, VASP, GPAW, xTB, phonopy, or
  MPI runtime. The production image now includes
  `pymatgen-analysis-defects==2025.1.18` for supported defect geometry, while the
  independent MatTools evaluator remains separately pinned to its historical
  `2024.7.19` environment.
- CALPHAD is no longer limited to a synthetic parser smoke. The release contains
  the NIST Al-Co-W reassessment associated with DOI
  `10.1016/j.calphad.2017.09.007`, whose repository item is CC0. The manifest
  separately binds the retrieved CRLF source hash and the stored LF-normalized
  hash, phase/element inventory, assessment scope, reference state, and caveats.
  The same required validator now reproduces two phase fields published with
  the assessment: Al-W at 1000 K and X(W)=0.20 gives single-phase Al4W
  (Fig. 8(b)); Al-Co-W at 1173 K and X(Al,Co,W)=(0.675,0.260,0.065) gives the
  AlCo-Al4W-Al5Co2 three-phase triangle (Fig. 12(a)). The v2 runtime additionally
  proves finite chemical potentials and per-vertex compositions, phase and
  composition closure, vertex-weighted bulk mass balance, and Gibbs-Euler
  consistency between molar Gibbs energy and chemical potentials.
  Copyrighted figure bytes are not vendored; their publisher URLs, byte sizes,
  and SHA-256 digests are recorded in the checkpoint evidence.
  User Thermo-Calc TDB and ChemSage DAT files remain tenant Resources;
  PostgreSQL supplies access control and the
  server-authored resource ID/SHA-256/size binding captured on each run, while
  the TDB or DAT bytes remain the authoritative Gibbs-energy model. `.db` is
  rejected rather than guessed from MIME type. A separate append-only
  revision/validation ledger records governance facts without relationalizing
  Gibbs functions. Its production claim remains blocked until the dedicated
  PostgreSQL qualification report proves the database triggers and constraints
  against clean release source. A fresh local WIP trace has passed the exact
  typed-CLI, Go HTTP, role-separated PostgreSQL, retained-byte, and lineage
  checks; because the worktree is dirty, it is diagnostic evidence only and
  does not satisfy the promotion report requirement.

  A same-prompt live trace then exposed two solver-selection hazards that the
  original closure checks did not catch. Selecting `AL` and `CO` as independent
  coordinates made pycalphad converge to `LIQUID + AL5CO2` at
  `-85512.6057 J/mol`, about `457.46 J/mol` above the published three-phase
  solution. Canonicalizing the alphabetically first physical component (`AL`)
  as dependent removed that parameterization sensitivity, but the live agent
  still omitted `VA`; that omission removed the substitutional phase models
  and reproduced the same higher-Gibbs basin. The typed host now authenticates
  the exact retained inspection artifact before request hashing, automatically
  retains `VA` only when that inventory declares it, and the sandbox rejects a
  direct typed omission. Coupled Cartesian grids that cannot be reframed
  without changing points fail closed.

  The first scientifically corrected live run
  `run_65ef275f67bbe597a2d53f809ad32b90` on immutable local
  code-execution image
  `sha256:db9d72ba1af59c942b0e513afdfc3f6620af1515886be17cf7e4acdc484305e2`
  used components `AL,CO,VA,W`, canonical independent axes `CO=0.260` and
  `W=0.065`, and all 18 phases. It recovered `AL4W`, `AL5CO2`, and `BCC_B2`
  with fractions `0.3249914280`, `0.3487946716`, and `0.3262139004`, molar
  Gibbs energy `-85970.0674612 J/mol`, maximum bulk residual
  `5.32e-12`, and Gibbs-Euler residual `5.92e-7 J/mol`. The retained inspection
  and equilibrium hashes are `35cc76cdf8ef3174bdaa0e5429ace9a1aba4bbab1fbf8575953207b6ae3f6f74`
  and `b49d8b7b42a8fb91cc0c7e188bffb564d868ca04af7093837af8ce2a96da7b46`.
  No BisQue mutation tool ran.

  That trace established typed-equilibrium correctness but was not
  promotion-grade orchestration evidence. It consumed 1,310,099
  input and 21,189 output tokens across 29 model/tool rounds in 118 seconds,
  including eight generic execution calls to discover the validation API. Its
  hand-built `materials_validation.json` recomputes to `verified`, but the
  canonical parser correctly rejects the serialized file because its
  `required_validator_ids` are not canonically ordered.

  After making prompt-inferred materials routing mandatory and placing the exact
  `assess_scientific_status` -> `canonical_record_json` ->
  `parse_assessment_record` contract in both the runtime guidance and skill, the
  same browser prompt passed end to end as
  `run_08cc83bc063527d5e95873383acb0e6d`. The 58-second run read the materials
  skill first, authenticated inspection hash
  `35cc76cdf8ef3174bdaa0e5429ace9a1aba4bbab1fbf8575953207b6ae3f6f74`, and
  retained equilibrium artifact
  `6601f2c661fa8db6e58880ce969d165393c906b19939d7033914c85b3f2f2b2d`
  with canonical evidence hash
  `4eeb95c9c30a4848794d108c05b613ebeb86dca0d6548a43671c20bec3eca16b`.
  It recovered fractions `0.3249914280` Al4W, `0.3487946716` Al5Co2, and
  `0.3262139004` BCC_B2, `GM=-85970.0674612 J/mol`, maximum bulk residual
  `5.73e-12`, and Gibbs-Euler residual `6.05e-7 J/mol` using `AL,CO,VA,W`,
  canonical `CO/W` axes, and all 18 database phases. The backend's exact
  `parse_assessment_record` independently accepts the retained
  `materials_validation.json` as `verified`: all eight required validators pass
  with no missing validators, critical failures, or contradictions. The
  pretty-printed retained file hashes to
  `389abc5a7a76dd1cd2ffb4943d60c4a52da045d0b023f1e9bc463a0cf16360cb`;
  its deterministic canonical record hashes to
  `655c4537a8d6f2651983176cc42aeaa7a21d3588ec15b7a7d10681f04bcaad33`.
  No BisQue or remote-mutation tool ran.

  This is strong diagnostic evidence, not a release attestation. It used 11
  model calls, 13 completed tool calls, 418,681 input and 11,902 output tokens;
  two of four generic execution calls failed while the agent corrected a local
  parser-use mistake and an output-path assumption. The durable verdict is now
  correct, but orchestration cost and exploratory-shell reliability remain
  optimization targets, and the dirty worktree still prevents this run from
  satisfying the protected promotion contract.

  The two figure checkpoints still compare pycalphad with calculations from the
  same assessment and therefore remain cross-engine regressions. A new
  provenance-bound two-lane benchmark adds experimental evidence without
  blurring that distinction. Its calibration lane uses the NIST CC0 900 degrees C
  dataset associated with DOI `10.1007/s11669-014-0346-2`: six scalar phase-
  composition coordinates carry published 95% confidence intervals, but the
  data contributed to the 2017 assessment and are not held out. The pinned TDB
  gives weighted RMS z `0.48684` and maximum z `0.78468`, below locked limits
  `1.0` and `2.0`.

  The independent lane uses two CC-BY-4.0 primary DTA studies published after
  the assessment: DOI `10.1007/s10973-018-7431-4` at its measured
  X(Al,Co,W)=(`0.091`,`0.817`,`0.092`) and DOI
  `10.1007/s10973-020-10279-9` at nominal (`0.09`,`0.82`,`0.09`). The four
  reported transition observations are 1447/1487 degrees C and 1444/1468
  degrees C (1720.15/1760.15 K and 1717.15/1741.15 K). Pinned pycalphad predicts
  1458.378/1487.466 degrees C and 1461.078/1488.418 degrees C, giving MAE
  `12.3348 K` and maximum absolute error `20.4176 K`. This passes the fixed,
  pre-execution engineering limits of `20 K` MAE and `30 K` maximum error.
  Neither held-out article reports numerical measurement uncertainty, so this
  is explicitly an independent engineering validation, not a metrology-grade
  uncertainty-normalized benchmark. Missing evidence, threshold relaxation, or
  a future residual above either locked limit blocks production promotion.

These findings support focused skills plus executable verification. They do not
support either a broad “materials expert” label or a new autonomous materials
orchestrator.

## External benchmark evidence

The primary promotion benchmark is MatTools, specifically its real-world
tool-usage benchmark. The paper defines 49 parent functions and 138 scientific
tasks, evaluated by safe code execution and verifier scripts. It reports
GPT-4o alone at 45.58% runnable and 18.36% task success, and GPT-4o with the
best single documentation-RAG configuration at 67.35% runnable and 39.61% task
success. Its best three-trial documentation-plus-reflection result was 125/147
runnable attempts (85.03%) and 229/414 passing scientific subtasks (55.31%).
Ultra's gate therefore permits seven fewer runnable parents than that published
best while requiring 20 more correct scientific subtasks: at least 118/147 and
249/414. The paper also reports that the simpler reflection design outperformed
more elaborate agentic RAG and LightRAG configurations. Those results argue
against adding an agent hierarchy without evidence.

Primary sources:

- MatTools paper: <https://arxiv.org/html/2505.10852>
- MatTools dataset card and license: <https://huggingface.co/datasets/SiyuLiu/MatTools>
- Matbench repository: <https://github.com/materialsproject/matbench>
- Matbench paper: <https://doi.org/10.1038/s41524-020-00406-3>

MatTools measures tool use and executable scientific correctness. Matbench is
reserved for a later property-prediction track; it must not be substituted for
MatTools in this promotion decision. Multimodal materials benchmarks are also
reported separately and do not dilute the executable gate.

## Promotion metric contract

The release candidate must run the official MatTools real-world corpus, not a
locally rewritten or model-generated facsimile. The runner must lock and record:

- source URL, immutable revision or content-addressed snapshot identifier;
- license and allowed-use declaration;
- SHA-256 for the task manifest, every problem/verification file, and the
  resolved environment lock;
- Ultra commit, dirty-worktree state, skill hashes, model/provider identifier,
  model routing settings, image digest, dependency versions, and run budgets;
- task order, seed where supported, timestamps, Ultra run IDs, artifact IDs,
  exit status, timeout/OOM state, and verifier output.

For comparable official scoring, the evaluator is locked to MatTools upstream
commit `1803a6abfe23a9da56c894076c59117873b758ff`, pymatgen `2024.8.9`, and
pymatgen-analysis-defects `2024.7.19`. These historical evaluator dependencies
are distinct from Ultra's production sandbox dependencies and must be reported
separately. Evaluating the corpus with production library versions is a useful
compatibility experiment, but it is a benchmark variant and cannot satisfy this
promotion gate.

The benchmark prompts and verifier answers must not be inserted into production
skills, examples, retrieval indexes, memory, or fine-tuning data. The harness
may expose only the problem statement to Ultra. Verification code and expected
values remain isolated until the submitted Ultra run is complete.

### Unit of evaluation

Match MatTools' published denominators exactly:

- **Function Runnable Rate (FRR):** runnable parent functions / 49.
- **Task Success Rate (TSR):** verifier-passing scientific subtasks / 138.

For a release decision, run three complete independent trials, matching the
paper's reporting protocol. Aggregate counts are the primary score:

```text
FRR = sum(runnable_parent_functions) / (49 * 3)
TSR = sum(passing_scientific_subtasks) / (138 * 3)
```

Therefore an aggregate pass requires at least 118 of 147 runnable parent
attempts and 249 of 414 successful strict scientific subtask attempts. Each
trial must independently reach at least 40 of 49 runnable parents and 83 of 138
strict scientific subtasks. The official-upstream scientific total is retained
as a separate score and must independently reach 249 of 414. A checkpoint resume
continues an interrupted attempt; it does not retry, replace, or erase a
scientifically failed attempt.
Infrastructure-invalid attempts are retained in the audit log and rerun only
after the report records the invalidation reason.

### Runnable

A parent function is runnable only when the hashed strict shadow independently
observes candidate execution returning a parseable, non-empty dictionary under
the pinned environment and budget. The unmodified upstream runner is still run
and its historical runnable summary is reported separately, because its final
accounting incorrectly converts some verifier-only failures into function
errors. Ultra must submit its answer through the ordinary control-plane/Deep
Agents/code-execution path. Direct model calls, direct invocation of a generated
script outside Ultra, hand-edited answers, and benchmark-specific answer
injection are invalid.

Timeout, OOM, missing dependency, malformed output, sandbox policy failure,
uncaught exception, or a run that claims success without the required material
property output is not runnable. Dependency gaps are product failures for this
gate, not exclusions from the denominator.

### Scientifically correct

The decisive strict classification starts from the pinned upstream MatTools
verification output and applies only reviewed, hash-bound semantic repairs for
documented upstream evaluator defects. A repair may raise or lower a
classification; ad hoc release-specific edits are forbidden. The unmodified
upstream classification remains retained and separately gated, so repairs cannot
hide a weak official score. A run status of `succeeded`, a generated plot, a
plausible narrative, or successful Python exit is not scientific correctness.
Unknown, ungraded, skipped, or verifier-crashed subtasks count as failures unless
the entire attempt is formally invalidated as benchmark-infrastructure failure.

### Hard promotion gates

All of the following are required:

| Gate | Requirement |
| --- | --- |
| MatTools FRR | `>= 118/147` aggregate and `>= 40/49` in every full trial |
| MatTools strict TSR | `>= 249/414` aggregate and `>= 83/138` in every full trial |
| MatTools official-upstream TSR | independently `>= 249/414` aggregate |
| Trial completeness | 49/49 parents and 138/138 subtasks attempted per valid trial |
| Deterministic domain suite | exact required 13/13 JUnit checks passed, with zero failures, errors, or skips |
| Critical invariants | 100% pass for unit, symmetry, convention, sentinel, and phase-identity checks |
| CALPHAD experimental evidence | calibration RMS/max z `<= 1/2`; independent four-point thermometric MAE/max `<= 20/30 K`; thresholds fixed before execution |
| Silent-success rate | exactly zero attempts may be reported successful by Ultra while a critical verifier fails |
| Reproducibility | identical pinned inputs yield the same verifier classification on immediate replay |
| Provenance | all required hashes, versions, run/artifact IDs, and raw verifier records present |
| Security | benchmark answers isolated; sandbox has no unauthorized network or host access |
| Operator trust anchor | tracked, reviewed `security/release-operator-public.pem` verifies signed isolation evidence before candidate execution |
| License authorization | protected-environment manual approval records the approved MatTools snapshot and allowed-use basis before execution |
| Evidence custody | exact raw closure rehashes from restricted encrypted/WORM storage; no raw campaign material is a public Actions artifact |
| Public attestation | sanitized envelope has a valid GitHub/Sigstore attestation bound to the exact repository, signer workflow, ref, source SHA, run, and release digest |
| Runner provenance | qualification runs on the dedicated ephemeral Linux ARM64 lane; publication runs separately on a fresh GitHub-hosted least-privilege job |

Any missing denominator, skipped required check, unverifiable snapshot, edited
upstream verifier, or selectively omitted failure blocks promotion. A high
average cannot compensate for a failed critical invariant. Passing this table
makes the readiness JSON a candidate for attestation; it does not by itself make
the release distribution-ready.

## Deterministic domain suite

The fast suite exists to catch convention and integration failures that a broad
benchmark average can hide. It must execute inside the release sandbox image.
At minimum it covers:

1. ordered L1_2 Ni3Al and FCC Ni/Cu symmetry across a declared `symprec` range;
2. FCC Ni Cu-Kalpha XRD peak identity/order and tolerance-bounded 2-theta;
3. finite, stable composition features with schema and units recorded;
4. anisotropic-voxel volume/length conversion with a known analytic object;
5. EBSD cubic IPF convention, including [001] red for the declared TSL key;
6. Mackenzie cubic misorientation distribution invariants;
7. DREAM.3D complete-geometry selection, feature-zero sentinel handling,
   grain count, phase identity, and bounded reads;
8. PoreSpy solid/void convention and local-thickness behavior;
9. ASE equation-of-state smoke with a physically plausible equilibrium region;
10. NaCl point-defect generator stoichiometry, nearest-neighbor coordination,
    and charge-neutral vacancy/interstitial accounting;
11. CALPHAD assessed-database provenance, manifest/hash/phase validation,
    bounded equilibrium axes/units, finite GM/MU/X/NP values, phase and bulk
    composition closure, the published Al4W binary field and
    AlCo-Al4W-Al5Co2 ternary three-phase field, plus explicit rejection of
    package test databases and renamed fixture copies as production evidence.
12. A separate CALPHAD experimental gate with (a) uncertainty-normalized NIST
    900 degrees C phase-vertex calibration labeled as assessment-basis data and
    (b) four post-assessment DTA solidus/liquidus observations labeled as an
    independent engineering holdout, including exact Celsius-to-kelvin
    conversions, composition provenance, source licenses, residuals, and locked
    promotion thresholds.

Each check returns a machine-readable result with observed value, expected
value/range, tolerance rationale, units, library version, and pass/fail. Tests
must fail closed when an optional scientific dependency is missing; they may
not silently skip in the promotion environment.

## CALPHAD database acceptance contract

One server-authored run binding—resource ID, content SHA-256, byte size, and
explicit `tdb` or `dat` format—represents one CALPHAD database byte revision.
The generic Resource row owns user/org/project ACLs and the
current catalog identity, but it is mutable by design and must not be presented as
an append-only scientific revision ledger. The worker snapshots the binding onto the
run and rehashes staged bytes before every inspection/solve; a later catalog or file
change therefore fails rather than silently changing an existing run. PostgreSQL must
not duplicate, reinterpret, or merge Gibbs-energy functions.

Modified TDB or ChemSage DAT bytes must be uploaded under a new resource ID and
revalidated. `.db` is not an accepted alias, even with a TDB-looking MIME type.
The
server-managed ledger snapshots resource ownership, byte hash/size, parent lineage,
database format, assessment-pressure limits, runtime image, pycalphad version,
run authority, and content-addressed evidence. Owner-writable Resource metadata
cannot change those records. Every non-registration result—including bounded
`failed`, `timeout`, and `unsupported` outcomes—must carry exact retained evidence
bytes. The server
rehashes and parses it, verifies its database binding against the live resource and
immutable revision, and requires the active worker lease before appending an
idempotent event. PostgreSQL rejects UPDATE, DELETE, and TRUNCATE of ledger records.
The ledger deliberately has no lifecycle foreign key to the mutable Resource table so
audit history survives resource garbage collection.

The serving role has SELECT on the five CALPHAD ledger tables but no raw INSERT,
UPDATE, DELETE, or TRUNCATE capability. It can execute exactly two versioned
`SECURITY DEFINER` writers—revision creation and validation append—whose fixed
`pg_catalog` search path rechecks tenant/resource bindings, retained bytes,
runtime policy, run/lease authority, and lineage transactionally. Per normalized
user/organization tenant, the default capacity is 4 GiB of logical retained
input-plus-evidence references and 100,000 validation events. Exact retries are
zero-charge; distinct references to identical blobs are deliberately charged
per reference, and quota rejection rolls back the complete transaction.
This is a least-privilege ledger-integrity boundary, not a complete defense
against a compromised serving credential. The shared serving role currently has
raw SELECT on all five CALPHAD tables, including retained proprietary input and
evidence bytes; tenant filtering is enforced by parameterized Go read paths,
not PostgreSQL row-level security. Safe database-enforced confidentiality needs
transaction-scoped trusted tenant identity (or an execute-only scoped reader
surface) before raw SELECT can be revoked. Adding nominal RLS without a trusted
principal would create false isolation because the shared role could choose its
own tenant setting. The authenticated HTTP verifier therefore remains part of
the trusted computing base, and this gap blocks a production-complete score.

Idempotency is keyed by revision, run, operation, and the exact retained evidence SHA-256—not by
operation or request alone—so one run can append multiple grids and multiple valid observations of
the same request while an exact callback retry converges on the existing record. The separately
stored server-recomputed request SHA-256 identifies the scientific request without assuming solver
replay determinism. Inspection and equilibrium events also carry a server-recomputed
selection-independent database-inventory SHA-256. An equilibrium may reference
an inspection only when that inventory fingerprint, revision, run, image, and retained artifact
all match; changing provenance, phase models, references, available inventory, or assessment limits
breaks the lineage even when the TDB byte hash is copied into a newly sealed-looking artifact.

Inspection events retain the exact server-rehashed JSON bytes and must match the full
pycalphad inventory/phase-model/reference manifest plus its canonical hash. Equilibrium
events must name a prior retained inspection from the same revision, run, and immutable
runtime image; the control plane recomputes grid coverage, composition closure, phase
and vertex fractions, bulk reconstruction residuals, Gibbs-Euler consistency, units,
and the canonical v2 evidence hash before insertion. The run's accepted image ID and
`pycalphad==0.11.2` are stamped by control-plane configuration rather than accepted from
worker claims. The v2 runtime policy also fixes network `none`, no-new-privileges, read-only root
filesystem, complete capability drop, no GPU, and maximum CPU, memory, PID, and outer wall-time
bounds for the typed primitive. Historical events without the exact retained-evidence contract are exposed as
`legacy_unretained` and non-promotable. These are technical governance guarantees, not
independent validation of the assessment's scientific domain.

Owner-selected resources may append technically promotable validation events. A resource made
available only through a read/public grant remains usable for analysis, but returns a
`read_only_unpromoted` content-addressed artifact and never writes into the owner's ledger. The
control plane stamps that distinction into the selected-resource capability before dispatch.

Production promotion additionally requires the dedicated live PostgreSQL report to
prove the exact columns, constraints, indexes, functions, triggers, payload SHA-256
binding, and a least-privilege serving role distinct from the schema owner. Static SQL
inspection and the in-memory store cannot satisfy that operational gate.

It also requires a content-addressed cross-language report generated from the exact
production runtime image: real pycalphad 0.11.2 typed-CLI inspection and equilibrium bytes must
pass through the real Go HTTP callback into the dedicated PostgreSQL database and be read back
byte-for-byte with matching request, inventory, and inspection lineage. Branch CI exercises the
fail-closed cross-language harness and parser contracts, but it does not emit live qualification
evidence from the lean deterministic materials image. The live target is deliberately post-image:
only the exact deployed production worker runtime can satisfy the readiness aggregate.
Final readiness also re-posts the retained content-addressed database input and
both typed output artifacts through the real Go HTTP scientific verifier on an
in-memory store; that independent semantic replay complements, but never replaces,
the original role-separated live-PostgreSQL callback and retained-byte marker.

The production sandbox exposes the embedded open reference under
`$ULTRA_CALPHAD_DATABASE_ROOT` and verifies its manifest, stored hash/size,
physical elements, vacancy component, pseudo-elements, and phase inventory at
image build and runtime resolution. User databases require source/citation,
license/use authorization, assessment scope and limits, reference-state
convention, expected catalog hash/size, and an immutable resource/artifact ID.
Package tests, examples, copied fixture hashes, symlinks, non-regular files, and
oversized or structurally excessive databases fail closed.

Every calculation requires finite kelvin, pascal, mole, and independent-
composition inputs; explicit component and phase subsets; composition closure;
bounded database/parameter/phase/grid/result sizes; and an operator wall-clock
limit. Outputs record stable vertices, per-vertex phase compositions, phase-amount
fractions, chemical potentials, phase/bulk-composition closure, Gibbs-Euler
consistency, molar Gibbs energy, warnings,
database revision, and pycalphad version. The surrounding
immutable sandbox execution trace—not the numerical JSON alone—attests the
runtime image ID and source revision.
Timeouts, solver failures, empty/non-finite vertices, and closure failures cannot
be presented as successful diagrams. Restricted-phase/metastable calculations
must list suspended phases and must never be called global equilibrium.

Before proprietary TDBs are enabled, selected-upload staging, run-scoped resource
search/resolve, public-share parity, and model-visible metadata projection must be
tenant-safe. Arbitrary license text, credentials, vendor account data, or contract
terms are never projected to the model.

## Runtime behavior contract

### Routing

- `selection_context.suggested_domain = "materials"` is consumed by the Deep
  Agents run context and is auditable routing evidence.
- Canonical requests such as “identify the space group,” “calculate an XRD
  pattern,” and “build a CALPHAD phase diagram” trigger the corresponding
  focused skill and the normal scoped-delegation decision.
- Deterministic crystallography, diffraction, and equilibrium calculations do
  not inherit dynamics-only requirements such as random seeds, observation
  durations, initial-condition sweeps, or integration time steps.

### Verification

For supported materials tasks the coordinator must:

1. state conventions, units, assumptions, and data provenance;
2. prefer a pinned library implementation over hand-rolled scientific formulas;
3. execute a task-specific deterministic validator before presenting a result;
4. distinguish computed, measured, database-retrieved, and illustrative values;
5. preserve code, environment, input, output, and validator artifacts; and
6. refuse or clearly scope unsupported heavy-simulation claims.

The optional `materials-verifier` receives only the completed candidate result,
inputs, conventions, and validation contract. It is read-only: it cannot edit
the candidate artifact, choose a new scientific question, or conceal failures.
Its verdict is advisory until a deterministic verifier confirms it.

### Success semantics and traceability

Run status and scientific status are separate fields. A materials result cannot
be presented as verified solely because the orchestration run completed.
Tracing must record:

- which materials skill and recipe were read;
- which library/API implementation was selected;
- input and output artifact hashes;
- validator identifiers and results;
- scientific status (`verified`, `failed`, `unsupported`, or `unverified`);
- contradiction checks between prose, tables, and artifacts; and
- resource/cost metrics, retry count, and visible execution failures.

The canonical validation record is exactly `outputs/materials_validation.json`
in the durable artifact namespace and is bounded to 1,000,000 bytes. Trace
inspection recomputes its decision fields, hashes the record, resolves every
evidence reference to a same-run durable artifact, and compares the declared
SHA-256 (and size when supplied) with control-plane artifact metadata. A record
in a workspace/arbitrary path or a fabricated/unresolvable evidence digest does
not satisfy trace quality. This record remains first-party evidence; MatTools is
the independent scientific verifier for promotion.

The control plane may expose a collected `/outputs/name` artifact as basename
`name`. Trace normalization accepts that form only when the same-run artifact
has an ID and `tool_name=outputs_collector`; an unprovenanced basename remains
invalid. Remote BisQue upload/dataset tools are outside the durable-output
contract and require explicit current-turn user intent. A live XRD trace exposed
that the Python tool guard can stop an unrequested remote mutation, but the Go
`/v2/bisque` mutation boundary still lacks an immutable intent capability plus a
transactional running-run/lease check. That control-plane gap is a production
blocker even when the scientific result itself is correct.

## DREAM.3D/HDF5 acceptance contract

The dashboard and backend must select a complete image geometry rather than the
first `_SIMPL_GEOMETRY` encountered. A candidate geometry is complete only when
dimensions, spacing, origin, and the associated cell/feature data are mutually
consistent. Stats-generator placeholders are not preferred over a populated
synthetic volume.

Feature groups include established DREAM.3D spellings such as `CellFeatureData`,
`Feature Data`, and `Grain Data`. Index zero is treated as a sentinel only where
the file schema/array semantics establish it; only that row is excluded. Zero
measurements in real grains are not removed by value. Grain counts, phase names,
phase IDs, crystal structures, and ensemble metadata are deduplicated by stable
identity rather than display string alone. Multi-gigabyte files are probed with
metadata and bounded/chunked reads, never whole-dataset materialization.

The real 7.39 GB local file is a required non-CI qualification fixture, while a
small synthetic file with the same naming and sentinel semantics is the PR test.

## Sandbox and release contract

The repository deployment wrappers consume an already extracted
`/srv/ultra/releases/<sha>` tree; they do not accept a tarball or checksum-file
argument. The provisioning/extraction boundary must therefore verify the
published `ultra-release-<sha>.tar.gz.sha256` before invoking a wrapper. Once
extracted, production sandbox parity rehashes the manifest-bound control binary
and complete frontend distribution, and the control wrapper never rebuilds or
replaces a binary from an immutable release manifest.

`production-full` qualification must run from that extracted release root, not
from a mutable Git worktree. Its schema-v1 evidence bundle retains every byte by
a report-relative, content-addressed path: `release-manifest.json`, the control
binary, the complete frontend file closure, the exact staged-source closure
(including generated import shims and the CALPHAD probe), raw Docker inspect
records and OCI labels, the resolved package inventory and `pip freeze`, domain
and CALPHAD/JUnit outputs, and execution logs. Tree aggregates are SHA-256 hashes
of canonical sorted `{path, sha256, size_bytes}` JSON records and reject added,
removed, symlinked, or changed files when rehashed. The readiness aggregator must
rehash these retained bytes independently; hashes without retained bytes are not
promotion evidence.

That complete evidence bundle is the restricted evidence closure, not a public
workflow artifact. It is retained in encrypted, access-controlled WORM storage
alongside the raw MatTools campaign. GitHub Actions and public releases may
carry only the sanitized hash/count envelope and its Sigstore attestation. A
public envelope hash is meaningful only while the final verifier can retrieve
and exact-rehash the corresponding restricted closure under the approved
evidence-root manifest.

Every copied sandbox source file is re-bound to the retained release manifest
after copying, so a verify-then-mutate race fails before execution. The staged
allowlist also binds the generated probe, empty import shims, and the two
DockerSandboxBackend-generated matplotlib configurations. After execution the
whole file closure is compared again; only new files under the declared
`.cache` and `.tmp` scratch roots are excluded, while modified baseline files,
new source files, deletions, and symlinks fail the gate.

Host CALPHAD orchestration never executes pytest from the extracted release
tree. It runs the exact release-manifest subset from a retained source snapshot
with `--no-sync`, `/dev/null` pytest configuration, no conftest loading, cleared
`PYTEST_ADDOPTS`/`PYTEST_PLUGINS`, and plugin autoload disabled. The sandbox
runtime suite uses the same pytest isolation. Production Docker inspect must
bind `PYTHONPATH=/opt/ultra-runtime`, and the CALPHAD probe must prove that the
imported materials package came from `/opt/ultra-runtime`, not `/workspace`.

The seven-file domain evidence closure (JSON, Markdown, JUnit, pip freeze,
stdout, stderr, and the CALPHAD experimental benchmark JSON) is exact. JUnit scientific records, direct pins, complete installed
package inventory, canonical `pip freeze`, staged requirements/test hashes, and
the staged CALPHAD registry/TDB/module hashes are independently cross-checked;
coherently replacing both a report and its content-addressed companion is not
sufficient to pass.

`DockerSandboxBackend` currently returns one bounded output field after combining
the child process stdout and stderr. The verifier therefore records capture mode
`docker_sandbox_backend_combined_stdout_stderr` and retains those exact combined
bytes; it must not pretend separate streams exist. The host orchestration suite,
which uses a subprocess API with separate streams, retains stdout and stderr as
separate content-addressed files.

- Pin direct materials dependencies to exact versions in a reviewed lock or
  constraints file and record the resolved transitive environment.
- Build and publish by immutable image digest. Mutable tags are aliases, not
  promotion evidence.
- Generate the advertised compute-resource manifest from the built image and
  test that it includes the supported materials libraries and versions.
- Keep production `pymatgen-analysis-defects` pinned and verified in the release
  sandbox. Build the separate evaluator with the upstream MatTools historical
  dependency versions before executing the promotion corpus; never substitute
  production versions for the official scoring environment.
- Do not claim LAMMPS, DFT, phonons, or MPI support until binaries, pseudopotential
  provenance, resource limits, and domain benchmarks are present.
- Test fixtures shipped inside third-party wheels (including pycalphad test
  databases) are prohibited as evidence for a user-facing scientific result.

## CI and qualification tiers

### Pull requests

- lean unit tests for routing, contracts, score arithmetic, result parsing,
  checkpoint/resume, provenance validation, and synthetic DREAM.3D semantics;
- scientific tests may use controlled stubs only to test orchestration, never
  to claim physical correctness.

### Nightly and release-candidate sandbox gate

- build the full code-execution image, resolve its immutable image ID, and treat
  that exact image instance—not the mutable tag—as the qualification subject;
- run all deterministic materials checks inside that image with zero skips;
- require the exact reviewed 36 CALPHAD runtime test identities plus two real
  pycalphad CLI test identities (38 image tests total), and the separate 48-test
  host orchestration contract, rather than accepting only aggregate counts;
- verify the image manifest against installed packages and image digest;
- store machine-readable JSON, JUnit, logs, and a human-readable Markdown
  summary in the restricted immutable evidence closure; publish only the
  sanitized envelope described above.

### Protected MatTools qualification gate

- protected-environment manual approval of the allowed-use/license basis and an
  explicitly supplied, immutable MatTools snapshot before any candidate code
  executes;
- dedicated ephemeral Linux ARM64 qualification runner with no OIDC or
  repository-write permission;
- fail closed before execution unless tracked
  `security/release-operator-public.pem` verifies the external sandbox-isolation
  attestation;
- three full trials through the actual Ultra runtime;
- checkpoint/resume and bounded concurrency without selective retry;
- raw per-attempt records plus aggregate JSON/Markdown report retained only in
  restricted encrypted/WORM storage;
- fail the protected promotion job below 0.80 FRR or 0.60 TSR.

The current campaign is unrun: there is no complete three-trial MatTools result
and no measured FRR or strict-scientific TSR for this release candidate.

### Sanitized attestation and final verification

- emit an evidence-qualified readiness candidate and a sanitized public
  envelope only after every protected qualification gate passes;
- transfer only that sanitized envelope to a fresh GitHub-hosted publish job;
- give the publish job only `contents: read`, `id-token: write`,
  `attestations: write`, and `artifact-metadata: write`, with all other GitHub
  permissions absent;
- verify the attestation under the exact repository, signer workflow, source
  ref/SHA, release digest, run ID, and run attempt policy, and reject an
  attestation issued from a self-hosted publish runner;
- exact-rehash the restricted evidence closure and its decisive roles before
  emitting the final production-verification record;
- never infer production readiness from an unsigned/self-consistent readiness
  JSON, a mutable Actions artifact, or a hash whose retained bytes are absent.

### CALPHAD PostgreSQL ledger gate

- run only on a dedicated database whose name explicitly identifies test, CI,
  sandbox, or qualification use; never point the gate at production;
- execute the exact non-skipped live PostgreSQL trigger/FK/content-binding test
  plus HTTP authority and schema-contract tests;
- hash the complete control-plane/worker source closure and emit a
  content-addressed `calphad-ledger-postgres-qualification` report;
- require that report, the verified release control-binary identity, and the
  same clean Git SHA in the final materials-readiness decision.

### Quarterly extended qualification

- real multi-gigabyte DREAM.3D fixture;
- multimodal materials questions and figure interpretation;
- Matbench/property-prediction track where enabled;
- HPC/solver benchmarks only after those capabilities ship.

## Required report schema

Each promotion report contains:

```json
{
  "schema_version": "1",
  "benchmark": {"name": "MatTools-real-world", "revision": "...", "sha256": "..."},
  "ultra": {"commit": "...", "dirty": false, "image_digest": "sha256:..."},
  "trials": [],
  "counts": {
    "runnable": 0,
    "runnable_denominator": 147,
    "scientific_pass": 0,
    "strict_scientific_pass": 0,
    "scientific_denominator": 414
  },
  "rates": {
    "function_runnable": 0.0,
    "official_task_success": 0.0,
    "strict_task_success": 0.0
  },
  "per_trial_minima": {"runnable": 40, "strict_scientific_pass": 83},
  "deterministic_suite": {"passed": 0, "total": 13, "skipped": 0},
  "hard_gates": {},
  "promotion": {
    "status": "not_qualified",
    "evidence_passed": false,
    "attestation_required": true,
    "distribution_ready": false,
    "full_materials_production_ready": false,
    "reasons": []
  }
}
```

The schema is illustrative; the implementation may add fields but cannot
remove denominators, hashes, raw attempt linkage, or individual gate verdicts.
When every evidence gate passes, `status` becomes
`candidate_for_attestation`; the other three readiness booleans remain false
until the separate final verifier validates both the GitHub/Sigstore-attested
sanitized envelope and the exact restricted evidence closure. The final
verification record must be a distinct schema and is the only record permitted
to assert `full_materials_production_ready=true`.

## Implementation sequence

1. Repair DREAM.3D semantics and add the real-file qualification probe.
2. Split/narrow skills, consume materials selection context, and add
   materials-aware Pro contracts.
3. Add deterministic validators and correctness-aware trace fields.
4. Make the full-image domain suite non-skipping and pin the sandbox release.
5. Integrate the licensed, immutable MatTools corpus and required dependencies.
6. Run one diagnostic trial, classify failures, and fix general product
   behavior without copying benchmark answers into skills.
7. After protected manual license approval and isolation-attestation preflight,
   run three clean release trials on the dedicated ephemeral Linux ARM64 lane.
8. Retain the complete raw closure in restricted encrypted/WORM storage and
   emit only an evidence-qualified readiness candidate plus sanitized envelope.
9. Attest the sanitized envelope in a fresh, least-privilege GitHub-hosted job.
10. Promote only after the final verifier validates the GitHub/Sigstore
    attestation and exact-rehashes the complete restricted evidence closure.

## Completion checklist

The materials-readiness goal remains open until evidence shows all boxes checked:

- [ ] focused materials skill structure is shipped and routed;
- [ ] deterministic materials contracts and validators are traceable;
- [ ] synthetic and real DREAM.3D acceptance tests pass;
- [ ] release sandbox is pinned, discoverable, and non-skipping;
- [ ] dedicated-PostgreSQL CALPHAD ledger qualification passes with zero skips;
- [ ] reviewed `security/release-operator-public.pem` is tracked and verifies
      signed sandbox-isolation evidence before generated code executes;
- [ ] MatTools snapshot/provenance are locked and the allowed-use/license basis
      has protected-environment manual approval;
- [ ] all 49 parent functions and 138 subtasks run in each of three trials via
      Ultra on the dedicated ephemeral Linux ARM64 qualification runner;
- [ ] aggregate FRR is at least 0.80 (at least 118/147 runnable parents);
- [ ] every trial reaches at least 40/49 runnable parents and 83/138 strict
      scientific subtasks;
- [ ] aggregate strict-scientific TSR is at least 0.60 (at least 249/414
      verifier-passing subtasks), and the separate official-upstream scientific
      total independently reaches at least 249/414;
- [ ] critical and silent-success gates pass;
- [ ] complete raw MatTools prompts/code/logs/workbooks/artifacts/traces and all
      decisive reports are exact-bound in restricted encrypted/WORM storage,
      never a public GitHub Actions artifact;
- [ ] immutable JSON and Markdown candidate reports link every result to Ultra
      traces and artifacts without asserting final production readiness;
- [ ] sanitized public envelope is attested by GitHub/Sigstore in a fresh
      least-privilege GitHub-hosted publish job;
- [ ] final verifier validates the exact repository/workflow/ref/SHA/run policy
      and exact-rehashes the restricted evidence closure before setting
      `full_materials_production_ready=true`.

The MatTools benchmark boxes are currently unchecked: the required three-trial
campaign is unrun and the production qualification is incomplete. Until the
entire checklist passes, the accurate product label is **materials science
research preview**, with stronger support for microstructure/EBSD than for
general computational materials science.
