# Apple Metal Current State

Snapshot date: 2026-09-01

This is the current state of the Apple Silicon/Metal work across the Hesiod
application and its HighMap backend. The snapshot is on the personal GitHub
repositories, not the upstream `ottolink-dev` repositories.

## Repository state

| Repository | Personal remote | Branch | Implementation HEAD at snapshot | Upstream base |
|---|---|---|---|---|
| Hesiod | `Leonhardmaster2/Hesiod` | `feature/apple-metal-integration` | see `git rev-parse HEAD` at snapshot | `324c3ed51202ec6aa43d89730da39c913060d392` |
| HighMap | `Leonhardmaster2/HighMap` | `feature/apple-metal-backend` | `a0c416dc` + `test_metal_phase8` | `e0d1279fced75cf7556de233128abfb5650e25b5` |

Both feature histories are linear from their current upstream `dev` bases and
contain zero merge commits in the feature-only range. The Hesiod submodule is
pinned to HighMap `a0c416dc` (+ the Phase 8 test file in the pinned history's
working tree). Existing local build directories, generated images, logs, and
other untracked artifacts were preserved.

## What has been done

### HighMap backend

* Added a capability-gated native Metal backend with a portable no-Metal stub.
* Added explicit GPU-resident `DeviceArray` and ordered `DeviceSession`
  ownership/synchronization.
* Added resident noise, spectral, thermal, blend, Gabor, and supporting
  operations across the previous phases.
* Added the Phase 7 `morphological_gradient` Metal kernel and synchronous
  wrapper.
* Added Phase 8 focused DeviceArray tests (7 tests) for residency, hybrid
  boundaries, fallback exception, repeated execution, and cache-handoff
  discipline.

### Hesiod integration

* Phases 5–7 established the four-node `SpectralEqualizer.hsd` resident chain
  (CoherentNoise FBM → SpectralEqualizer → Thermal/Linear → Blend ADD) with
  one terminal readback.
* Phase 7 added explicit `MorphologicalGradient` eligibility and lifecycle
  guards, plus `texturing_101.hsd` Graph D coverage.
* Phase 8 tightens hybrid execution: the two remaining host-only fallbacks
  now explicitly materialize (`prepare_host_node`) before their CPU/OpenCL
  path reads a resident input, and `SpectralEqualizer` reports
  `multi-tile halo fallback` rather than a silent host mismatch.

## Validation summary (2026-09-01, Apple M3, Release)

* HighMap Metal-focused suite: 39/39 passed (32 MetalBackend + 7 MetalPhase8).
* HighMap full suite: 360 passed, 1 pre-existing `PathSplines.PreservePathShape` failure.
* HighMap no-Metal suite: 39 Metal tests skipped cleanly, same spline failure.
* HighMap Phase 3 chain benchmarks: 0 uploads / 1 readback / 1 command buffer / 1 sync.
* Hesiod `SpectralEqualizer.hsd` parity `1e-2` PASS at 512²/1024²/2048² (4096² once).
* Hesiod 1024² median: ~242 ms fallback vs ~115 ms resident (2.1×).

## Current limitations and next work

The `SpectralEqualizer` global-frequency/blurs remain single-logical-tile only;
tiled graphs fall back with seam checks. The persistent cache is opt-in. See
`PHASE8_REVIEW.md` for the full benchmark and transfer table, and for the
suggested next steps.

