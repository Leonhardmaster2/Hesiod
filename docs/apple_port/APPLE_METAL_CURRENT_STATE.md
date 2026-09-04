# Apple Metal Current State

Snapshot date: 2026-09-04

This is the current state of the Apple Silicon/Metal work across the Hesiod
application and its HighMap backend. The snapshot is on the personal GitHub
repositories, not the upstream `ottolink-dev` repositories.

## Repository state

| Repository | Personal remote | Branch | Implementation HEAD at snapshot | Upstream base |
|---|---|---|---|---|
| Hesiod | `Leonhardmaster2/Hesiod` | `feature/apple-metal-integration` | `ac11110a7ea25c3364b4d1f7f9525af8456c6aca` before this documentation refresh | `b44dfe8203ab827c9f76b84dacd18a4f25922d19` |
| HighMap | `Leonhardmaster2/HighMap` | `feature/apple-metal-backend` | `d335ec051f7c7d5cb241e80baa5e4139cf4361de` | `aa57b8e7638f68bafd2b9fc595c682ec9204cd4e` |

Both feature histories are linear from the actual fetched upstream `dev` bases
and contain zero merge commits in the feature-only range. The Hesiod submodule
is pinned to the published personal HighMap feature revision
`d335ec051f7c7d5cb241e80baa5e4139cf4361de`. Existing local build directories,
generated images, logs, and other untracked artifacts were preserved. The
pre-sync states are also retained on personal backup branches.

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

## Validation summary (2026-09-04, Apple M3, Release)

* HighMap Metal-focused suite: 39/39 passed (32 MetalBackend + 7 MetalPhase8).
* HighMap full suite: 433 passed, 2 intentional portability skips, and 1
  pre-existing `PathSplines.PreservePathShape` failure.
* HighMap no-Metal full suite: 385 passed, 50 expected backend skips, and the
  same pre-existing spline failure. The no-Metal focused Metal/Phase 8 tests
  skip cleanly because no Metal device is available.
* HighMap isolated 1024² resident Chain A: 1.477 ms, 0 uploads, 1 readback,
  1 command buffer, and 1 final synchronization.
* HighMap isolated 1024² hydraulic VPipes: 172.811 ms, with the expected
  explicit hydraulic pass allocations and transfer accounting.
* Hesiod `SpectralEqualizer.hsd` parity `1e-2` PASS at 512²/1024²/2048²/4096².
  The current resident legs report 4 resident nodes, 0 uploads, and 1 terminal
  readback at every tested size.
* Current resident wall times: 19.727 ms (512²), 99.458 ms (1024²),
  747.170 ms (2048²), and 13,058.750 ms (4096²). The 4096² peak resident
  allocation is 1,073,741,836 bytes.
* Current 1024² parity maxima are `0.00459123` for Metal and `0.00065053`
  for the no-Metal reference run; both pass.

The historical Phase 5 and Phase 8 benchmark tables retain their original
measurement dates and are not silently replaced by this post-sync smoke
qualification.

## Current limitations and next work

The `SpectralEqualizer` global-frequency/blurs remain single-logical-tile only;
tiled graphs fall back with seam checks. The persistent cache is opt-in. The
OpenCL settings API is explicitly guarded so the no-Metal/optional-OpenCL
build remains compilable. See `PHASE8_REVIEW.md` for the full benchmark and
transfer table, and for the suggested next steps.
