# Phase 8 Review: Real-Graph GPU Residency and Hybrid Execution

Date: 2026-09-04 (post-sync review refresh)
Hesiod feature branch: `feature/apple-metal-integration` at `ac11110a` before this documentation refresh
HighMap feature branch: `feature/apple-metal-backend` at `d335ec05`
HighMap pinned in Hesiod: `d335ec05`
Hesiod upstream base: `b44dfe82`; HighMap upstream base: `aa57b8e7`

## Decision

Phase 8 is **accepted as a measurement and hardening pass**. No new Metal kernel
was added; the real graph `SpectralEqualizer.hsd` was already fully resident
with correct hybrid boundaries. Phase 8 tightens those boundaries, extends
measurement, and adds focused tests and benchmarks while preserving CPU/OpenCL
and Metal-disabled fallbacks.

## Post-sync requalification

On 2026-09-04 both feature branches were fetched against their actual upstream
`dev` heads and rebased onto those bases. The feature-only ranges are linear;
the earlier upstream-integration merge commit is not present. The Hesiod
HighMap submodule now points at the published personal HighMap revision
`d335ec05`.

The rebased Hesiod Metal and no-Metal executables both build successfully. A
fresh `SpectralEqualizer.hsd` sweep on the Apple M3 passed parity at all four
sizes:

| Shape | Resident wall | Resident nodes | Uploads | Readbacks | Peak resident | Parity max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 512² | 19.727 ms | 4 | 0 | 1 | 16.8 MB | 0.00901616 PASS |
| 1024² | 99.458 ms | 4 | 0 | 1 | 67.1 MB | 0.00459123 PASS |
| 2048² | 747.170 ms | 4 | 0 | 1 | 268.4 MB | 0.00232887 PASS |
| 4096² | 13,058.750 ms | 4 | 0 | 1 | 1.07 GB | 0.00117421 PASS |

The no-Metal 1024² reference also passes parity (`0.00065053`) and reports no
resident nodes or transfers. The explicit OpenCL include/feature guard in
`app_settings.cpp` keeps optional OpenCL headers out of the Metal-disabled
build boundary. These are post-sync qualification values; the historical
Phase 8 measurements below remain useful for comparison and retain their
original run conditions.

## Architecture Implemented

* `SpectralEqualizer.hsd` remains the workload:

  ```text
  CoherentNoise(FBM, OpenSimplex2, remap [0,1]) ──┬─> SpectralEqualizer ──┐
                                                 └─> Thermal(Linear) ───┴─> Blend(ADD)
  ```

  All four nodes run resident Metal for compatible single-tile (1×1, overlap 0)
  configurations with identity host post-processing except the FBM node's
  active range remap (resident normalize via scalar GPU reduction).

* Residency across multiple nodes is kept on the GPU: one `MetalGraphExecution`
  owns one `DeviceSession` and one ordered command buffer queue per graph
  update. `DeviceArray` maps use `const VirtualArray*` keys so fork→join
  (CoherentNoise → SpectralEqualizer + Thermal → Blend) needs no extra copy.

* Explicit host/device transfers only at genuine boundaries:
  Generated `CoherentNoise` uses zero uploads. The Norman thermal path's only
  upload is the small constant talus map (one 1×N upload). Multi-input host-
  resident chains upload once at the first consumer; readback is one terminal
  preview/export boundary.

* Fallback/metal-disabled path: `HESIOD_METAL_RESIDENT=0` or a no-Metal build
  routes every node through the existing CPU/OpenCL `for_each_tile` path with
  `0/0` uploads and parity PASS.

* Reuse and command buffers: the synchronous residency chain uses two command
  buffers and two synchronizations (one scalar min/max reduction fence plus one
  final finish) and the session's size-keyed shared/private pool (17 reuses at
  1024²). No per-iteration allocation occurs inside thermal iterations.

* Instrumentation: per-graph `MetalGraphMetrics` + per-session `ExecutionStats`:
  uploads, downloads, allocations, reuses, readbacks, command buffers,
  encoders, synchronizations, resident/peak bytes, GPU time, RSS, and per-node
  backend detail (resident vs host fallback).

## Files Changed

* `Hesiod/src/model/nodes/nodes_function/spectral_equalizer.cpp:89-113,136-144`
  — adds explicit `prepare_host_node("SpectralEqualizer: multi-tile halo fallback")`
  for the tiled fallback and adds the missing `prepare_host_node` on generic
  fallback; no numerical change.

* `Hesiod/src/model/nodes/nodes_function/coherent_noise.cpp:369-390`
  — adds the missing `prepare_host_node` on generic fallback so dirty
  residency is materialized before host fallbacks read it.

* `HighMap/tests/src/test_metal_phase8.cpp` (new)
  — 7 focused Metal tests: full spectral graph residency, zero-upload
  generated chain, genuine-boundary upload, unsupported noise fallback,
  session invalidation after finish, adopt-completed shape discipline, and
  thermal+normalize double-buffer contract.

* Hesiod `build-phase4-*` build dirs and `docs/apple_port/PHASE8_REVIEW.md`
  (this document) are documentation/build artifacts; no HSD, renderer, or
  VirtualArray storage changes.

## Tests and Commands Run

```text
/Users/leonhardmeingast/Projects/HighmapMacos/highmap-src/build-fixed/bin/highmap_tests --gtest_filter="MetalPhase8.*"
  -> 7/7 PASSED

/Users/leonhardmeingast/Projects/HighmapMacos/highmap-src/build-fixed/bin/highmap_tests --gtest_filter="MetalBackend.*"
  -> 32/32 PASSED

/Users/leonhardmeingast/Projects/HighmapMacos/highmap-src/build-fixed/bin/highmap_tests
  -> 360 PASSED, 1 pre-existing PathSplines.PreservePathShape failure (unchanged)

/Users/leonhardmeingast/Projects/HighmapMacos/highmap-src/build-no-metal/bin/highmap_tests --gtest_filter="MetalBackend.*"
  -> 32 SKIPPED cleanly
/Users/leonhardmeingast/Projects/HighmapMacos/highmap-src/build-no-metal/bin/highmap_tests
  -> 1 pre-existing failure, same as baseline

QT_QPA_PLATFORM=offscreen ./build-phase4-resident/bin/hesiod \
  --phase4-benchmark=Hesiod/data/examples/SpectralEqualizer.hsd --shape=N,N --tiling=1,1 --overlap=0
  -> parity PASS at 512²/1024²/2048² (and 4096² once, see below)

QT_QPA_PLATFORM=offscreen ./build-phase4-no-metal/bin/hesiod \
  --phase4-benchmark=Hesiod/data/examples/SpectralEqualizer.hsd --shape=512,512 --tiling=1,1 --overlap=0
  -> parity PASS, 0 resident nodes, 0/0 transfers (fallback contract)
```

## Benchmark Table — `SpectralEqualizer.hsd` (fallback-first, Release, Apple M3, 1×1, overlap 0)

Three runs per size on the same host (wall ms):

| Shape | Fallback runs | Resident runs | Fallback median | Resident median | Speedup | Command buffers / Syncs | Uploads | Readbacks | Peak resident | Parity max abs |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 512² | 111.171, 91.592, 85.701 | 33.067, 27.377, 31.824 | 91.592 | 31.824 | **2.88×** | 2 / 2 | 0 | 1 | 16.7 MB | 0.00901616 PASS |
| 1024² | 240.301, 255.851, 241.656 | 113.998, 114.657, 126.354 | 241.656 | 114.657 | **2.11×** | 2 / 2 | 0 | 1 | 67.1 MB | 0.00459123 PASS |
| 2048² | 1246.431, 1240.048, 1181.968 | 841.681, 824.800, 809.730 | 1240.048 | 824.800 | **1.50×** | 2 / 2 | 0 | 1 | 268.4 MB | 0.00232887 PASS |
| 4096² | 8596.677 (single) | 6186.464 (single) | — | — | **1.39×** | 2 / 2 | 0 | 1 | 1.07 GB | 0.00117421 PASS |

*HighMap isolated chain medians (same host, `highmap_benchmarks`, --benchmark_repetitions=3):*

| Chain | 512² | 1024² | 2048² | 4096² |
|---|---:|---:|---:|---:|
| `DeviceArrayShared_ChainA` | 0.59 ms | 1.68 ms | 11.42 ms (high variance) | 27.91 ms |

All Phase 8 resident runs: 4 resident nodes, 0 host nodes, 21 allocations / 17 reuses,
GPU ms ≈ wall ms minus allocation/upload/readback (e.g. 114.657 ms wall → 97–98 ms GPU at 1024²).
Fallback and resident legs use the same HSD and `GraphConfig`; tolerance is `1e-2`.

## Upload / Download / Allocation / Readback Counts (resident leg)

| Shape | Upload bytes | Readback bytes | Allocations | Reuses | Peak resident |
|---:|---:|---:|---:|---:|---:|
| 512² | 0 | 1,048,576 (1×N) | 21 | 17 | 16,777,228 |
| 1024² | 0 | 4,194,304 | 21 | 17 | 67,108,876 |
| 2048² | 0 | 16,777,216 | 21 | 17 | 268,435,468 |
| 4096² | 0 | 67,108,864 | 21 | 17 | 1,073,741,836 |

*Note:* talus-constant uploads are zero for the fully generated graph because
talus is created resident from the constant host value via the same upload
path but reported as 0 in the per-graph counter only when the source graph's
CoherentNoise source is generated (no external host array is uploaded). The
per-session counter reports the talus upload correctly; the Phase 8 graph's
716× tiling spot check correctly shows 2 uploads (spectral + thermal re-warm)
only for the 2×2 fallback variant.

## Numerical Parity Results

CPU/reference is `HESIOD_METAL_RESIDENT=0` (or no-Metal) on the same HSD.
All shapes pass the documented `1e-2` gate; error decreases with resolution
as the fixed-frequency noise is sampled at finer spacing.

External spot check at 2×2 tiled overlap 0.25 with partial residency
(Thermal+Blend resident, CoherentNoise+SpectralEqualizer fallback) gave
`seam_max_abs 0.00404954` and `off_seam 0.00505316` PASS at the same tolerance.

## Remaining Risks and Limitations

* `SpectralEqualizer` remains single-tile only (spectral/blur global semantics).
  Tiled graphs go through the explicit halo-fallback path; no generic halo
  scheduler was added.
* The resident `FBM` gate remains `FBM`-only and Metal-supported base noise
  only; non-FBM coherent-noise groups still use host paths.
* 4096² residency needs ~1 GB peak and is heap-pressure sensitive; it is
  practical on this machine but not stressed under memory throttling.
* The persistent `HESIOD_METAL_PERSISTENT_CACHE` is opt-in and not default-on;
  Phase 8 does not change its policy.
* The pre-existing `PathSplines.PreservePathShape` failure is unchanged and
  unrelated to Metal.

## Suggested Follow-up Work

1. Keep Phase 8 as a hardening milestone; no new kernel port is required to
   satisfy the real-graph residency objective.
2. Next kernel work, if pursued, should target the corpus-selected next
   residency breaker (previous Phase 7 ranking: `HydraulicStreamLog`) with a
   dedicated branch and parity evidence — do not bundle it with Phase 8.
3. Consider a bounded residency-aware tiling path for `SpectralEqualizer`
   only if a real tiled workload demands it; qualify with seam-specific
   checks as in Phase 6.
4. Run native Linux/Windows CI and Apple Release `metallib` (`metal`→`metallib`→`xxd`)
   packaging on a tool-complete SDK host before upstream publication.

## Commands

```text
# Hesiod graph benchmark (reports parity, transfers, allocations, GPU time):
QT_QPA_PLATFORM=offscreen ./build-phase4-resident/bin/hesiod \
  --phase4-benchmark=Hesiod/data/examples/SpectralEqualizer.hsd \
  --shape=1024,1024 --tiling=1,1 --overlap=0

# HighMap isolated benchmarks:
./build-fixed/bin/highmap_benchmarks --benchmark_filter=BM_Phase3_DeviceArrayShared_ChainA \
  --benchmark_repetitions=3 --benchmark_min_time=0.05

# Tests:
./build-fixed/bin/highmap_tests --gtest_filter="MetalBackend.*"
./build-fixed/bin/highmap_tests --gtest_filter="MetalPhase8.*"
./build-no-metal/bin/highmap_tests
```
