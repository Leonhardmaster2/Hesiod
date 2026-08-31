# Apple Metal Current State

Snapshot date: 2026-08-31

This is the current state of the Apple Silicon/Metal work across the Hesiod
application and its HighMap backend. The snapshot is on the personal GitHub
repositories, not the upstream `ottolink-dev` repositories.

## Repository state

| Repository | Personal remote | Branch | Phase 7 implementation HEAD | Upstream base |
|---|---|---|---|---|
| Hesiod | `Leonhardmaster2/Hesiod` | `feature/apple-metal-integration` | `b38d91dc322e5ec0ac960b355528ba768170753f` | `324c3ed51202ec6aa43d89730da39c913060d392` |
| HighMap | `Leonhardmaster2/HighMap` | `feature/apple-metal-backend` | `a0c416dc54d3c57143a589cbf2118b6ceb5bb489` | `e0d1279fced75cf7556de233128abfb5650e25b5` |

The Hesiod branch tip will advance by the documentation commit that contains
this snapshot; the immutable backup ref and final command-line report record
that exact final tip. The implementation heads shown above identify the
tested code state. Both feature histories are linear from their current upstream `dev` bases and
contain zero merge commits in the feature-only range. The Hesiod submodule is
pinned to HighMap `a0c416dc54d3c57143a589cbf2118b6ceb5bb489`.

The durable backup refs are:

```text
Hesiod:  backup/apple-metal-phase7-2026-08-31
HighMap: backup/apple-metal-phase7-2026-08-31
```

They point to the published snapshot heads and are hosted on the personal
`origin` remotes. Existing local build directories, generated images, logs,
and other untracked artifacts were preserved and are intentionally not part
of the Git backup.

## What has been done

### HighMap backend

* Added a capability-gated native Metal backend with a portable no-Metal stub.
* Added explicit GPU-resident `DeviceArray` and ordered `DeviceSession`
  ownership/synchronization.
* Added resident noise, spectral, thermal, blend, Gabor, and supporting
  operations across the previous phases.
* Added the Phase 7 `morphological_gradient` Metal kernel and synchronous
  wrapper. It computes the same clamped disk neighborhood as the existing
  OpenCL path.
* Added OpenCL parity coverage, boundary stress coverage, and CPU/OpenCL/Metal
  benchmarks through 4096².
* Kept shader source embedding and the release `metal` → `metallib` → `xxd`
  packaging path in CMake. Runtime MSL compilation remains available when the
  precompiled tools are absent.

### Hesiod integration

* Added explicit Metal graph eligibility for `MorphologicalGradient`.
* Qualified the resident path for one logical tile, connected float input and
  output, non-negative radius, and identity post-processing; active range
  remap uses the existing resident normalize operation.
* Kept inverse, gain/gamma, smoothing, saturation, mix, multi-tile, missing
  port, unavailable-device, and closed-session cases on explicit host/OpenCL
  fallback paths.
* Added node-level backend diagnostics, including the reason for selected-node
  fallback.
* Added a session lifecycle guard so a host materialization cannot cause a
  later resident node to encode into a finished command buffer.
* Added Graph D coverage using the real `texturing_101.hsd` graph and lifecycle
  fallback coverage using `meander.hsd`.
* Added a deterministic 100-cycle persistent-cache soak with shape/tiling
  changes, graph recreation, project switching, finite-output checks, and
  fingerprint comparison.

## Validation summary

* HighMap Metal-focused tests: 32/32 passed.
* HighMap full suite: 321 passed; the one existing
  `PathSplines.PreservePathShape` failure remains unchanged.
* HighMap no-Metal suite: 321 passed, 32 Metal tests skipped, and the same
  unchanged spline failure.
* Hesiod resident and no-Metal targets both build successfully.
* Graph A (`SpectralEqualizer.hsd`) at 4096²: 6085.190 ms resident versus
  8393.292 ms fallback; parity `0.00117421 PASS`.
* Graph D (`texturing_101.hsd`) median speedup: 1.31× at 512² and 1.06× at
  1024², with whole-output parity passing at every sample.
* Isolated Phase 7 HighMap operation median CPU/Metal speedup: 5.62× at
  2048² and 5.85× at 4096².
* Cache soak at 128 MiB: 146 hits, 2 misses, 0 evictions, 102 finite-output
  checks, 0 invalid outputs, and 0 project fingerprint collisions.
* Cache soak at 1 MiB: 72 hits, 76 misses, 105 evictions, budget respected,
  102 finite-output checks, 0 invalid outputs, and 0 project fingerprint
  collisions.

## Current limitations and next work

The new MorphologicalGradient Metal path is intentionally single-logical-tile
only. Tiled 2×1, 1×2, 2×2, and 4×4 configurations use the original fallback
and pass seam checks. Host-only post-processing remains a deliberate boundary.
The persistent cache is opt-in and is not default-on.

The current Command Line Tools installation does not provide `xcrun metal` or
`xcrun metallib`, so a precompiled metallib and signed application bundle were
not built on this machine. Release CI still needs to exercise that path, along
with native Linux/Windows builds. The Python Hesiod test suite was not run
because `pytest` is not installed in the available system Python.

No universal graph scheduler or renderer migration was added. No Phase 8 work
has started. The next responsible step is upstream/cleanup/release review:
native cross-platform CI, Apple release packaging, review of the unrelated
spline failure, and deciding whether the experimental cache/soak harness
should be split from the first upstream patches.

## Authorship and safety

All feature commits use the configured `Leonhardmaster2` Git identity. The
full feature-range message audit found zero `Co-authored-by`, `Codex`,
`OpenAI`, `ChatGPT`, `Claude`, `Luna`, or `Anthropic` matches. Only the
personal `origin` remotes were used for publication; upstream `dev` and
`main` were not rewritten.
