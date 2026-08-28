# Hesiod Apple Metal Phase 6 Baseline

Date: 2026-08-28. This baseline was collected after refreshing both remotes
with `git fetch upstream --prune` and `git fetch origin --prune`.

## Revisions and machine

| Item | Value |
|---|---|
| Hesiod branch | `feature/apple-metal-integration` |
| Hesiod upstream/dev | `4c0ee2156b987ec221e92a229e81d4d9ec9c6f2b` |
| HighMap branch | `feature/apple-metal-backend` |
| HighMap upstream/dev | `e0d1279fced75cf7556de233128abfb5650e25b5` |
| HighMap Phase 6 dependency | `f8b91e1f12c7e1ab9cb796c8aae822c880525734` |
| macOS | 27.0 (`26A5378j`) |
| architecture | arm64 |
| machine identifier | `Mac15,12` |
| Metal SDK/framework | `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` |
| HighMap build | Release, OpenCL ON, Metal ON, runtime shader compilation |

The pre-existing untracked build directories and generated images remain
outside the commits. The HighMap submodule is pinned to the dependency commit
listed above.

## HighMap checks

Commands:

```text
./build-fixed/bin/highmap_tests --gtest_filter='MetalBackend.*'
./build-fixed/bin/highmap_tests
./build-no-metal/bin/highmap_tests
```

Results:

| Suite | Result |
|---|---|
| Metal-focused | 31/31 passed |
| Full Metal-enabled | 352 passed, 1 pre-existing `PathSplines.PreservePathShape` failure, 353 total |
| No-Metal | 321 passed, 31 Metal tests skipped, 1 pre-existing `PathSplines.PreservePathShape` failure, 353 total |
| New Gabor parity test | passed against OpenCL |
| New completed-resource handoff test | passed |

The `PathSplines` tolerance failure is unchanged from the Phase 5 baseline;
it is not in the Metal or Phase 6 changes. The no-Metal build compiles the
same public API through the existing unavailable-backend stubs.

## Hesiod Phase 5 reproduction anchor

The established primary `SpectralEqualizer.hsd` graph remains the four-node
resident chain: CoherentNoise → SpectralEqualizer, with Thermal and ADD Blend
feeding the terminal output. The Phase 5 five-sample medians remain the
comparison anchor:

| Shape | Fallback median | Resident median | Speedup | Resident parity max abs |
|---:|---:|---:|---:|---:|
| 512² | 94.925 ms | 21.308 ms | 4.45× | 0.00901616 |
| 1024² | 250.236 ms | 122.364 ms | 2.05× | 0.00459123 |
| 2048² | 1,270.759 ms | 854.844 ms | 1.49× | 0.00232887 |
| 4096² | 8,707.482 ms | 6,593.112 ms | 1.32× | 0.00117421 |

The Phase 6 implementation preserves this path. Current spot checks also
show zero host uploads, one terminal readback, and four resident nodes for
the primary graph. The broader Phase 6 benchmark table is in
`PHASE6_BENCHMARKS.md`.

## Baseline policy

`HESIOD_METAL_RESIDENT=0` remains the explicit fallback switch. The optional
persistent cache is off by default and is enabled only for the Phase 6 cache
experiment with `HESIOD_METAL_PERSISTENT_CACHE=1`.
