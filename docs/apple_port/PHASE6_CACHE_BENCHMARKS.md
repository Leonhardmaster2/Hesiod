# Phase 6 Persistent Cache Benchmarks

## Method

The cache is enabled only for these measurements with
`HESIOD_METAL_PERSISTENT_CACHE=1`. The same graph is loaded, one node is
edited, and the dirty update is timed. Cache-off and cache-on runs use the
same resident graph and shape. Every row below is one recorded spot sample;
the Phase 5 repeated medians remain the historical performance anchor.

The edit matrix is:

```text
CoherentNoise seed
SpectralEqualizer rmax
Thermal duration
Blend input1_weight
```

Each cache-on row produced zero uploads and one terminal readback. CoherentNoise
is a full resident recompute, so it is expected to show little cache speedup;
the cache benefit is for unchanged upstream results used by downstream edits.

## Incremental edit timings

| Shape | Edit | Cache off ms | Cache on ms | Speedup | Off uploads | On uploads |
|---:|---|---:|---:|---:|---:|---:|
| 512² | CoherentNoise | 20.877 | 19.143 | 1.09× | 0 | 0 |
| 512² | SpectralEqualizer | 16.714 | 10.497 | 1.59× | 2 | 0 |
| 512² | Thermal | 9.716 | 6.763 | 1.44× | 2 | 0 |
| 512² | Blend | 6.174 | 3.402 | 1.81× | 2 | 0 |
| 1024² | CoherentNoise | 117.117 | 118.642 | 0.99× | 0 | 0 |
| 1024² | SpectralEqualizer | 75.989 | 59.244 | 1.28× | 2 | 0 |
| 1024² | Thermal | 52.277 | 38.967 | 1.34× | 2 | 0 |
| 1024² | Blend | 27.212 | 10.627 | 2.56× | 2 | 0 |
| 2048² | CoherentNoise | 818.336 | 838.839 | 0.98× | 0 | 0 |
| 2048² | SpectralEqualizer | 456.988 | 405.582 | 1.13× | 2 | 0 |
| 2048² | Thermal | 297.523 | 246.180 | 1.21× | 2 | 0 |
| 2048² | Blend | 72.673 | 26.913 | 2.70× | 2 | 0 |
| 4096² | CoherentNoise | 6,335.287 | 6,284.345 | 1.01× | 0 | 0 |
| 4096² | SpectralEqualizer | 3,123.509 | 2,936.288 | 1.06× | 2 | 0 |
| 4096² | Thermal | 2,040.691 | 1,825.564 | 1.12× | 2 | 0 |
| 4096² | Blend | 284.597 | 102.937 | 2.76× | 2 | 0 |

The cache-on path therefore removes two uploads (up to 128 MiB at 4096²)
from each unchanged-input downstream edit. The 4096² resident primary
evaluation remained within the historical 6.5–6.6 s range.

## Cache counters and pressure

The dedicated Blend edit benchmark reported these counters after a cold
evaluation and one warm update:

| Shape | Budget | Persistent bytes | Warm hits | Warm misses | Warm evictions | Warm uploads | Warm readbacks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512² | 1,365 MiB default | 3 MiB | 2 | 0 | 0 | 0 | 1 |
| 1024² | 1,365 MiB default | 12 MiB | 2 | 0 | 0 | 0 | 1 |
| 2048² | 1,365 MiB default | 48 MiB | 2 | 0 | 0 | 0 | 1 |
| 4096² | 512 MiB explicit | 192 MiB | 2 | 0 | 0 | 0 | 1 |

The default budget is one quarter of the reported 5,726,633,984-byte Metal
recommended working set. The explicit 512 MiB run kept the cache at 192 MiB
while the resident peak remained 1,073,741,836 bytes.

A 4096² pressure run with `HESIOD_METAL_CACHE_MB=128` kept persistent bytes at
exactly 128 MiB. It reported one warm hit, one miss, one eviction, one upload,
and one readback; parity remained `0.00117421`. This demonstrates bounded
fallback behavior when the budget cannot retain both parent resources.

Cache entries are completed HighMap DeviceArrays scoped to one GraphNode and
keyed with logical shape, tile shape, and halo metadata. Output invalidation
occurs before a dirty node runs. Preview/export data is not inserted into the
cache.
