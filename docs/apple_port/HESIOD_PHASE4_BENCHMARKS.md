# Hesiod Phase 4 Benchmarks

## Hesiod real-graph sweep

Measured on the Apple M3 with one tile and zero overlap. Times are wall-clock
milliseconds for graph evaluation; `gpu_ms` is the Metal execution statistic.

| Shape | Mode | Wall ms | Preview ms | GPU ms | Upload bytes | Readback bytes |
|---:|---|---:|---:|---:|---:|---:|
| 512² | fallback | 78.684 | 1.222 | 0 | 0 | 0 |
| 512² | resident | 66.088 | 1.433 | 14.454 | 2,097,152 | 1,048,576 |
| 1024² | fallback | 224.279 | 6.615 | 0 | 0 | 0 |
| 1024² | resident | 176.573 | 4.470 | 36.748 | 8,388,608 | 4,194,304 |
| 2048² | fallback | 1,158.378 | 18.042 | 0 | 0 | 0 |
| 2048² | resident | 1,083.302 | 18.757 | 339.815 | 33,554,432 | 16,777,216 |
| 4096² | fallback | 32,691.875 | 82.682 | 0 | 0 | 0 |
| 4096² | resident | 36,493.219 | 93.273 | 17,848.396 | 134,217,728 | 67,108,864 |

Node-level timing and evaluation counters are emitted as `PHASE4_NODE` lines
by the benchmark CLI. Resident Thermal and Blend are the two resident nodes;
CoherentNoise and SpectralEqualizer remain host nodes. The 4096² resident run
is slower wall-clock in this sample, while the smaller runs show the expected
benefit from avoiding an intermediate readback; this is a measurement result,
not a claim of universal speedup.

The measured node timings (milliseconds, in node-id order `10 Blend`, `11
Thermal`, `8 CoherentNoise`, `9 SpectralEqualizer`) were:

| Shape | Fallback node ms | Resident node ms |
|---:|---|---|
| 512² | 0.132, 21.783, 7.496, 46.556 | 1.470, 1.657, 4.698, 40.035 |
| 1024² | 0.800, 62.888, 16.925, 138.989 | 5.001, 5.575, 11.098, 109.667 |
| 2048² | 2.780, 403.317, 44.718, 706.054 | 19.908, 21.783, 30.914, 647.143 |
| 4096² | 18.502, 18,152.254, 150.093, 14,369.456 | 97.467, 107.036, 535.610, 17,774.143 |

## Supporting HighMap resident measurements

The required 1024² HighMap checks were run from the pinned HighMap Metal
branch:

| Benchmark | Measured total | Transfers / sync |
|---|---:|---|
| `BM_Phase3_DeviceArrayShared_ChainA/1024` | 3.41467 ms | 0 uploads, 1 final sync, 4.1943M readback bytes |
| `BM_Apple_Metal_HydraulicVPipes/1024/100` | 178.821 ms | 8.38861M upload bytes, 4.1943M readback bytes, 900 encoders, 164.815 ms sync |
