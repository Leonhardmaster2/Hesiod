# Phase 5: 4096² Regression Investigation

Date: 2026-08-28. Machine: Apple M3, macOS Metal backend, OpenCL reference
also available on the Apple M3.

## Workload and method

The workload is `data/examples/SpectralEqualizer.hsd`. Its runtime graph is
`CoherentNoise`/the HSD `NoiseFbm` label → `SpectralEqualizer` and `Thermal`
→ ADD `Blend`. Phase 4 used host/OpenCL CoherentNoise and SpectralEqualizer,
with resident Metal Thermal and Blend.

The Phase 4 4096² result was not a stable per-node regression. The benchmark
was extended with process RSS, process high-water RSS, Metal working-set
capacity, live/peak Metal bytes, allocation/reuse counters, transfer bytes,
command-buffer/synchronization counts, GPU time, and an order switch. Each
sample evaluates fallback and resident modes in one process; the two order
sets below were run separately to expose warm-up and system-state effects.

## Repeated Phase 4 measurements at 4096²

| Order | Mode | Median wall ms | Min–max ms | Population SD ms |
|---|---|---:|---:|---:|
| resident first | fallback | 22,444.205 | 21,575.632–23,988.268 | 963.046 |
| resident first | resident | 22,743.073 | 20,866.542–23,504.583 | 1,009.468 |
| fallback first | fallback | 12,424.796 | 11,172.369–12,489.736 | 503.438 |
| fallback first | resident | 13,041.692 | 12,741.389–13,211.713 | 156.093 |

Five samples were collected in each order. The same graph and parameters
therefore produce roughly 11–13 second or 21–24 second runs depending on
the preceding system state. This is strong evidence against describing the
old result as a deterministic 4096² Metal allocation failure.

## Resource evidence

In the Phase 4 partial-residency runs, peak Metal bytes were
`335,544,320` (~320 MiB), while Metal reported a recommended maximum working
set of `5,726,633,984` bytes (~5.73 GiB). Allocation time was sub-millisecond
to roughly 1.7 ms, and the run recorded six allocations and six reuses. The
Metal resource footprint was not close to the reported working-set limit.

The old resident-first samples did show large and variable process RSS, up to
about 1.71 GiB, while resident Metal bytes remained fixed. This is consistent
with unified-memory/cache and OpenCL/Metal system-state contention, but it is
not evidence of Metal working-set exhaustion. No Instruments or Metal System
Trace capture was available in this run, so a separate thermal-throttling
contribution cannot be quantified.

## Causal conclusion

The reproducible factor is benchmark ordering/system state. The old graph
left CoherentNoise and the bandwidth-heavy OpenCL SpectralEqualizer on the
host-side graph while Metal work was also part of the evaluation. On a unified
memory Apple M3, this creates an opportunity for CPU/OpenCL/Metal bandwidth
contention. The diagnostics show no allocation bottleneck and no approach to
`recommendedMaxWorkingSetSize`.

The experiment did not include a standalone A/B harness that holds equivalent
DeviceArrays alive while running only a host SpectralEqualizer, so it does not
claim to isolate memory bandwidth from all other warm-up and driver effects.
It does establish that the previous single-sample regression was confounded
by ordering and that eliminating the host/OpenCL spectral branch is the
appropriate graph-level intervention.

## Phase 5 result

After making the selected graph fully resident, fresh 4096² runs measured
`6,518.662`, `6,593.112`, and `6,595.403` ms resident, versus
`8,642.986`, `8,707.482`, and `8,806.522` ms fallback. The resident graph had
four resident nodes, zero host compute nodes, zero uploads, one terminal
readback, and a peak Metal footprint of `1,073,741,836` bytes (~1.0 GiB),
still well below the recommended working set. Graph parity passed with a
maximum absolute error of `0.00117421`.

The large-map regression is therefore removed for this compatible graph
configuration. The explicit single-tile eligibility is intentional: tiled or
masked configurations continue through the established fallback path until
their boundary and overlap semantics are audited separately.
