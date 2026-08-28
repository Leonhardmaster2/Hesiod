# Phase 5: SpectralEqualizer Audit

## Existing algorithm

HighMap's `hmap::gpu::spectral_equalizer` is a spatial multi-band filter; it
does not use an FFT or frequency-domain dependency. For six weights it builds
five logarithmically spaced radii with `logspace`, creates the original input
plus five smoothed copies, forms adjacent differences, and accumulates the
bands in reverse weight order.

Each `gpu::smooth_cpulse` is a separable Gaussian-like pulse: one horizontal
and one vertical neighborhood pass. Thus the selected six-weight graph uses
ten neighborhood passes, followed by pointwise subtraction and weighted
accumulation. The OpenCL implementation dispatches the smooth kernels, but
the surrounding `Array` pyramid and arithmetic are still host-managed.

## Cost and residency analysis

The dominant costs are repeated full-array neighborhood bandwidth, five full
size temporary arrays, and the pointwise band rebuild. There is no reduction,
sort, histogram, or FFT to move to Accelerate or MPS. A CPU-only optimization
would still leave the graph boundary between SpectralEqualizer and resident
Blend/Thermal, and would retain the unified-memory contention observed in the
4096² investigation.

## Metal implementation

HighMap now provides both:

- `DeviceSession::smooth_cpulse(DeviceArray, radius)`, implemented as two
  in-session Metal compute passes with session buffer reuse.
- `DeviceSession::spectral_equalizer(DeviceArray, weights, ir_min, ir_max)`,
  which keeps the blur pyramid, band differences, and accumulation as
  `DeviceArray` values in one session.

Synchronous `Array` wrappers are also present for independent HighMap use.
The Hesiod node opts into the resident form only for a single tile, no mask,
and identity generic post-processing. The terminal output is materialized
once by the graph bridge; intermediate spectral arrays never cross to the
host.

## Numerical evidence

`MetalBackend.ResidentSpectralEqualizerMatchesOpenCL` passes in the HighMap
Metal suite. The primary Hesiod graph passes the existing `1e-2` maximum
absolute-error gate at every tested resolution:

| Shape | Maximum absolute error |
|---:|---:|
| 512² | 0.00901616 |
| 1024² | 0.00459123 |
| 2048² | 0.00232887 |
| 4096² | 0.00117421 |

The 512² result is the tightest margin and remains below the existing gate;
the direct HighMap primitive test is substantially tighter on its focused
fixture.

## Decision

Metal is justified for this graph because it removes the dominant residency
break, preserves the existing spatial algorithm, avoids a new FFT dependency,
and keeps all intermediate state device-resident. No separate Accelerate,
MPS, or CPU vectorization path was added in this phase.
