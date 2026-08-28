# Phase 5: CoherentNoise Audit

## Selected configuration

The primary HSD stores the source node with the `NoiseFbm` label. At runtime
Hesiod dispatches it through the CoherentNoise implementation, in the `FBM`
group, using OpenSimplex2 (`noise_type=4`), seed 0, eight octaves, weight
0.7, persistence 0.5, lacunarity 2.0, and an active `[0, 1]` remap.

The legacy implementation calls HighMap's FBM path per tile and then applies
generic post-processing. The selected configuration has no envelope, warp
family, or mask input.

## HighMap mapping

The resident path reuses the existing Metal base-noise coverage through a new
`DeviceSession::noise_fbm` operation. Its octave recurrence matches the
HighMap/OpenCL FBM implementation, including weighted amplitude update,
persistence, lacunarity, seed conversion, optional periodic cell snapping,
and optional control/noise-x/noise-y DeviceArray inputs.

The active range remap is implemented as a resident global min/max reduction
followed by a pointwise normalize kernel. It synchronizes two scalar values,
not the terrain buffer. The graph still reports zero host upload bytes for
generated noise.

## Eligibility boundaries

The integration is deliberately conservative:

- FBM only; Ridged, IQ, Jordan, Parberry, PingPong, and Swiss remain fallback.
- Metal-supported base noise types only.
- No envelope input.
- Single-tile output, because the resident bbox and overlap contract is not
  yet generalized to distributed tiled VirtualArrays.
- Identity inverse/gamma/gain/smoothing/saturate post-processing; active range
  remap is the one supported post-process operation.

Unsupported cases retain the existing CPU/OpenCL behavior rather than silently
changing semantics.

## Evidence

`MetalBackend.ResidentNoiseFbmAndNormalizationMatchOpenCL` passes and checks
that generated resident noise uses zero upload bytes. Repeated same-parameter
primary-graph evaluations produce the same parity result, and the full graph
passes the `1e-2` gate at 512², 1024², 2048², and 4096². The selected graph's
resident execution reports four resident nodes, zero host compute nodes, zero
uploads, and one final readback.
