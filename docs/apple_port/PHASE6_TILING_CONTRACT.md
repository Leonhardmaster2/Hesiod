# Phase 6 Tiling Contract

## Actual VirtualArray model

`VirtualArray` owns a logical `shape` and `bbox`, a `tile_shape`, a halo width,
and a tile storage backend. `get_max_tiles()` is the ceiling of logical shape
over tile shape. A `TileRegion` includes the tile's interior shape, halo, and
region bbox; edge tiles have reduced interior dimensions and asymmetric halo.

The established whole-array boundary is:

```text
VirtualArray::to_array(VA_SINGLE_ARRAY) -> logical Array
Metal DeviceArray operation -> logical Array
VirtualArray::from_array(VA_SINGLE_ARRAY) -> tile interiors/halo storage
```

`from_array` and `to_array` preserve the existing inner-cell mapping. They do
not silently discard the configured halo. Existing
`smooth_overlap_buffers()` remains the explicit overlap blend operation for
tile-producing host paths.

## Operation classification

| Class | Examples | Phase 6 treatment |
|---|---|---|
| Pointwise/procedural whole-array | GaborWaveFbm, resident FBM | one logical DeviceArray; scatter through VA boundary |
| Neighbor/halo-sensitive | Thermal, smoothing, advection | retain existing eligibility; no new generic tiled claim |
| Global reduction | normalize/min/max, spectral range | reduction semantics remain explicit; scalar sync is allowed only inside the resident API |
| Boundary/stitching | MakePeriodic, MakePeriodicStitching | host boundary remains visible and is benchmarked |
| Unsupported/post-process | envelope, gamma/gain/smoothing/saturation | materialize and use original fallback |

The Gabor Phase 6 operation is whole-logical-array resident even for a tiled
VirtualArray. This is safe for its deterministic procedural evaluation and
avoids pretending that a single tile is an independent Gabor domain. Inputs
are gathered once; outputs are scattered by the existing VA implementation.

## Required invariants

- logical shape and bbox are passed unchanged;
- optional displacement/control/angle arrays must match logical shape;
- tile grids and edge tiles are represented by the existing VirtualArray
  metadata;
- downstream host operations materialize before reading a resident array;
- global reductions return one scalar range for the logical array, not one
  unrelated range per tile;
- no seam correction is enabled implicitly by the Metal bridge.

The CLI emits `PHASE6_TILING` with full parity, seam-line parity, and off-seam
parity. Seam lines include both cells adjacent to each internal tile boundary;
this is a diagnostic comparison between fallback and resident logical outputs,
not a new smoothing algorithm.
