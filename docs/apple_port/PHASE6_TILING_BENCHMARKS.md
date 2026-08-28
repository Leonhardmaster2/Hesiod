# Phase 6 Tiling and Overlap Benchmarks

## Method

The authoritative path is the headless CLI benchmark:

```text
./build-phase4-resident/bin/hesiod \
  --phase4-benchmark=Hesiod/data/examples/MakePeriodicStitching.hsd \
  --shape=<width>,<height> --tiling=<x>,<y> --overlap=0.25
```

The command evaluates the established fallback first and then resident Metal,
materializes the same terminal output, and reports full parity plus errors on
the two cells adjacent to every internal tile boundary. The resident path for
this graph is GaborWaveFbm followed by the existing host stitching boundary;
therefore the resident run deliberately reports one resident and one host
node. These are spot measurements for semantic qualification, not a claim of
a repeated-sample speedup.

## Graph C: MakePeriodicStitching, 512²

All runs used overlap `0.25`, the configured overlap value used by the
qualification graph.

| Tiling | Fallback wall ms | Resident wall ms | Resident GPU ms | Resident peak bytes | Resident/fallback tiles | Seam max abs | Off-seam max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1×1 | 95.119 | 75.455 | 3.592 | 2,097,168 | 1 / 1 | n/a | n/a |
| 2×1 | 93.409 | 76.661 | 3.591 | 2,097,168 | 2 / 2 | 1.2e-7 | 3.0e-7 |
| 1×2 | 93.978 | 74.744 | 3.595 | 2,097,168 | 2 / 2 | 2.4e-7 | 3.0e-7 |
| 2×2 | 95.975 | 75.125 | 3.909 | 2,097,168 | 4 / 4 | 2.4e-7 | 3.0e-7 |
| 4×4 | 87.245 | 75.548 | 3.600 | 2,097,168 | 16 / 16 | 2.4e-7 | 3.0e-7 |

Every resident row used two command buffers, two synchronizations, zero host
uploads, and one terminal readback. The 512² parity maximum was `3.0e-7`.

## Graph C: 1024²

| Tiling | Fallback wall ms | Resident wall ms | Resident GPU ms | Resident peak bytes | Seam max abs | Off-seam max abs |
|---|---:|---:|---:|---:|---:|---:|
| 4×4 | 758.313 | 744.549 | 29.173 | 8,388,624 | 3.0e-7 | 4.2e-7 |

This run also used two command buffers, two synchronizations, zero uploads,
and one terminal readback. The resident whole-logical-array operation does
not allocate one independent Gabor buffer per tile; the tile count is still
tracked so downstream host work remains visible.

## Interpretation

The qualified tiling path is a VirtualArray gather/compute/scatter boundary.
It preserves the logical shape and bbox and does not perform implicit seam
blending. The seam-specific maximum across all multi-tile qualification runs
was `3.0e-7`, with no visible or numerical seam introduced. Operations that
need neighborhood halos or whole-array reductions remain on their established
eligibility paths; this benchmark does not promote them to independent tile
kernels.
