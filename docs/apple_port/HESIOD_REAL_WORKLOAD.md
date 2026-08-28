# Hesiod Phase 4 Real Workload

## Workload

The integration uses `Hesiod/data/examples/SpectralEqualizer.hsd` rather than
a synthetic graph. It contains a genuine branch and convergence:

```text
CoherentNoise(8) ──> SpectralEqualizer(9) ──> Blend(10)
        └──────────> Thermal(11) ───────────> Blend(10)
```

Thermal is Linear with the default talus and no mask or deposition output.
Blend is ADD with unit weights. These settings satisfy the narrow resident
gates while leaving the noise and spectral nodes on their established host
path.

## Fallback/resident parity

The benchmark command is:

```text
QT_QPA_PLATFORM=offscreen ./build-phase4-resident/bin/hesiod \
  --phase4-benchmark=Hesiod/data/examples/SpectralEqualizer.hsd \
  --shape=<N>,<N> --tiling=1,1 --overlap=0
```

| Shape | Resident nodes | Host nodes | Uploads | Readbacks | Max absolute difference | Status |
|---:|---:|---:|---:|---:|---:|---|
| 512² | 2 | 2 | 2 | 1 | 0.00505316 | PASS |
| 1024² | 2 | 2 | 2 | 1 | 0.00267053 | PASS |
| 2048² | 2 | 2 | 2 | 1 | 0.00136781 | PASS |
| 4096² | 2 | 2 | 2 | 1 | 0.00068808 | PASS |

The comparison gate is `1e-2`. The resident output is materialized once at the
terminal Blend output; the preview conversion then reads the ordinary host
VirtualArray. The dirty Thermal edit recomputes the Thermal→Blend resident
subgraph and reports two resident nodes with zero host nodes.
