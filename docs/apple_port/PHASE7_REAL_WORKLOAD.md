# Phase 7 Real Workload: Graph D

## Selected graph

Graph D is the shipped bootstrap `data/bootstraps/texturing_101.hsd`.

The selected node is `MorphologicalGradient` id `26`. Its real graph context is:

```text
terrain/noise and soil branches → SelectSoilFlow (id 20)
                                      ↓
                               MorphologicalGradient (id 26)
                                      ↓
                           Colorize/texture branches
                                      ↓
                              TextureAdvection → Preview
```

This is not a synthetic two-node test: the selected family has meaningful
terrain work upstream and real color, texture, advection, and presentation
branches downstream. The bootstrap intentionally contains many host-only
branches, so the measurement exposes the selected resident island and its
explicit graph boundaries.

## Phase 6 baseline boundary

Before Phase 7, this node was not a resident candidate and the selected graph
used the host/OpenCL implementation throughout. The Phase 7 benchmark retains
that execution as the fallback control: at 512² its three samples were
207.138, 227.725, and 231.444 ms; at 1024² they were 1163.717, 1170.064, and
1183.983 ms. These are the pre-Phase-7-equivalent controls; repeated Phase 7
measurements use median/min/max samples and retain parity as the primary gate.

## Phase 7 measurement contract

The same graph and CLI configuration are used before and after the new path at
512², 1024², 2048², and 4096² where practical. The report records:

* node execution classification and fallback reason;
* wall, GPU, upload, readback, and resident-memory metrics;
* output parity against the fallback run;
* the 1×1 control and 2×1, 1×2, 2×2, and 4×4 tiled classifications;
* seam-specific error for tiled fallback runs.

Graph D's `MorphologicalGradient` core runs on Metal at 1×1 with identity
post-processing, and its node diagnostic is `resident Metal` with detail
`morphological_gradient`. The graph has one upload for its host-produced input
and one terminal readback; downstream host branches remain unchanged. At 2×1,
1×2, 2×2, and 4×4 the selected node reports
`MorphologicalGradient: multi-tile halo fallback`, with whole-output and seam
parity passing. The separate `meander.hsd` workload is retained as a lifecycle
boundary check: its selected node correctly reports
`MorphologicalGradient: closed Metal session` after an earlier host materialize,
and its end-to-end parity also passes. These are deliberate boundaries, not a
claim that a generic halo scheduler or host-only post-processing is resident.
