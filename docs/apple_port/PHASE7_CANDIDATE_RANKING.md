# Phase 7 Candidate Ranking

## Corpus and method

The current `Hesiod/data/examples/` and `Hesiod/data/bootstraps/` corpus was
scanned after the Phase 7 upstream refresh. Counts below are graph-node labels
from `graph_manager.graph_nodes[*].nodes[*]`; UI captions and repeated JSON
presentation copies were excluded.

| Label/family | Occurrences | Files | Current interpretation |
|---|---:|---:|---|
| GaborWaveFbm | 177 | 157 | Phase 6 family; excluded as already qualified |
| NoiseFbm | 23 | 21 | Mostly legacy-converted to resident-compatible CoherentNoise FBM |
| HydraulicStreamLog | 14 | 14 | Repeated expensive host boundary |
| ColorizeGradient | 46 | 46 | Visual/texture output; excluded from array residency |
| Preview | 33 | 33 | Presentation boundary; excluded |
| CloudLattice | 19 | 19 | Cloud output, not a heightmap array |
| CloudToPath | 14 | 14 | Path/cloud boundary, not an array kernel |
| FloodingUniformLevel | 12 | 12 | Global/water-output semantics; not a safe local resident candidate |
| Blend | 10 | 10 | Phase 5/6 family; already qualified for its supported form |
| MorphologicalGradient | 4 | 4 | Array neighborhood operation; remaining targeted candidate |
| VorolinesFbm | 3 | 2 | Cellular/line kernel with no existing Metal building block |
| Rifts | 2 | 2 | Multi-stage terrain filter with several host-side prerequisites |

The scan found 260 parseable HSD files. The lower-frequency values are still
important when they create a large boundary, but frequency alone is not the
decision rule.

## Ranking

The ranking weighs frequency, distinct real graphs, chain disruption, transfer
and compute cost, existing implementation, numerical/tile risk, and the
amount of new backend surface required.

| Rank | Candidate | Frequency/impact | Existing building blocks | Risk/complexity | Decision |
|---:|---|---|---|---|---|
| 1 | MorphologicalGradient | Four real graphs; radius-based neighborhood output often feeds a visual or terrain branch | CPU and OpenCL implementations are direct and well-defined; one Metal neighborhood kernel is sufficient | Low-to-medium; halo-required and single-tile resident path | **Select** |
| 2 | HydraulicStreamLog | 14 graphs and potentially high per-node cost; multiple output maps disrupt chains | OpenCL implementation exists, but no matching resident Metal API | Very high: flow accumulation, gradient filtering, deposition, three auxiliary outputs, remaps, and host post-process | Defer |
| 3 | Vorolines/Rifts-related | A few graphs, but meaningful terrain use where present | OpenCL kernels exist; no reusable Metal implementation | Medium-to-high numerical and geometry risk | Defer |
| 4 | FloodingUniformLevel | 12 graphs | No direct resident primitive | Global water semantics and output coupling do not fit this phase | Defer |
| 5 | Remaining NoiseFbm groups | 23 labels, but the current HSD loader maps the legacy FBM label to CoherentNoise; compatible FBM is already resident | Existing Metal FBM path is qualified | Non-FBM groups are materially different algorithms; porting them would be a new noise campaign | Defer |

`MorphologicalGradient` wins the maintainability-adjusted ranking, not raw
frequency. HydraulicStreamLog is the runner-up because its raw corpus impact is
higher, but its dependency tree would exceed the one-family Phase 7 cap and
would require a new global/iterative execution subsystem.

## Scope boundary

The selected scope is one HighMap `morphological_gradient` DeviceSession and
synchronous wrapper, one Metal kernel, its no-Metal stub, parity/benchmark
coverage, and one Hesiod eligibility bridge. No scheduler, renderer, hydraulic
pipeline, or unrelated filter family is included.
