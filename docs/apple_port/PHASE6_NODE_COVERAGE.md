# Phase 6 Node Coverage and Corpus Selection

## Corpus

The scan covered parseable HSD JSON under `Hesiod/data`: 260 files, including
253 examples and six bootstraps. Graph-node labels were counted from
`graph_manager.graph_nodes[*].nodes[*]`; UI captions and repeated JSON
presentation copies were excluded.

The most relevant counts are:

| Node label | Occurrences |
|---|---:|
| GaborWaveFbm | 178 |
| NoiseFbm | 23 |
| Blend | 10 |
| SpectralEqualizer | 1 |
| Thermal | 2 |

The frequency result is why Gabor was selected as the one major Phase 6
family. It occurs in many small built-in examples as a substrate, so a
resident implementation also tests the common case where the next node is
still a host boundary.

## Selected graphs

| Role | Built-in graph | Purpose |
|---|---|---|
| Graph A | `data/examples/SpectralEqualizer.hsd` | Phase 5 four-node full-residency control |
| Graph B | `data/examples/MakePeriodic.hsd` | GaborWaveFbm plus a real periodic boundary consumer |
| Graph C | `data/examples/MakePeriodicStitching.hsd` | GaborWaveFbm plus stitching boundary under tiled overlap |

Graph B and Graph C are intentionally built-in files. Both contain a
configured GaborWaveFbm node and a downstream boundary operation; the latter
exercises the explicit overlap/stitching contract instead of a synthetic
graph.

## Coverage result

Graph A retains four resident Metal nodes and one terminal readback at the
qualified single-tile configuration. Graph B and Graph C residentize the
Gabor source and leave the downstream boundary operation on its existing host
path. This makes the host readback/materialization visible and keeps the
fallback behavior auditable.

No universal graph scheduler, renderer rewrite, or automatic residency planner
was introduced.
