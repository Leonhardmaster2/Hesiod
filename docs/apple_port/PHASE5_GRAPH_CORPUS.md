# Phase 5: Built-In Graph Corpus

The corpus scan covered 260 parseable files in `data/bootstraps/` and
`data/examples/`, plus `data/default.hsd`, for 762 serialized nodes and 250
node types. Average position is the serialized node index divided by graph
node count; it is a coarse location indicator, not a runtime critical-path
measure.

## Most frequent node families

| Node type | Count | Average position |
|---|---:|---:|
| GaborWaveFbm | 178 | 0.098 |
| ColorizeGradient | 46 | 0.445 |
| Preview | 34 | 0.392 |
| NoiseFbm | 23 | 0.333 |
| CloudLattice | 19 | 0.026 |
| HydraulicStreamLog | 15 | 0.274 |
| CloudToPath | 14 | 0.329 |
| FloodingUniformLevel | 13 | 0.387 |
| MixTexture | 12 | 0.403 |
| Blend | 10 | 0.497 |
| SelectSoilWeathered | 9 | 0.278 |
| SmoothFill | 6 | 0.304 |

## Boundary evidence

The most frequent serialized edges were:

| Count | Edge |
|---:|---|
| 24 | ColorizeGradient → MixTexture |
| 16 | GaborWaveFbm → ColorizeGradient |
| 11 | SelectSoilWeathered → ColorizeGradient |
| 11 | CloudLattice → CloudToPath |
| 10 | GaborWaveFbm → HydraulicStreamLog |
| 8 | CloudToVectors → ExportPointsToPly |
| 8 | HydraulicStreamLog → FloodingUniformLevel |
| 8 | HydraulicStreamLog → Preview |
| 7 | MixTexture → Preview |
| 7 | FloodingUniformLevel → Preview |

Among the currently relevant resident/fallback families, NoiseFbm is the
clearest repeated compute boundary: it feeds VorolinesFbm and Rifts four
times each, ColorizeGradient three times, CloudSetValuesFromHeightmap three
times, and several other CPU-oriented nodes. The selected NoiseFbm →
SpectralEqualizer/Thermal → Blend chain occurs once and is now fully resident.

## Transparent residency-break score

For prioritization, this phase uses:

`score = occurrence × chain disruption × estimated transfer/working-set cost ÷ implementation complexity`

Each component is a coarse ordinal from 1 to 5. Occurrence comes from the
serialized corpus; disruption counts downstream resident-capable consumers;
transfer cost estimates full-array bytes; complexity is a review estimate.
The score is not a performance prediction.

| Rank | Candidate | Evidence | Phase 5 outcome |
|---:|---|---|---|
| 1 | NoiseFbm / CoherentNoise FBM | 23 nodes; repeated downstream boundaries; existing base Metal noise | ported for compatible FBM |
| 2 | SpectralEqualizer | dominant selected-graph host cost; five blur temporaries | ported resident |
| 3 | GaborWaveFbm | 178 nodes; highest frequency | not ported; algorithm audit required |
| 4 | HydraulicStreamLog | 15 nodes and preview/flooding chains | existing Phase 3/4 hydraulic work is separate |
| 5 | ColorizeGradient / MixTexture | common but mostly visual/color branch traffic | renderer/color residency deferred |

The corpus does not justify a universal scheduler in this phase. It does
justify targeted shared numerical primitives and continued audits of
GaborWaveFbm, NoiseFbm's non-FBM families, and hydraulic downstream nodes.
