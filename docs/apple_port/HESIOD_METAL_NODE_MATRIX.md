# Hesiod Apple Metal Node Matrix

| Node or boundary | Phase 4 path | Resident eligibility / fallback |
|---|---|---|
| CoherentNoise / HSD `NoiseFbm` | Resident Metal or host fallback | Resident FBM only for Metal-supported noise, no envelope, single tile, and identity post-processing except active range remap; all other groups/configurations fall back |
| GaborWaveFbm | Resident Metal or host fallback | Resident whole-logical-array Gabor fBm when no envelope and no nontrivial post-process; active range remap uses the resident normalize reduction; other cases use the original tiled path |
| SpectralEqualizer | Resident Metal or host fallback | Resident no-mask, single-tile blur pyramid and band rebuild; nontrivial post-processing, masks, and tiled configurations fall back |
| Thermal | Resident Metal or host fallback | Resident only for unmasked Standard/Linear, no deposition, no scale-talus, identity post-process; all other forms fall back |
| Blend | Resident Metal or host fallback | Resident only for ADD, no swap, identity post-process; all other methods fall back |
| MorphologicalGradient | Resident Metal core or host fallback | Single-tile input/output uses the local max-minus-local min Metal kernel; active range remap stays resident, while legacy host-only post-process is an explicit materialize-and-host boundary; tiled, unavailable, or closed-session cases use the original OpenCL/CPU path |
| Generic post-process | Host boundary unless identity-gated | Selected FBM range remap uses resident min/max + normalize; inverse, gain, gamma, saturate, generic mix, and tiled smoothing remain conservative host boundaries |
| DataPreview | Host boundary | Materializes the requested VirtualArray for preview |
| Flatten/export | Host boundary | Existing export code consumes host VirtualArrays |
| Scalars and non-VirtualArray ports | Existing path | Not part of the resident bridge |

Thermal's resident Linear form uses the Metal thermal and thermal-ridge
primitives and preserves the legacy border extrapolation after each half. The
resident ADD blend uses `DeviceSession::linear_combine`. The fallback path is
still selected for unsupported parameters, unavailable Metal, or
`HESIOD_METAL_RESIDENT=0`.

Phase 5 adds HighMap `DeviceSession::noise_fbm`, `smooth_cpulse`,
`spectral_equalizer`, and `normalize`. The selected
`SpectralEqualizer.hsd` graph therefore runs four compute nodes in one session,
with no host compute nodes, no uploads, and one final readback. The resident
graph policy remains explicit and capability-based; it is not a universal
scheduler.

Phase 6 adds the GaborWaveFbm entry after a corpus audit found it in 178 graph
nodes across the parseable HSD corpus. Its tiled configuration is deliberately
qualified as a whole-logical-array operation: VirtualArray gathers the logical
array once and scatters it back through its existing tile/halo mapping. The
downstream boundary and stitching nodes remain host boundaries. An optional,
graph-scoped completed-DeviceArray cache is off by default and does not change
any eligibility rule.

Phase 7 adds the bounded `MorphologicalGradient` DeviceSession operation and
the `texturing_101.hsd` Graph D integration. The node is deprecated in the UI
but remains present in four real graphs, making it a useful compatibility
workload. The kernel is halo-required; Phase 7 deliberately qualifies only a
single logical tile and leaves tiled execution on the established fallback
path. `meander.hsd` is also used to verify that a selected node after a host
materialize is reported as a closed-session fallback rather than raising a
session-lifecycle error.
