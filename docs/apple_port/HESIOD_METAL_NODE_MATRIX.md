# Hesiod Phase 5 Metal Node Matrix

| Node or boundary | Phase 4 path | Resident eligibility / fallback |
|---|---|---|
| CoherentNoise / HSD `NoiseFbm` | Resident Metal or host fallback | Resident FBM only for Metal-supported noise, no envelope, single tile, and identity post-processing except active range remap; all other groups/configurations fall back |
| SpectralEqualizer | Resident Metal or host fallback | Resident no-mask, single-tile blur pyramid and band rebuild; nontrivial post-processing, masks, and tiled configurations fall back |
| Thermal | Resident Metal or host fallback | Resident only for unmasked Standard/Linear, no deposition, no scale-talus, identity post-process; all other forms fall back |
| Blend | Resident Metal or host fallback | Resident only for ADD, no swap, identity post-process; all other methods fall back |
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
