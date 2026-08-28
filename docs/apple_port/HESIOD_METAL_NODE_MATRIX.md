# Hesiod Phase 4 Metal Node Matrix

| Node or boundary | Phase 4 path | Resident eligibility / fallback |
|---|---|---|
| CoherentNoise | Host fallback | Existing CPU/OpenCL path; no resident implementation in this phase |
| SpectralEqualizer | Host fallback | Existing composed/tiled implementation remains authoritative |
| Thermal | Resident Metal or host fallback | Resident only for unmasked Standard/Linear, no deposition, no scale-talus, identity post-process; all other forms fall back |
| Blend | Resident Metal or host fallback | Resident only for ADD, no swap, identity post-process; all other methods fall back |
| Generic post-process | Host boundary unless identity-gated | Resident nodes opt out when post-processing would change semantics |
| DataPreview | Host boundary | Materializes the requested VirtualArray for preview |
| Flatten/export | Host boundary | Existing export code consumes host VirtualArrays |
| Scalars and non-VirtualArray ports | Existing path | Not part of the resident bridge |

Thermal's resident Linear form uses the Metal thermal and thermal-ridge
primitives and preserves the legacy border extrapolation after each half. The
resident ADD blend uses `DeviceSession::linear_combine`. The fallback path is
still selected for unsupported parameters, unavailable Metal, or
`HESIOD_METAL_RESIDENT=0`.
