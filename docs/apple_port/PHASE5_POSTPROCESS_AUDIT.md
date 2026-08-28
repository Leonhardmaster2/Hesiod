# Phase 5: Generic Post-Processing Audit

The corpus contains 762 nodes across 260 parseable `.hsd` files. The audit
counts serialized attributes, so absent attributes are not treated as an
active operation.

| Operation | Serialized count | Observed distribution | Phase 5 decision |
|---|---:|---|---|
| inverse | 169 | 169 false | no new kernel; identity is free |
| gamma | 30 | all 1.0 | no new kernel; identity is free |
| gain | 169 | 165 at 1.0; 4 non-identity | defer; semantics include range preservation |
| smoothing radius | 169 | 168 at 0; 1 nonzero | existing resident primitive, but generic tiled overlap remains fallback |
| remap | 169 | 129 active; 116 active `[0,1]`, 13 active non-identity | resident normalize used for CoherentNoise |
| saturate | 95 | 7 active non-identity; 88 inactive | defer; percentile/range semantics need reductions |
| mix | 47 | all 1.0 | existing resident linear combine is sufficient for selected Blend |

The active `[0,1]` remap is not assumed to be a no-op: it requests automatic
normalization in the existing `VirtualArray::remap` behavior. The resident
CoherentNoise path therefore computes the source range and maps it explicitly.

## Why the remaining operations stay conservative

`inverse`, `gain`, and `gamma` are pointwise in isolation, but the current
post-processing implementation preserves the input range around gain/gamma.
`saturate` computes dynamic range values and can use percentile behavior.
Generic smoothing also has VirtualArray tile-overlap behavior that is not
equivalent to a single flat-buffer pass for all configurations. Adding a
parameterized fused post-process shader without first defining these
contracts would risk silent numerical changes.

The selected graph has no active generic operation beyond CoherentNoise's
range remap. That remap is now resident, while unsupported nontrivial generic
post-processing continues to force the established host boundary. No shader
compiler, expression language, or universal scheduler was introduced.
