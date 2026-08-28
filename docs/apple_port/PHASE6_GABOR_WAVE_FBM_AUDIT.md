# Phase 6 GaborWaveFbm Audit

## Corpus signal

The parseable corpus contains 260 `.hsd` files: 253 examples, six bootstrap
files, and the default project. A graph-node-label scan finds 178
`GaborWaveFbm` occurrences in examples/graphs, versus 23 `NoiseFbm`
occurrences. The scan intentionally counts graph-manager node labels only and
does not count duplicate UI captions.

GaborWaveFbm is therefore the highest-frequency procedural family in the
current built-in corpus. It is also a frequent first-stage source for
downstream host nodes, making it a high-value residency boundary even when a
later node remains a fallback.

## Existing semantics

The established OpenCL implementation evaluates the exact Gabor-wavelet fBm
algorithm: a 5×5 neighborhood, polynomial hash, direction jitter, weighted
octaves, optional control multiplier, optional X/Y displacement, and a
per-pixel angle. Its position mapping uses the existing half-wavenumber
convention (`0.5 * kx`, `0.5 * ky`) and the VirtualArray region bounding box.

Hesiod's scalar angle attribute is in degrees. The optional angle port is a
radian displacement converted to degrees before the OpenCL call. The Metal
API keeps that contract explicit: `angle_degrees` is the scalar base and the
optional DeviceArray angle is converted from radians in the Metal kernel.

## Port decision

Phase 6 adds one major family: `GaborWaveFbm`. HighMap provides a synchronous
wrapper and a `DeviceSession::gabor_wave_fbm` operation. The Hesiod node uses
the operation when Metal is enabled, the envelope is disconnected, and the
post-process is compatible. Active range remap is retained in-resident through
the existing Metal normalize reduction.

The implementation was checked against OpenCL with control, X/Y displacement,
per-pixel angle, non-square dimensions, non-default bbox, six octaves, and the
same seed/weights. The focused parity test passes at a 3e-4 maximum absolute
error threshold. The initial implementation also caught and corrected a
half-wavenumber mismatch before integration.

## Eligibility and fallback

Resident GaborWaveFbm is selected for:

- no envelope;
- no inverse, gamma, gain, smoothing, or saturation post-process;
- optional active range remap only, handled by Metal normalize;
- any VirtualArray tile grid, because the operation evaluates the complete
  logical array once and writes it back through the existing `from_array`
  contract.

Envelope application, nontrivial post-processing, and unsupported backend
configurations use the original tiled OpenCL/CPU path. The fallback explicitly
materializes any resident input before ordinary tile iteration.

This is intentionally not a claim that Gabor is a general halo-aware tiled
Metal kernel. The Phase 6 tiled path is whole-logical-array resident and uses
VirtualArray's established host-side gather/scatter semantics at the boundary.
