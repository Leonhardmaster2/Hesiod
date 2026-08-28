# Phase 7 MorphologicalGradient Audit

## Hesiod node

Implementation: `src/model/nodes/nodes_function/morphological_gradient.cpp`.

The node has one heightmap input and one heightmap output. Its floating-point
`radius` is converted to a pixel radius with
`max(1, int(radius * output_width))`. The existing compute path evaluates each
virtual-array tile with `hmap::gpu::morphological_gradient`, smooths overlapping
tiles, and applies the normal Hesiod post-process contract.

The node is marked deprecated in the user-facing implementation, but it is
still present in four real shipped graphs and is used as a heightmap signal
before color/texture branches. It remains a valid compatibility workload for a
targeted backend path.

## Numerical implementation

The CPU implementation is `HighMap/src/morphology/morphology.cpp` and computes:

```text
vmin = global minimum(input)
dilation(input - vmin, radius) - erosion(input - vmin, radius)
```

The GPU/OpenCL implementation is `HighMap/src/morphology/morphology_gpu.cpp`.
It calls `gpu::local_max` and `gpu::local_min`, backed by the `local_max` and
`local_min` OpenCL kernels. Each output pixel scans the square bounding box
around the pixel and accepts samples inside the Euclidean radius. Addressing
clamps at the array edge. The global minimum shift cancels in the subtraction,
so the resident Metal implementation can directly emit local maximum minus
local minimum without changing the result.

There are no iterations, reductions, random state, masks, optional outputs,
or global coupling in the core operation. The only temporary state in the
resident implementation is the output DeviceArray; extrema are accumulated in
thread-local registers.

## Tiling and halo contract

This is a `HALO_REQUIRED` family. A radius-`r` output reads neighboring pixels
within `r`, so an independently evaluated tile needs at least `r` pixels of
valid halo and overlap blending must follow the existing VirtualArray contract.
Phase 7 uses the conservative and already-established resident policy:

* a single logical array may use the Metal DeviceArray path;
* multi-tile configurations remain on the original OpenCL/CPU tiled path;
* no implicit seam correction or generic halo scheduler is introduced.

The fallback remains authoritative for tiled and unsupported post-process
configurations.

## Post-processing and eligibility

The Metal core is eligible for a connected input/output on a single logical
tile. An active range remap uses the established DeviceSession normalize
reduction. Identity post-processing keeps the result resident for a following
resident-capable node or the final readback.

Legacy post-processing that cannot run in the resident backend (inverse, gamma,
non-identity gain, smoothing, saturation, or mix) is an explicit boundary:
the neighborhood core runs on Metal, the result is materialized once, and the
existing `post_process_heightmap` code continues on the host. The node is
reported as `resident Metal` with detail
`morphological_gradient + host post-process`, so the core work is measurable
without claiming that host-only post-processing is resident.

Multi-tile arrays and unavailable/closed Metal sessions use the unchanged
OpenCL/CPU implementation and record the explicit fallback reason. This
conservative rule avoids introducing a general halo scheduler in Phase 7.

The real `texturing_101` bootstrap therefore qualifies the Metal neighborhood
core and deliberately retains the surrounding host boundaries; its fallback
and resident outputs are parity-checked end to end. The `meander` bootstrap is
also covered as a lifecycle case: because an earlier host node closes the
shared session before this node, it takes the explicit closed-session fallback
without raising a node error.

## Backend comparison

CPU is the portable scalar/reference implementation. OpenCL is the existing
portable GPU implementation and remains the tiled/default path. Metal is
appropriate for the compatible single-tile case because the operation is a
regular, massively parallel neighborhood scan and can avoid a host round trip
when its output stays in a resident chain. The Metal implementation does not
replace CPU/OpenCL and is not selected for tiled arrays without a separately
qualified halo contract.

## Complexity and memory

The direct scan is O(N·r²) for an N-pixel array, matching the existing OpenCL
algorithmic shape. It allocates one output buffer in the DeviceSession and
does not allocate per-pixel auxiliary arrays. The implementation is therefore
bounded to one major family and does not require a new scheduler or resource
manager.
