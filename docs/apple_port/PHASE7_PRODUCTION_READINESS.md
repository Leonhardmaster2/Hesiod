# Phase 7 Production Readiness Audit

## Runtime and packaging

The HighMap CMake path detects Metal by framework headers/library rather than
by Mac model name. On Apple release builds with `xcrun metal`, `xcrun
metallib`, and `xxd`, it compiles `src/gpu_metal/highmap.metal` to AIR,
links a `highmap.metallib`, embeds it in `metal_library.hpp`, and links the
Metal/Foundation frameworks. When the precompiled toolchain is unavailable,
the configured runtime-source path embeds `metal_shader_source.hpp` and
compiles the MSL once during Metal context initialization. A build with
`HIGHMAP_METAL_RUNTIME_COMPILE=OFF` fails configuration unless the precompiled
path is available, avoiding a silent release without shaders.

The current machine's Command Line Tools expose `xcrun` but not the `metal` or
`metallib` utilities, so this task verified the packaging logic and the
runtime-source build, not a finished precompiled metallib artifact or signed
Hesiod application bundle. A release CI/macOS SDK image must exercise the
precompiled branch before shipping.

The first-use path was exercised by the focused
`MetalBackend.MorphologicalGradientMatchesOpenCL` test. It initialized the
Metal library and pipeline successfully. Warm benchmark samples reported no
pipeline creation; pipeline setup is cached by function name after first use.

## Diagnostics and fallback behavior

The graph bridge now reports node-level backend records through the Phase 4
diagnostic output. The selected node distinguishes Metal unavailable, closed
session, missing ports, multi-tile halo fallback, and unsupported resident
state. Existing host/OpenCL implementations remain authoritative for all
fallbacks. The session lifecycle guard prevents a later resident candidate from
encoding into a command buffer already closed by a host materialization.

Persistent caching remains explicit opt-in through
`HESIOD_METAL_PERSISTENT_CACHE=1`. The cache is scoped to the graph bridge,
stores only completed HighMap DeviceArrays, validates shape/tile/halo metadata,
and evicts by least-recent use under a byte budget. No HSD serialization, raw
Metal object, renderer, or universal scheduler was added.

## API and platform audit

HighMap's public addition is the existing Metal GPU namespace's
`DeviceSession::morphological_gradient` plus the matching synchronous
`gpu::metal::morphological_gradient` wrapper. The implementation, MSL kernel,
stub, and parity test are self-contained in HighMap and do not include Hesiod
or Qt. The Hesiod side only adds an optional node eligibility bridge and CLI
diagnostics/soak hooks.

Metal source is gated by `HIGHMAP_HAS_METAL`; the no-Metal stub preserves the
same API surface and throws a clear capability error when called. Non-Apple
builds therefore do not require Objective-C++, Metal headers, or Metal
frameworks. Windows/Linux CI still needs to compile the source and stub on
their native runners; this macOS task provides structural review, not native
cross-platform execution.

The existing QTerrainRenderer and renderer resource ownership are untouched.
CPU, OpenCL, and no-Metal paths remain first-class and are not replaced by the
new operation.

## Release recommendation

The targeted operation is suitable for continued controlled use on one
logical tile with cache off. The full application is not yet a blanket
default-on Metal claim: the selected node is deprecated, several post-process
forms remain host-only, multi-tile execution remains fallback-only, and a
precompiled-metallib release build still needs CI verification. The next
engineering step is upstream/cleanup/release preparation, not a universal
residency rollout.
