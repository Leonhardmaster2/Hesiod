# Phase 2 Progress — GPU-Resident Pipeline

## Current Phase: C (Vulkan Renderer)
## Current Step: COMPLETE
## Last Completed: C.7 — Build and validate
## Blockers: none
## Last Updated: 2026-03-12

## Phase A Status

### Completed Steps

**A.1: GpuBuffer class** — DONE
- File: `VkCompute/include/vk_compute/gpu_buffer.hpp`
- File: `VkCompute/src/gpu_buffer.cpp`
- Features: VkBuffer + VmaAllocation wrapper, dirty tracking, lazy staging buffer,
  upload/download with force flag, move-only semantics, thread-safe (mutex)

**A.2: Run::bind_buffer(id, GpuBuffer&)** — DONE
- Modified: `VkCompute/include/vk_compute/run.hpp`
- Modified: `VkCompute/src/run.cpp`
- Features: Zero-copy binding (no staging buffer allocated), GpuBuffer-backed
  BufferEntry with gpu_buffer_ref pointer, Run destructor skips destroying
  GpuBuffer-owned buffers, execute() marks GpuBuffers as gpu_dirty

**A.3: GpuArray class** — DONE
- File: `HighMap/include/highmap/vulkan/gpu_array.hpp`
- File: `HighMap/src/gpu_vulkan/gpu_array.cpp`
- Features: Vec2<int> shape + GpuBuffer, from_array/to_array conversion,
  upload_from/download_to for existing Arrays

**A.3b: GpuBufferRegistry** — DONE
- File: `VkCompute/include/vk_compute/gpu_buffer_registry.hpp`
- File: `VkCompute/src/gpu_buffer_registry.cpp`
- Features: Global thread-safe map (void* → GpuBuffer*), allows GPU functions
  to auto-detect GpuBuffer-backed Arrays without signature changes

**A.4: GPU function modifications** — DONE (thermal family)
- Modified: `HighMap/src/erosion/thermal_gpu.cpp`
- All thermal variants use helper_bind_array() for auto-detection
- Helper added to `gpu_vulkan.hpp`: helper_bind_array(run, id, array)
- Pattern: check registry → bind GpuBuffer directly (skip write/read) or
  fall back to legacy path
- extrapolate_borders: download → CPU modify → re-upload when GPU-resident

**A.5: Benchmark pipeline test** — DONE
- File: `tests/benchmark_pipeline/main.cpp`
- Tests: GpuBuffer roundtrip, thermal correctness (legacy vs zero-copy),
  single thermal benchmark (legacy vs zero-copy) at 256-4096,
  3-node thermal chain benchmark

**A.6: CMake changes** — DONE (none needed)
- All CMakeLists use GLOB_RECURSE, new files auto-discovered

### A.7: Build and validate — DONE
- Full build succeeds (Hesiod + HighMap + VkCompute + benchmark_pipeline)
- Regression: 48/77 pass (same as Phase 1, no regressions)
- GpuBuffer roundtrip: PASS
- Thermal correctness: PASS (max_diff=0.00029)
- Benchmark results (RTX 5080):
  - Laplace 5-call chain 1024x1024: legacy=334ms, zerocopy=185ms (1.81x speedup)
  - Laplace 5-call chain 4096x4096: legacy=7420ms, zerocopy=4725ms (1.57x speedup)
  - Thermal chains: slower due to extrapolate_borders CPU roundtrip (expected)
  - Key insight: zero-copy benefits pure GPU chains; functions with CPU post-processing
    need extrapolate_borders on GPU (future optimization)

## Architecture Summary

```
BEFORE (per GPU function call):
  CPU array → [allocate staging+device] → [memcpy→staging→device] → compute → [device→staging→memcpy] → CPU array
  (Run destructor frees staging+device)

AFTER (with GpuBuffer registered):
  CPU array → [GpuBuffer persists] → compute (bind existing VkBuffer) → [GpuBuffer stays resident]
  Only download when CPU function (extrapolate_borders) or final export needs it
```

### Key Files Created/Modified
| File | Status | Purpose |
|------|--------|---------|
| `VkCompute/include/vk_compute/gpu_buffer.hpp` | NEW | GpuBuffer class |
| `VkCompute/src/gpu_buffer.cpp` | NEW | GpuBuffer implementation |
| `VkCompute/include/vk_compute/gpu_buffer_registry.hpp` | NEW | Global buffer registry |
| `VkCompute/src/gpu_buffer_registry.cpp` | NEW | Registry implementation |
| `VkCompute/include/vk_compute/vk_compute.hpp` | MODIFIED | Added new includes |
| `VkCompute/include/vk_compute/run.hpp` | MODIFIED | GpuBuffer bind_buffer overload |
| `VkCompute/src/run.cpp` | MODIFIED | GpuBuffer binding + mark dirty |
| `HighMap/include/highmap/vulkan/gpu_array.hpp` | NEW | GPU-resident Array |
| `HighMap/src/gpu_vulkan/gpu_array.cpp` | NEW | GpuArray implementation |
| `HighMap/include/highmap/vulkan/gpu_vulkan.hpp` | MODIFIED | helper_bind_array |
| `HighMap/src/erosion/thermal_gpu.cpp` | MODIFIED | Zero-copy thermal functions |
| `tests/benchmark_pipeline/main.cpp` | NEW | Benchmark program |
| `tests/benchmark_pipeline/CMakeLists.txt` | NEW | Benchmark CMake |

## Phase B Status

### Completed Steps

**B.1: Analyze GNode execution model** -- DONE
- Graph::update() marks all nodes dirty, topological sorts, executes sequentially
- Graph::update(node_id) finds downstream dirty nodes, executes sequentially
- topological_sort() uses Kahn's algorithm -- naturally identifies wavefronts

**B.2: Wavefront parallel scheduler** -- DONE
- Added `update_parallel()` to `GNode/include/gnode/graph.hpp`
- Implemented in `GNode/src/graph.cpp`
- Wavefront Kahn's algorithm: groups nodes into waves by dependency level
- Nodes within a wave execute concurrently via `std::async`
- Single-node waves skip thread overhead
- Callbacks fire sequentially (before all, parallel compute, after all)
- Wave timing logged via spdlog

**B.3: Thread safety for VkCompute** -- DONE
- PipelineManager: added `std::recursive_mutex` protecting shader_modules,
  pipeline_cache, descriptor set alloc/free
- DeviceManager: added `command_pool_mutex()` for VkCommandPool alloc/free,
  added `queue_mutex()` for VkQueueSubmit serialization
- Run::copy_buffer, Run::submit_and_wait, Run::dispatch_compute: wrapped
  command buffer alloc/free and queue submit in lock_guards
- GpuBuffer::copy_buffer: same treatment
- VMA allocator is thread-safe by default (internal mutexes)
- GpuBuffer already has per-instance mutex

**B.3b: GraphNode wiring** -- DONE
- `set_parallel_update(true)` in GraphNode constructor
- Graph::update() and update(node_id) check parallel_update_ flag
- Falls through to update_parallel() when enabled

### Key Files Modified
| File | Status | Purpose |
|------|--------|---------|
| `GNode/include/gnode/graph.hpp` | MODIFIED | update_parallel(), set_parallel_update() |
| `GNode/src/graph.cpp` | MODIFIED | Wavefront scheduler implementation |
| `VkCompute/include/vk_compute/pipeline_manager.hpp` | MODIFIED | recursive_mutex |
| `VkCompute/src/pipeline_manager.cpp` | MODIFIED | Lock guards on all methods |
| `VkCompute/include/vk_compute/device_manager.hpp` | MODIFIED | cmd_pool_mutex_, queue_mutex_ |
| `VkCompute/src/run.cpp` | MODIFIED | Thread-safe cmd alloc/free/submit |
| `VkCompute/src/gpu_buffer.cpp` | MODIFIED | Thread-safe cmd alloc/free/submit |
| `Hesiod/src/model/graph/graph_node.cpp` | MODIFIED | Enable parallel update |

## Phase C Status

### Completed Steps

**C.1: Analyze existing renderer** -- DONE
- OpenGL 3.3 renderer: shadow mapping, AO, fog, water, instanced meshes
- 6 shader pairs (shadow_map_lit, depth, viewer2d, etc.)
- Integration via Viewer3D::update_renderer() with set_heightmap_geometry()

**C.2: DeviceManager graphics support** -- DONE
- Added VK_KHR_surface + VK_KHR_win32_surface instance extensions
- Added VK_KHR_swapchain device extension
- Queue family selection: prefer graphics+compute combined (NVIDIA family 0)
- New methods: create_surface(), destroy_surface(), queue_supports_present()
- Gated behind `VKCOMPUTE_ENABLE_GRAPHICS` compile define

**C.3: Vulkan GLSL 460 shaders** -- DONE
- terrain.vert: Reads heightmap from SSBO (zero-copy), generates position/normal/UV
- terrain.frag: Blinn-Phong lighting, PCF shadows, HBAO, albedo/normal textures
- shadow_depth.vert/frag: Depth-only pass from light POV
- viewer2d.vert/frag: Fullscreen triangle with colormaps (gray/viridis/turbo/magma)

**C.4: VkTerrainRenderer library** -- DONE
- New library: external/VkTerrainRenderer/
- RenderWidget: QWidget subclass with Vulkan surface, swapchain, render pass
- Swapchain: FIFO present mode, dynamic resize handling
- Render pass: color + D32 depth, single subpass
- Graphics pipeline: dynamic viewport/scissor, no vertex input (SSBO-driven)
- Command buffers: per-swapchain-image, reset each frame
- Sync: semaphores for acquire/present, per-image fences

**C.5: Descriptor sets and UBOs** -- DONE
- Set 0: FrameUBO (matrices + camera + light) + HeightmapSSBO + MaterialUBO
- Set 1: 4 combined image samplers (albedo, hmap, normal, shadow_map)
- UBOs: persistently mapped host-visible buffers, updated every frame
- SSBO: device-local, upload via staging buffer (legacy) or direct bind (zero-copy)

**C.6: Hesiod Viewer3D integration** -- DONE
- viewer_3d.hpp: conditional include (vktr vs qtr) via HESIOD_VULKAN_RENDERER
- RendererWidget typedef: maps to vktr::RenderWidget or qtr::RenderWidget
- Same public API: set_heightmap_geometry, set_texture, clear, json_from/to
- New zero-copy method: set_heightmap_buffer(GpuBuffer*, w, h)

**C.7: Build and validate** -- DONE
- Fixed windows.h include issue: `vulkan_win32.h` moved out of public header
  to avoid IN/OUT macro conflicts with GNode PortType enum
- Added NOMINMAX + WIN32_LEAN_AND_MEAN guards in device_manager.cpp
- Full build succeeds: hesiod.exe + vkterrain-renderer.lib + all 6 SPIR-V shaders
- Regression: 48 OK / 29 NOK (same as Phase 1 baseline, no regressions)
- VkCompute init test: PASS (graphics support enabled, queue family 0 GRAPHICS+COMPUTE)

### Key Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `VkCompute/include/vk_compute/device_manager.hpp` | MODIFIED | Graphics extensions, create_surface() |
| `VkCompute/src/device_manager.cpp` | MODIFIED | Surface/swapchain, queue selection |
| `VkCompute/VkCompute/CMakeLists.txt` | MODIFIED | VKCOMPUTE_ENABLE_GRAPHICS define |
| `VkTerrainRenderer/CMakeLists.txt` | NEW | Root CMake |
| `VkTerrainRenderer/VkTerrainRenderer/CMakeLists.txt` | NEW | Library + shader compilation |
| `VkTerrainRenderer/.../include/vktr/render_widget.hpp` | NEW | Vulkan renderer widget |
| `VkTerrainRenderer/.../src/render_widget.cpp` | NEW | Full Vulkan renderer impl (~1400 lines) |
| `VkTerrainRenderer/.../include/vktr/shaders/terrain.vert` | NEW | SSBO heightmap vertex shader |
| `VkTerrainRenderer/.../include/vktr/shaders/terrain.frag` | NEW | Lit terrain fragment shader |
| `VkTerrainRenderer/.../include/vktr/shaders/shadow_depth.vert` | NEW | Shadow depth pass |
| `VkTerrainRenderer/.../include/vktr/shaders/shadow_depth.frag` | NEW | Shadow depth pass |
| `VkTerrainRenderer/.../include/vktr/shaders/viewer2d.vert` | NEW | 2D colormap vertex |
| `VkTerrainRenderer/.../include/vktr/shaders/viewer2d.frag` | NEW | 2D colormap fragment |
| `external/CMakeLists.txt` | MODIFIED | add_subdirectory(VkTerrainRenderer) |
| `Hesiod/CMakeLists.txt` | MODIFIED | Link vkterrain-renderer, HESIOD_VULKAN_RENDERER |
| `Hesiod/include/.../viewer_3d.hpp` | MODIFIED | Conditional vktr/qtr include |
| `Hesiod/src/.../viewer_3d.cpp` | MODIFIED | Use RendererWidget typedef |
