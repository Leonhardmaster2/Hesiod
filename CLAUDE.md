# Autonomous OpenCL → Vulkan Compute Pipeline Conversion

## Context

You are converting the **Hesiod** terrain engine (https://github.com/otto-link/Hesiod) and its compute library **HighMap** (https://github.com/otto-link/HighMap) from an OpenCL compute pipeline to a fully optimized **Vulkan compute shader** pipeline. The rendering component (**QTerrainRenderer**) uses OpenGL and is out of scope for now.

### Current Architecture

```
Hesiod (Qt6 GUI, 281 node functions)
  └── HighMap (compute library)
       ├── 46 OpenCL kernels (.cl files in HighMap/src/gpu_opencl/kernels/)
       ├── CLWrapper (OpenCL abstraction: DeviceManager, KernelManager, Run)
       ├── ~24 *_gpu.cpp files that call CLWrapper
       └── ~146 CPU-only .cpp files (some should stay CPU, some should get GPU paths)
```

### Target Architecture

```
Hesiod (Qt6 GUI, 281 node functions — unchanged interface)
  └── HighMap (compute library)
       ├── 46+ GLSL compute shaders (.comp in HighMap/src/gpu_vulkan/shaders/)
       ├── VkCompute (new Vulkan abstraction replacing CLWrapper)
       ├── ~24+ *_gpu.cpp files rewritten to use VkCompute
       ├── New GPU paths for CPU functions that benefit from parallelism
       └── CPU-only functions preserved where GPU would be slower
```

---

## PHASE 0 — Planning (Gemini CLI)

Before writing any code, use Gemini to plan the architecture. Run this:

```bash
gemini -p "You are a senior graphics engineer. I'm converting a terrain engine from OpenCL to Vulkan compute shaders.

Current setup:
- CLWrapper: thin OpenCL abstraction with DeviceManager (picks GPU), KernelManager (compiles .cl at runtime via raw string includes), and Run (bind_buffer, bind_arguments, write_buffer, execute, read_buffer)
- 46 OpenCL .cl kernels embedded as raw strings, compiled at init
- GPU functions follow pattern: create Run, bind buffers/args, write, execute, read back
- All buffers are std::vector<float> on host, transferred per-dispatch

I need you to design:
1. A VkCompute wrapper class hierarchy that mirrors CLWrapper's simplicity but uses Vulkan compute. Must handle: device selection, queue creation, descriptor set management, pipeline caching, buffer management (VkBuffer + VMA), shader module loading (SPIR-V), command buffer recording + submission, synchronization. The API surface should be similar: bind_buffer, bind_arguments, execute, read_buffer.
2. A shader compilation strategy: should I precompile GLSL→SPIR-V at build time (via glslc/CMake), or use shaderc at runtime? Pros/cons for a terrain engine with ~50 shaders.
3. A buffer management strategy: the current system transfers ALL data host→device→host per dispatch. Design a smarter approach with persistent device buffers, dirty flags, and lazy readback — but that still allows the simple per-dispatch pattern as fallback.
4. A migration path that lets both OpenCL and Vulkan coexist during conversion (compile-time switch via CMake option).

Output a complete header file design for VkCompute with full class declarations and comments. Then output the CMake integration strategy."
```

**Save Gemini's output** to `docs/vulkan_architecture_plan.md`. Review it. If it looks solid, proceed. If not, iterate with Gemini until the VkCompute API design is clean.

---

## PHASE 1 — VkCompute Wrapper (Foundation)

### Step 1.1: Create the Vulkan compute abstraction

Create `external/VkCompute/` as a new submodule/library mirroring CLWrapper's structure:

```
external/VkCompute/
├── CMakeLists.txt
├── include/
│   └── vk_compute/
│       ├── vk_compute.hpp          (umbrella header)
│       ├── device_manager.hpp      (GPU selection, queue families)
│       ├── pipeline_manager.hpp    (compute pipeline + descriptor set layout caching)
│       ├── buffer_manager.hpp      (VkBuffer via VMA, staging, persistent buffers)
│       └── run.hpp                 (dispatch abstraction — mirrors CLWrapper::Run)
└── src/
    ├── device_manager.cpp
    ├── pipeline_manager.cpp
    ├── buffer_manager.cpp
    └── run.cpp
```

**Critical requirements for `Run` class:**

```cpp
namespace vkcompute {

class Run {
public:
    // Constructor loads the SPIR-V shader by name (looked up from compiled shaders)
    explicit Run(const std::string& shader_name);
    ~Run();

    // Buffer binding — MUST mirror CLWrapper::Run exactly
    template<typename T>
    void bind_buffer(const std::string& id, std::vector<T>& vector,
                     BufferUsage usage = BufferUsage::READ_WRITE);

    // Scalar argument binding — packed into a push constant block
    template<typename... Args>
    void bind_arguments(Args... args);

    // Transfer host → device
    void write_buffer(const std::string& id);

    // Dispatch compute shader
    void execute(int total_elements, float* p_elapsed_time = nullptr);
    void execute(const std::vector<int>& global_range_2d,
                 float* p_elapsed_time = nullptr);

    // Transfer device → host
    void read_buffer(const std::string& id);

    void reset_argcount();

private:
    // Vulkan handles, descriptor sets, pipeline, command buffer
};

} // namespace vkcompute
```

**Dependencies:** Vulkan SDK, VulkanMemoryAllocator (VMA), glslc (build-time SPIR-V compilation).

### Step 1.2: CMake integration

Add to the root `CMakeLists.txt`:

```cmake
option(HIGHMAP_ENABLE_VULKAN "Use Vulkan compute instead of OpenCL" ON)
option(HIGHMAP_ENABLE_OPENCL "Use OpenCL compute (legacy)" OFF)

if(HIGHMAP_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    add_subdirectory(external/VkCompute)
    target_compile_definitions(highmap PUBLIC HSD_USE_VULKAN)
endif()

if(HIGHMAP_ENABLE_OPENCL)
    add_subdirectory(external/CLWrapper)
    target_compile_definitions(highmap PUBLIC HSD_USE_OPENCL)
endif()
```

### Step 1.3: Validation

Write a minimal test that:
1. Initializes VkCompute
2. Uploads a 256×256 float buffer
3. Runs a trivial compute shader (multiply all values by 2.0)
4. Reads back and verifies every value

```bash
cd build && cmake .. -DHIGHMAP_ENABLE_VULKAN=ON && make test_vkcompute && ./test_vkcompute
```

**DO NOT proceed to Phase 2 until this test passes.**

---

## PHASE 2 — Shader Conversion (46 Kernels)

### Step 2.0: Consult Gemini for complex kernels

Before converting any kernel that involves:
- Multi-pass iterative algorithms (hydraulic erosion, thermal erosion)
- Atomic operations (particle-based erosion)
- Shared memory / local memory optimizations
- Image sampling (median, blur kernels)

Run:
```bash
gemini -p "Convert this OpenCL kernel to an optimized GLSL 460 Vulkan compute shader. Preserve exact numerical behavior. Use shared memory where beneficial. Explain any precision differences between OpenCL and Vulkan GLSL.

OpenCL kernel:
$(cat HighMap/src/gpu_opencl/kernels/KERNEL_NAME.cl)"
```

### Step 2.1: Shader compilation setup

Create `HighMap/src/gpu_vulkan/shaders/` and a CMake rule:

```cmake
file(GLOB GLSL_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
foreach(SHADER ${GLSL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPIRV_OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND Vulkan::glslc ${SHADER} -o ${SPIRV_OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME}.comp → SPIR-V"
    )
    list(APPEND SPIRV_BINARIES ${SPIRV_OUTPUT})
endforeach()
add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARIES})
```

### Step 2.2: Convert each kernel

For **every** `.cl` file in `HighMap/src/gpu_opencl/kernels/` (excluding `_common_*.cl`):

1. **Read** the OpenCL kernel
2. **Identify** the buffer bindings, scalar arguments, work dimensions
3. **Write** the equivalent `.comp` GLSL 460 compute shader with:
   - `layout(local_size_x = 16, local_size_y = 16) in;` (tune per kernel)
   - SSBOs for buffer bindings (same order as CLWrapper arg binding)
   - Push constants for scalar arguments
   - `gl_GlobalInvocationID` instead of `get_global_id()`
   - Bounds checking: `if (gid.x >= nx || gid.y >= ny) return;`
4. **Port** the `_common_*.cl` helper functions into a shared GLSL include (`_common.glsl`)

**Conversion checklist per kernel:**

| Check | Description |
|-------|-------------|
| ☐ | Buffer layout matches original bind order |
| ☐ | Push constant struct matches bind_arguments order |
| ☐ | Bounds check present |
| ☐ | float precision matches (no accidental double promotion) |
| ☐ | Atomic operations use GLSL equivalents (atomicAdd, etc.) |
| ☐ | Image/sampler operations converted to SSBO with manual interpolation OR use Vulkan image descriptors |
| ☐ | Shared memory (`__local` → `shared`) sized correctly |
| ☐ | Barrier calls converted (`barrier(CLK_LOCAL_MEM_FENCE)` → `barrier()` / `memoryBarrierShared()`) |

### Step 2.3: Conversion order (dependency-sorted)

Convert in this exact order — each group depends on the previous:

**Group A — Utilities (no dependencies):**
`_common_index`, `_common_math`, `_common_rand`, `_common_sort` → `_common.glsl`

**Group B — Simple per-pixel operations:**
`gradient_norm`, `maximum_local`, `maximum_smooth`, `minimum_smooth`, `mean_local`, `median_3x3`, `ruggedness`, `rugosity`, `smooth_cpulse`, `expand`, `rotate`, `warp`, `laplace`

**Group C — Noise generators:**
`noise`, `gavoronoise`, `gabor_wave`, `voronoi_base`, `voronoi_main`, `voronoi_fbm`, `voronoi_edge_distance`, `vorolines`, `voronoise`, `vororand_main`

**Group D — Complex terrain operations:**
`advection_particle`, `advection_warp`, `blend_poisson_bf`, `flow_direction_d8`, `generate_riverbed`, `hemisphere_field`, `interpolate_array`, `mountain_range_radial`, `normal_displacement`, `plateau`, `polygon_field`, `sdf_2d_polyline`, `skeleton`, `strata`, `rifts`

**Group E — Erosion (most complex, highest value):**
`thermal`, `thermal_inflate`, `thermal_rib`, `thermal_ridge`, `thermal_scree`, `hydraulic_particle`, `hydraulic_schott`

### Step 2.4: Validation per kernel

After converting each kernel, **immediately** validate:

```cpp
// In test_vulkan_vs_cpu.cpp — mirrors existing test_gpu_vs_cpu/main.cpp
template <typename F1, typename F2>
void compare(F1 cpu_fn, F2 vk_fn, float tolerance, const std::string& name) {
    Array z = noise_fbm(NoiseType::PERLIN, {256, 512}, {2.f, 4.f}, 1);
    remap(z);
    Array z_cpu = z, z_vk = z;

    Timer::Start(name + " - CPU");
    cpu_fn(z_cpu);
    Timer::Stop(name + " - CPU");

    Timer::Start(name + " - Vulkan");
    vk_fn(z_vk);
    Timer::Stop(name + " - Vulkan");

    AssertResults res;
    assert_almost_equal(z_cpu, z_vk, tolerance, "diff_vk_" + name + ".png", &res);
    // MUST pass. If it fails, fix the shader before moving on.
}
```

**Tolerance values:**
- Simple math operations: `1e-6`
- Noise generators: `1e-4` (floating point order differences)
- Erosion algorithms: `1e-3` (iterative accumulation)
- Particle-based: `1e-2` (stochastic, verify visually too)

**DO NOT skip validation. Every kernel must pass before proceeding.**

---

## PHASE 3 — GPU Function Rewrites

### Step 3.1: Update the *_gpu.cpp files

For each of the ~24 existing `*_gpu.cpp` files in HighMap:

1. Replace `#include "highmap/opencl/gpu_opencl.hpp"` → `#include "vk_compute/vk_compute.hpp"`
2. Replace `clwrapper::Run` → `vkcompute::Run`
3. Replace `hmap::gpu::init_opencl()` → `vkcompute::DeviceManager::init()`
4. The buffer binding / argument binding / execute / read pattern should be **identical** if VkCompute::Run is designed correctly

Use `#ifdef HSD_USE_VULKAN` / `#ifdef HSD_USE_OPENCL` guards so both backends compile.

### Step 3.2: Hesiod node GPU toggle

The Hesiod nodes already have a `GPU` boolean attribute. The node functions already branch on it:

```cpp
if (node.get_attr<BoolAttribute>("GPU")) {
    // calls hmap::gpu::some_function()
} else {
    // calls hmap::some_function() (CPU)
}
```

This pattern stays the same. The `hmap::gpu::` namespace functions just now call Vulkan internally.

---

## PHASE 4 — New GPU Paths for CPU-Only Functions

### Step 4.0: Classify CPU functions

Use Gemini to classify which CPU-only functions would benefit from GPU:

```bash
gemini -p "I have these CPU-only terrain functions. Classify each as GPU_BENEFICIAL, CPU_PREFERRED, or MIXED. Consider: data parallelism, memory access patterns, branching complexity, data dependencies.

Functions:
$(find HighMap/src -name '*.cpp' -not -name '*gpu*' -not -path '*/gpu_opencl/*' | xargs -I{} basename {} .cpp | sort)"
```

### Step 4.1: Functions that MUST stay CPU

These are inherently sequential or have complex branching:
- `dijkstra.cpp` — graph shortest path (sequential relaxation)
- `graph.cpp` — graph construction/manipulation
- `path.cpp`, `path_bezier.cpp`, `path_bspline.cpp` — curve evaluation with variable-length data
- `connected_components.cpp` — union-find (sequential)
- `flood_fill.cpp` — BFS/DFS queue-based
- `alpha_model.cpp` — network generation

**Do not attempt to GPU-accelerate these.** Mark them with a comment:
```cpp
// CPU-ONLY: Sequential algorithm, not suitable for GPU parallelization
```

### Step 4.2: Functions to add GPU paths for

High-value targets (large parallel workloads, currently CPU-only):

| Function | Why GPU | Approach |
|----------|---------|----------|
| `convolve2d_svd` | Large kernel convolution | Separable 1D passes in compute |
| `distance_transform` | Per-pixel, embarrassingly parallel | Jump flooding algorithm shader |
| `mean_shift` | Iterative per-pixel | Compute shader with shared memory |
| `kuwahara` | Per-pixel filter with neighborhood | Compute shader |
| `match_histogram` | CDF computation + remap | Parallel prefix sum + per-pixel remap |
| `phase_field` | Iterative PDE solver | Ping-pong compute shader |
| `fill_talus` | Per-pixel slope analysis | Compute shader |
| `sediment_deposition` | Per-pixel accumulation | Compute shader |
| `non_parametric_sampling` | Texture synthesis | Compute with texture lookups |
| `quilting` | Patch-based synthesis | Compute for seam finding |

For each, follow the same pattern: write `.comp` shader, write `*_gpu.cpp` wrapper, validate against CPU.

---

## PHASE 5 — Optimization Pass

### Step 5.1: Buffer persistence

After all kernels are converted, optimize buffer transfers:

1. **Track buffer residency**: If a buffer was written to device in a previous dispatch and hasn't been modified on host, skip the write.
2. **Lazy readback**: Don't read back until the host actually needs the data. The Hesiod node graph can chain GPU operations without intermediate readbacks.
3. **Double buffering for iterative algorithms**: Erosion shaders that ping-pong between two buffers should keep both on device.

### Step 5.2: Workgroup size tuning

For each shader, benchmark with workgroup sizes: 8×8, 16×16, 32×8, 8×32.
Pick the fastest for each shader category:
- Simple per-pixel: usually 16×16
- Neighborhood operations (median, blur): 16×16 with shared memory halo
- Noise generators: 8×8 (register pressure)
- Iterative erosion: 32×8 or 8×32 (memory access pattern dependent)

### Step 5.3: Async compute

Where the node graph allows, overlap CPU and GPU work:
- Submit Vulkan compute, then do CPU-only node processing, then wait for GPU result
- Use Vulkan timeline semaphores for fine-grained sync

---

## PHASE 6 — Final Validation

### Step 6.1: Full regression suite

Run the complete test_vulkan_vs_cpu with ALL kernels at multiple resolutions:
- 256×256 (fast iteration)
- 1024×1024 (production-like)
- 4096×4096 (stress test)

### Step 6.2: Visual regression

For every erosion and noise node, generate a 1024×1024 heightmap PNG from both CPU and Vulkan paths. Diff them visually. Acceptable: sub-pixel differences from float ordering. Unacceptable: visible artifacts, missing features, wrong scale.

### Step 6.3: Performance benchmarks

Generate a CSV like the existing `test_gpu_vs_cpu.csv`:

```
name;speedup;CPU_ms;Vulkan_ms;status;diff;tolerance
noise_perlin;12.3;456;37;ok;1.2e-5;1e-4
hydraulic_particle;8.7;2340;269;ok;3.1e-3;1e-2
...
```

**Target: every Vulkan kernel should be faster than CPU.** If any kernel is slower, either:
1. The workgroup size is wrong — retune
2. The transfer overhead dominates — enable buffer persistence
3. The algorithm is genuinely CPU-bound — keep CPU path as default

---

## Autonomous Execution Rules

### Resumption Protocol

**This script may be invoked multiple times** (max turns reached, network drop, crash). On every start:

1. **Check progress first.** Read `docs/PROGRESS.md` (you maintain this file).
2. **If `PROGRESS.md` exists**, resume from the last incomplete step. Do NOT redo completed work.
3. **If it doesn't exist**, create it and start from Phase 1 (Phase 0 is handled by the launcher).

`docs/PROGRESS.md` format — update this after every significant step:
```markdown
# Conversion Progress
## Current Phase: 2
## Current Step: 2.2 — Converting Group B kernels
## Last Completed: gradient_norm.comp, maximum_local.comp, maximum_smooth.comp
## Next Up: minimum_smooth.comp
## Blockers: none
## Skipped: none
## Last Updated: 2026-03-12 14:32 UTC
```

### Core Rules

1. **Work in dependency order.** Phase 1 before 2, Group A before B, etc.
2. **Compile and test after EVERY file change.** Never batch multiple untested changes.
3. **If a test fails, fix it immediately.** Do not proceed with a broken build.
4. **Use Gemini CLI for:**
   - Architecture decisions you're uncertain about
   - Complex kernel conversions (erosion, particle systems)
   - Debugging Vulkan validation layer errors
   - Optimizing workgroup sizes and memory access patterns
   - Call it like: `gemini -p "your prompt here"`
5. **Use web search for:**
   - Vulkan compute shader best practices
   - GLSL 460 compute shader syntax reference
   - VMA (VulkanMemoryAllocator) usage patterns
   - Specific Vulkan API calls you're unsure about
6. **Use the existing codebase as ground truth.** The CPU implementations define correct behavior. Your Vulkan output must match within tolerance.
7. **Commit and push after each completed group** with a descriptive message:
   ```
   git add -A && git commit -m "Phase 2B: Convert simple per-pixel kernels to Vulkan compute (13 shaders, all tests pass)" && git push
   ```
   **Always push.** The operator is remote and monitors progress via git history.
8. **If you encounter a Vulkan validation error you cannot resolve after 3 attempts**, ask Gemini with the full error + your shader + your C++ dispatch code. If Gemini can't solve it either, skip that kernel and move on — flag it with `// TODO: Vulkan validation error, needs manual review`. Add it to the Skipped list in `PROGRESS.md`.
9. **Log all performance results** to `docs/vulkan_benchmark_log.md` as you go.
10. **Update `docs/PROGRESS.md`** after every completed kernel/step. This is your save file.

### Telegram Notifications

Send a notification after completing each Phase and after any blocker:

```bash
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d "text=Phase N complete: summary here"
```

The environment variables `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are provided by the launcher. Use them directly.

Notification triggers:
- ✅ Phase completed
- ⚠️ Kernel skipped after 3 failed attempts
- 🔨 Build broken and self-repaired
- 📊 Benchmark results for a group (include speedup numbers)
- ❌ Fatal blocker that stops progress

### Error Recovery

- **Build fails:** Read the error, fix it, rebuild. If the fix isn't obvious, revert with `git checkout -- .` and try a different approach.
- **Test fails:** Compare output arrays, identify the divergence, check the shader for bugs. Common issues: wrong buffer binding order, missing bounds check, float vs int mismatch in push constants.
- **Gemini CLI fails/times out:** Proceed without it. Use web search or your own knowledge.
- **Network issues:** All work is local. Commit locally, push will happen when network returns.
- **Running out of turns:** Commit everything, update PROGRESS.md, the launcher will reinvoke you.

### Turn Budget Strategy

You have ~200 turns per session. Budget them:
- Phase 1 (VkCompute wrapper): ~40 turns
- Phase 2 (shader conversion): ~80 turns (this is the bulk)
- Phase 3 (GPU function rewrites): ~30 turns
- Phase 4 (new GPU paths): ~30 turns
- Phase 5-6 (optimization + validation): ~20 turns

If you're running low on turns, **prioritize committing and updating PROGRESS.md** so the next session can resume cleanly.

---

## Quick Reference: OpenCL → GLSL Mapping

| OpenCL | GLSL 460 Compute |
|--------|-----------------|
| `get_global_id(0)` | `gl_GlobalInvocationID.x` |
| `get_global_id(1)` | `gl_GlobalInvocationID.y` |
| `get_local_id(0)` | `gl_LocalInvocationID.x` |
| `get_group_id(0)` | `gl_WorkGroupID.x` |
| `get_local_size(0)` | `gl_WorkGroupSize.x` |
| `__kernel void name(...)` | `void main()` with `layout(local_size_x=X, local_size_y=Y) in;` |
| `__global float* buf` | `layout(set=0, binding=N) buffer Buf { float data[]; };` |
| `__local float arr[N]` | `shared float arr[N];` |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `barrier(); memoryBarrierShared();` |
| `atomic_add(p, v)` | `atomicAdd(p, v)` |
| `read_imagef(img, sampler, coord)` | Manual SSBO index or `texture(sampler2D, coord)` |
| `write_imagef(img, coord, val)` | `imageStore(image2D, coord, val)` |
| `clamp(x, lo, hi)` | `clamp(x, lo, hi)` (same) |
| `mix(a, b, t)` | `mix(a, b, t)` (same) |
| `native_exp(x)` | `exp(x)` |
| `native_sqrt(x)` | `sqrt(x)` |
| Push constants | `layout(push_constant) uniform PushConstants { int nx; int ny; ... };` |

---

## Start Command

The launcher script has already cloned the repo and run Gemini (Phase 0). The architecture plan is in `docs/vulkan_architecture_plan.md`.

**YOUR VERY FIRST ACTION — before reading ANY files:**

```bash
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d "text=👋 Claude Code is alive and running. Reading project files now..."
```

This confirms to the operator (who is remote) that you are actually executing. Do this IMMEDIATELY.

**Then:**

1. Read `docs/PROGRESS.md` — if it exists, resume from where it left off
2. If no `PROGRESS.md`, read `docs/vulkan_architecture_plan.md` to understand the VkCompute design
3. Begin Phase 1 (or resume the incomplete phase)

**Go.**
