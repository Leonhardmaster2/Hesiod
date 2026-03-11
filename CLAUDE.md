# Full Vulkan System — GPU-Resident Pipeline + Parallel Graph + Vulkan Renderer

## Context

This is Phase 2 of the Hesiod Vulkan conversion. Phase 1 (compute shader conversion) is complete — all 69 OpenCL kernels are now Vulkan compute shaders, VkCompute wrapper is working, hesiod.exe builds and runs.

**What's still wrong:** Every node dispatch copies data CPU→GPU, runs the shader, then copies GPU→CPU. The renderer is OpenGL. The node graph executes sequentially. This means:
- A 4096×4096 heightmap = 64MB copied FOUR times per node (write+read × input+output)
- Independent graph branches wait for each other
- Compute results copy back to CPU just to get re-uploaded to OpenGL for rendering

**Target:** A fully Vulkan system where data stays on GPU, nodes execute in parallel, and the renderer reads compute output directly.

---

## Architecture Overview

```
BEFORE (current):
Node A: CPU→VkBuffer→compute→VkBuffer→CPU → Node B: CPU→VkBuffer→compute→VkBuffer→CPU → OpenGL upload → render

AFTER (target):
Node A: VkBuffer→compute→VkBuffer → Node B: same VkBuffer→compute→VkBuffer → Vulkan render (same VkBuffer, zero copy)
                                ↗
Node C: VkBuffer→compute→VkBuffer ─┘  (parallel with A+B if independent)
```

### Key Components to Build/Modify

1. **GpuBuffer** — a wrapper around VkBuffer that tracks residency, dirty state, and optionally holds a CPU shadow copy
2. **GpuHeightmap** — extends or wraps `hmap::Heightmap` so tiles store `GpuBuffer` instead of `std::vector<float>`
3. **Parallel graph scheduler** — modifies GNode's `Graph::update()` to dispatch independent nodes concurrently using multiple Vulkan compute queues or a thread pool
4. **VkRenderer** — replaces QTerrainRenderer's OpenGL with Vulkan graphics pipeline, reading directly from GpuBuffers

---

## PHASE A — GPU-Resident Buffers

### Step A.1: GpuBuffer class

Create `external/HighMap/external/VkCompute/include/vk_compute/gpu_buffer.hpp`:

```cpp
namespace vkcompute {

class GpuBuffer {
public:
    GpuBuffer();  // empty, no allocation
    GpuBuffer(size_t size_bytes);  // allocate device buffer
    ~GpuBuffer();

    // Upload from CPU vector (only if dirty_cpu is true or force=true)
    void upload(const std::vector<float>& cpu_data, bool force = false);

    // Download to CPU vector (only if dirty_gpu is true or force=true)
    void download(std::vector<float>& cpu_data, bool force = false);

    // Mark that CPU data has changed (needs re-upload)
    void mark_cpu_dirty();

    // Mark that GPU data has changed (needs readback before CPU access)
    void mark_gpu_dirty();

    // Get the raw VkBuffer handle (for binding to compute/render pipelines)
    VkBuffer get_vk_buffer() const;

    // Get VkDeviceMemory or VmaAllocation for descriptor set binding
    VmaAllocation get_allocation() const;

    // Size
    size_t size() const;

    // Is data currently on GPU?
    bool is_resident() const;

private:
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    bool dirty_cpu_ = false;   // CPU data changed, GPU needs update
    bool dirty_gpu_ = false;   // GPU data changed, CPU needs update
    bool resident_ = false;    // Buffer exists on GPU
};

} // namespace vkcompute
```

### Step A.2: Modify VkCompute::Run to accept GpuBuffer

The `Run::bind_buffer` method should accept both `std::vector<float>&` (legacy, copies every time) AND `GpuBuffer&` (zero-copy if already resident):

```cpp
// Legacy path (backward compatible — copies every dispatch)
template<typename T>
void bind_buffer(const std::string& id, std::vector<T>& vector);

// GPU-resident path (zero-copy if already on device)
void bind_buffer(const std::string& id, GpuBuffer& gpu_buf);
```

When `bind_buffer(id, gpu_buf)` is called:
- If `gpu_buf.is_resident()` and not dirty, skip upload entirely
- Bind the existing `VkBuffer` directly to the descriptor set
- After `execute()`, mark `gpu_buf` as `dirty_gpu_` (compute shader wrote to it)

### Step A.3: GpuArray — GPU-resident version of hmap::Array

`hmap::Array` currently stores data as `std::vector<float> vector`. Create a parallel structure:

```cpp
namespace hmap {

class GpuArray {
public:
    Vec2<int> shape;
    GpuBuffer gpu_buffer;

    // Lazy CPU access — downloads only when needed
    const std::vector<float>& cpu_data();

    // Upload CPU changes to GPU
    void sync_to_gpu();

    // Convert from/to regular Array
    static GpuArray from_array(const Array& arr);
    Array to_array() const;

    // Direct GPU buffer access for compute dispatch
    GpuBuffer& buffer() { return gpu_buffer; }
};

} // namespace hmap
```

### Step A.4: Modify hmap::Heightmap to support GPU tiles

`hmap::Heightmap` holds a vector of `Tile` objects (which inherit from `Array`). For GPU residency, each tile needs an associated `GpuBuffer`.

**Option A (non-invasive):** Add a parallel `std::vector<GpuBuffer> gpu_tiles` to Heightmap. The `hmap::transform()` function checks if GPU mode is active and uses `gpu_tiles` instead of the CPU tile vectors.

**Option B (invasive but cleaner):** Make `Array::vector` optionally backed by a `GpuBuffer`, with lazy CPU readback via `operator[]` or explicit `cpu_data()` call.

**Use Option A** — it's safer and doesn't break existing CPU code paths. The `hmap::transform()` functions already have GPU branches that check the `GPU` boolean attribute.

### Step A.5: Modify hmap::transform() for GPU-resident dispatch

The current `hmap::transform()` iterates over tiles and calls a lambda per tile. For GPU mode, it should:

1. Check if input tiles have resident GpuBuffers
2. If yes, bind them directly (zero-copy)
3. If no, upload and mark resident
4. After compute, mark output GpuBuffers as gpu_dirty
5. Do NOT readback unless the next consumer is a CPU-only node

This requires knowing whether the downstream node is GPU or CPU. For now, use a simple heuristic: **never readback automatically**. Let the consumer trigger readback when it needs CPU data.

### Step A.6: Validation

Build a test that:
1. Creates two noise heightmaps on GPU
2. Blends them (GPU compute)
3. Applies erosion (GPU compute)
4. Verifies data never left the GPU between steps (no download calls)
5. Only downloads at the end for verification

**Benchmark:** Compare wall time for a 5-node chain at 4096×4096 with current (copy every node) vs GPU-resident (copy only at end). Target: 3-5x speedup from eliminated transfers alone.

---

## PHASE B — Parallel Node Graph Execution

### Step B.1: Analyze GNode's execution model

GNode's `Graph::update()` does:
1. Topological sort of dirty nodes
2. Sequential execution in sorted order

The topological sort already identifies which nodes have no dirty dependencies — these can run in parallel.

### Step B.2: Parallel scheduler

Modify `Graph::update()` to use a **wavefront parallel** approach:

```
Wave 0: [all nodes with in-degree 0] → dispatch all simultaneously
Wave 1: [nodes whose dependencies are all in wave 0] → dispatch when wave 0 completes
Wave 2: [nodes whose dependencies are all in waves 0-1] → etc.
```

Implementation:

```cpp
void Graph::update_parallel()
{
    auto connectivity_dw = get_connectivity_downstream();
    auto connectivity_up = get_connectivity_upstream();

    // Compute in-degrees
    std::unordered_map<std::string, int> in_degree;
    for (auto& [nid, _] : nodes) in_degree[nid] = 0;
    for (auto& [nid, deps] : connectivity_up)
        for (auto& dep : deps) in_degree[nid]++;

    // Process waves
    std::queue<std::string> ready;
    for (auto& [nid, deg] : in_degree)
        if (deg == 0) ready.push(nid);

    while (!ready.empty()) {
        // Collect current wave
        std::vector<std::string> wave;
        while (!ready.empty()) {
            wave.push_back(ready.front());
            ready.pop();
        }

        // Execute wave in parallel
        // GPU nodes: submit to separate Vulkan compute queues
        // CPU nodes: dispatch to thread pool
        std::vector<std::future<void>> futures;
        for (auto& nid : wave) {
            futures.push_back(std::async(std::launch::async, [&, nid]() {
                get_node_ref_by_id(nid)->update();
            }));
        }
        for (auto& f : futures) f.get();

        // Update in-degrees
        for (auto& nid : wave)
            for (auto& dw : connectivity_dw[nid])
                if (--in_degree[dw] == 0) ready.push(dw);
    }
}
```

### Step B.3: Vulkan queue management for parallel compute

The RTX 5080 has multiple compute queues. Assign each parallel GPU node to a separate queue:

```cpp
// In DeviceManager: expose multiple compute queues
std::vector<VkQueue> compute_queues;  // typically 8+ on NVIDIA

// In Run: accept a queue index
void execute(int total_elements, int queue_index = 0);
```

Use Vulkan semaphores between waves to ensure ordering:
- All dispatches in a wave use the same timeline semaphore signal value
- Next wave waits on that value before starting

### Step B.4: Thread safety

The parallel scheduler introduces concurrency. Ensure:
- `GpuBuffer` operations are thread-safe (mutex on upload/download)
- `VkCompute::Run` uses per-thread command buffers (don't share command buffers across threads)
- `PipelineManager` pipeline cache is thread-safe (read-only after init, or use a concurrent map)
- Node `update()` functions don't share mutable state (they shouldn't — each node owns its output data)

### Step B.5: Validation

Test with a diamond-shaped graph:
```
    A
   / \
  B   C    ← B and C should execute in parallel
   \ /
    D
```

Verify:
- B and C overlap in execution time
- D executes only after both B and C complete
- Results match sequential execution

---

## PHASE C — Vulkan Renderer

### Step C.1: Create VkTerrainRenderer

Replace `QTerrainRenderer` (OpenGL-based `QOpenGLWidget`) with a Vulkan-based renderer. Qt6 supports `QVulkanWindow` and `QVulkanWindowRenderer`.

Create `external/VkTerrainRenderer/`:

```
external/VkTerrainRenderer/
├── CMakeLists.txt
├── include/
│   └── vktr/
│       ├── vk_terrain_renderer.hpp    (umbrella header)
│       ├── render_widget.hpp          (QVulkanWindowRenderer subclass)
│       ├── pipeline.hpp               (graphics pipeline: vertex+fragment)
│       ├── swapchain.hpp              (swapchain management)
│       ├── camera.hpp                 (reuse from QTerrainRenderer)
│       ├── light.hpp                  (reuse from QTerrainRenderer)
│       ├── mesh.hpp                   (Vulkan vertex/index buffers)
│       └── shaders/
│           ├── terrain.vert
│           ├── terrain.frag
│           ├── shadow_depth.vert
│           ├── shadow_depth.frag
│           ├── viewer2d.vert
│           └── viewer2d.frag
└── src/
    ├── render_widget.cpp
    ├── pipeline.cpp
    ├── swapchain.cpp
    ├── camera.cpp
    ├── light.cpp
    └── mesh.cpp
```

### Step C.2: Shared Vulkan instance

The compute pipeline (VkCompute) and renderer (VkTerrainRenderer) MUST share the same `VkDevice` and `VkPhysicalDevice`. Otherwise you can't share buffers.

Modify `DeviceManager` to be a true singleton that both systems reference:

```cpp
class DeviceManager {
public:
    static DeviceManager& instance();

    VkInstance instance();
    VkPhysicalDevice physical_device();
    VkDevice device();
    VmaAllocator allocator();

    // Compute queues (used by VkCompute::Run)
    VkQueue compute_queue(int index = 0);
    uint32_t compute_queue_family();

    // Graphics queue (used by VkTerrainRenderer)
    VkQueue graphics_queue();
    uint32_t graphics_queue_family();

    // Transfer queue (optional, for async uploads)
    VkQueue transfer_queue();
};
```

### Step C.3: Convert GLSL shaders

The existing OpenGL shaders (.vert/.frag) need conversion to Vulkan GLSL 460:

| OpenGL | Vulkan |
|--------|--------|
| `uniform mat4 model` | `layout(push_constant) uniform PushConstants { mat4 model; ... }` or UBO |
| `in vec3 position` | `layout(location = 0) in vec3 position` |
| `out vec4 fragColor` | `layout(location = 0) out vec4 fragColor` |
| `uniform sampler2D tex` | `layout(set = 0, binding = N) uniform sampler2D tex` |
| `gl_Position = ...` | Same |

The terrain vertex shader should read heightmap data directly from the compute SSBO:

```glsl
// terrain.vert
#version 460

layout(set = 0, binding = 0) readonly buffer HeightmapData {
    float heights[];
};

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    int width;
    int height;
    float scale_y;
};

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv;

void main() {
    int x = gl_VertexIndex % width;
    int z = gl_VertexIndex / width;
    float y = heights[z * width + x] * scale_y;

    // Position
    vec3 pos = vec3(float(x) / float(width - 1) - 0.5,
                    y,
                    float(z) / float(height - 1) - 0.5);

    // Normal from finite differences
    float hL = (x > 0) ? heights[z * width + (x-1)] : heights[z * width + x];
    float hR = (x < width-1) ? heights[z * width + (x+1)] : heights[z * width + x];
    float hD = (z > 0) ? heights[(z-1) * width + x] : heights[z * width + x];
    float hU = (z < height-1) ? heights[(z+1) * width + x] : heights[z * width + x];
    normal = normalize(vec3(hL - hR, 2.0, hD - hU));

    uv = vec2(float(x) / float(width - 1), float(z) / float(height - 1));
    world_pos = pos;
    gl_Position = mvp * vec4(pos, 1.0);
}
```

This reads DIRECTLY from the compute shader's output buffer — zero copies.

### Step C.4: Render pass setup

The Vulkan renderer needs:
1. **Main render pass:** color + depth attachments, renders terrain mesh
2. **Shadow pass:** depth-only attachment, renders from light POV (same as current OpenGL shadow map)
3. **2D viewer mode:** fullscreen quad with colormap shader (replaces viewer2d_cmap.vert/.frag)

### Step C.5: Qt6 integration

Use `QVulkanWindow` + `QVulkanWindowRenderer`:

```cpp
class TerrainVulkanRenderer : public QVulkanWindowRenderer {
public:
    void initResources() override;        // Create pipelines, descriptor sets
    void releaseResources() override;     // Cleanup
    void startNextFrame() override;       // Record command buffers, submit

    // Called by Hesiod when heightmap data changes
    void update_heightmap(GpuBuffer& height_data, int width, int height);
    void update_albedo(GpuBuffer& color_data, int width, int height);
    void update_normal_map(GpuBuffer& normal_data, int width, int height);
};
```

The `update_heightmap` call just stores a reference to the `GpuBuffer` — no data copy. On next `startNextFrame()`, it binds that buffer to the terrain vertex shader descriptor set.

### Step C.6: Hesiod integration

In Hesiod's GUI code, replace `QOpenGLWidget` (RenderWidget) with the new `QVulkanWindow`:

```cpp
// Before:
auto* render_widget = new qtr::RenderWidget("3D View", parent);

// After:
auto* vk_window = new QVulkanWindow();
vk_window->setVulkanInstance(DeviceManager::instance().qt_vulkan_instance());
auto* renderer = new TerrainVulkanRenderer();
vk_window->setRenderer(renderer);
auto* container = QWidget::createWindowContainer(vk_window, parent);
```

### Step C.7: Validation

1. Render the same heightmap with OpenGL (old) and Vulkan (new) side by side
2. Screenshot both, pixel-diff them — should be near-identical (lighting math same)
3. Benchmark: measure frame time for rotating a 4096×4096 terrain
4. Verify that modifying a compute parameter (e.g. erosion strength) updates the viewport WITHOUT any CPU readback/re-upload

---

## PHASE D — Integration and Optimization

### Step D.1: End-to-end GPU pipeline test

Create a complete test scenario:
1. Noise generator (GPU compute) → outputs GpuBuffer
2. Erosion (GPU compute) → reads GpuBuffer directly, outputs GpuBuffer
3. Colorize (GPU compute) → reads height GpuBuffer, outputs color GpuBuffer
4. Render (Vulkan graphics) → reads both GpuBuffers directly

**Verify:** Zero `vkCmdCopyBuffer` device→host calls in the entire pipeline. Use Vulkan validation layers + `VK_EXT_debug_utils` to confirm.

### Step D.2: CPU node fallback

Some nodes are CPU-only (Dijkstra, flood fill, path evaluation). When the graph scheduler encounters a CPU node after a GPU node:
1. Insert a GPU→CPU readback (download GpuBuffer to std::vector)
2. Execute CPU node
3. If next node is GPU, re-upload

This is the only place transfers should happen. Log a warning when it occurs so the user knows which nodes break the GPU chain.

### Step D.3: Memory budget

Track total GPU memory usage. On the RTX 5080 you have ~16GB VRAM. A 4096×4096 float heightmap = 64MB. With all tiles + textures + staging buffers, set a budget:
- If approaching 80% VRAM, evict least-recently-used GpuBuffers
- Keep a priority system: currently-visible nodes > upstream nodes > disconnected nodes

### Step D.4: Final benchmarks

Run the full benchmark suite from Phase 1's `test_vulkan_vs_cpu.csv` but now measure the FULL pipeline including transfers:

```
name;old_total_ms;new_total_ms;speedup;notes
5_node_erosion_chain_4096;18000;720;25x;zero-copy between nodes
10_node_terrain_gen_4096;12000;800;15x;parallel waves + zero-copy
viewport_update_4096;180;32;5.6x;direct buffer binding
full_graph_rebuild_4096;8000;600;13x;everything
```

---

## Autonomous Execution Rules

Same rules as Phase 1 CLAUDE.md, plus the additions below.

### Gemini as Discussion Partner

Gemini is NOT a code generator for you to copy from. Gemini is a second opinion — a colleague you bounce ideas off of when you're unsure. Use it like this:

**When to consult Gemini:**
- You're choosing between two architectural approaches and aren't sure which is better
- A Vulkan validation error doesn't make sense to you
- You're unsure about synchronization (semaphores, fences, barriers) — this is where most Vulkan bugs hide
- You're designing a new abstraction (GpuBuffer, parallel scheduler) and want feedback on the API
- Something builds but behaves wrong at runtime

**How to consult Gemini:**

```bash
gemini -p "I'm building [specific thing]. Here's my current approach:

[paste your code or pseudocode]

My concern is [specific concern]. The alternative would be [alternative].

Which approach is better and why? Are there Vulkan-specific pitfalls I'm missing?"
```

Then READ Gemini's response, THINK about whether it makes sense given what you know about this codebase, and make YOUR OWN decision. If Gemini suggests something that contradicts the existing codebase patterns, trust the codebase — it's already working.

**Example of a good Gemini discussion:**

```bash
gemini -p "I need to share a VkBuffer between a compute dispatch and a graphics render pass on the same device. The compute shader writes to the buffer, then the vertex shader reads it.

Option A: Use a pipeline barrier (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT → VK_PIPELINE_STAGE_VERTEX_SHADER_BIT) with VK_ACCESS_SHADER_WRITE_BIT → VK_ACCESS_SHADER_READ_BIT.

Option B: Use a timeline semaphore between the compute and graphics queue submissions.

The compute and graphics operations are on the SAME queue. Which is correct? Are there edge cases?"
```

**Never do this:**
```bash
# BAD: asking Gemini to write entire files
gemini -p "Write a complete VkTerrainRenderer class with swapchain management"
```

That's YOUR job. Gemini helps you think, not write.

### Runtime Validation — Actually Run the Code

Do NOT just compile and assume it works. After every significant change:

**1. Add logging to every critical path:**

```cpp
// In GpuBuffer::upload()
spdlog::info("[GpuBuffer] upload: {} bytes, buffer={}, force={}", size_, (void*)buffer_, force);

// In GpuBuffer::download()
spdlog::info("[GpuBuffer] download: {} bytes, buffer={}", size_, (void*)buffer_);

// In Run::execute()
spdlog::info("[VkCompute] dispatch: shader={}, elements={}, queue={}, elapsed={:.3f}ms",
             shader_name_, total_elements, queue_index, elapsed_ms);

// In Graph::update_parallel()
spdlog::info("[Graph] wave {}: dispatching {} nodes: {}", wave_num, wave.size(), node_names);
spdlog::info("[Graph] wave {} complete: {:.1f}ms", wave_num, wave_time_ms);

// In TerrainVulkanRenderer::startNextFrame()
spdlog::info("[Renderer] frame {}: draw calls={}, buffer binds={}, frame_time={:.2f}ms",
             frame_count_, draw_calls, buffer_binds, frame_time_ms);
```

**2. Run the app/test and capture output:**

```bash
# Run test binary and capture output
build\bin\Release\test_vkcompute.exe > logs\test_output.log 2>&1

# Check the output for errors or unexpected behavior
type logs\test_output.log | findstr /i "error\|fail\|warn\|assert"

# Check for expected log lines
type logs\test_output.log | findstr /i "upload\|download\|dispatch"
```

**3. Parse the output and verify:**

After running, READ the log file. Check:
- Are there unexpected `download` calls between GPU nodes? (means zero-copy isn't working)
- Are wave dispatch times overlapping? (means parallelism is working)
- Are there Vulkan validation errors? (means synchronization bug)
- What's the total frame time? (means renderer is working)

**4. Performance measurement protocol:**

For every benchmark, run the operation 3 times and report the median:

```cpp
// Add this timing wrapper to all critical operations
auto t_start = std::chrono::high_resolution_clock::now();
// ... operation ...
auto t_end = std::chrono::high_resolution_clock::now();
float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
spdlog::info("[BENCHMARK] {}: {:.3f}ms", operation_name, ms);
```

Then parse the benchmark log:
```bash
type logs\benchmark.log | findstr "BENCHMARK"
```

Send benchmark results via Telegram:
```bash
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d "text=📊 Phase A benchmarks: 5-node chain 4096x4096: old=18000ms new=720ms (25x speedup). Zero downloads between nodes confirmed."
```

**5. Vulkan validation layers:**

ALWAYS run with validation layers enabled during development:

```cpp
// In DeviceManager init, enable:
VK_LAYER_KHRONOS_validation
```

Check for validation errors after every run:
```bash
type logs\test_output.log | findstr /i "validation\|VUID\|ERROR"
```

If there are validation errors, fix them before moving on. Common ones:
- Missing pipeline barriers → add proper synchronization
- Descriptor set not updated → check binding order
- Buffer not mapped → check VMA allocation flags

### Resumption Protocol

Maintain `docs/PROGRESS_PHASE2.md` with the same format. Update after every step.

### Phase priorities

If running low on turns:
- **Phase A is the biggest win** — GPU-resident buffers eliminate transfer overhead
- **Phase C is the most complex** — Vulkan renderer is a lot of code
- **Phase B is the easiest** — parallel scheduling is ~200 lines of code

If you can only finish one phase, do A. If two, do A+B. C is the cherry on top.

### Testing strategy

- Phase A: benchmark transfer elimination, verify zero downloads between GPU nodes via log parsing
- Phase B: verify parallel execution with timing logs, confirm waves overlap
- Phase C: visual comparison + frame timing, check for validation errors
- Phase D: end-to-end zero-copy verification, full pipeline benchmark

### Telegram notifications

```bash
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d "text=your message"
```

Send after: each Phase complete, any blocker, benchmark results (with actual numbers), Gemini consultation summaries.

### Commit strategy

```bash
git add -A && git commit -m "Phase A.3: GpuArray with lazy CPU readback, all tests pass" && git push
```

---

## Build Instructions

**IMPORTANT: A full build takes ~1 hour on this machine.** Do NOT build after every small change. Batch your work.

### DO NOT modify CMakeLists.txt files

The app is currently building and working. Any CMakeLists.txt change risks breaking the build and wasting an hour.

**Instead:** Write all planned CMake changes to `docs/CMAKE_CHANGES_PHASE2.txt`. Format:

```
# CMAKE CHANGES — Apply all at once before next build
# File: external/HighMap/external/VkCompute/CMakeLists.txt
# Change: Add gpu_buffer.cpp to SOURCES list
# Reason: New GpuBuffer class

# File: external/VkTerrainRenderer/CMakeLists.txt  
# Change: New CMakeLists for Vulkan renderer library
# Reason: Phase C renderer
# Full content:
#   [paste the entire CMakeLists.txt here]
```

Then, right before a build checkpoint, apply ALL the accumulated CMake changes at once, rebuild, and fix any issues in one go.

### When to build

Build ONLY at these checkpoints:
- **Phase A complete** — GpuBuffer, GpuArray, modified transform() all done → build + test
- **Phase B complete** — parallel scheduler done → build + test
- **Phase C.3 complete** — all Vulkan shaders converted, renderer compiles → build + test
- **Phase C complete** — full renderer integrated → build + test + visual check
- **Phase D complete** — final integration → build + benchmark

That's ~5 builds total. Between builds, you can still verify individual files compile with a quick syntax check:

```bash
# Quick compile check for a single file (seconds, not hours)
cl /c /std:c++20 /utf-8 /I... your_file.cpp
```

### When NOT to build
- After writing a single .hpp header
- After writing a single .comp shader (just run `glslc shader.comp -o shader.spv` to verify it compiles)
- After modifying CMakeLists.txt (save for the next real build)
- After any change that doesn't complete a full sub-phase

### Build commands

```bash
# Configure (from repo root) — run once or when CMake changes
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
    -DHIGHMAP_ENABLE_VULKAN=ON ^
    -DHIGHMAP_ENABLE_OPENCL=OFF ^
    -DNANOFLANN_BUILD_EXAMPLES=OFF ^
    -DNANOFLANN_BUILD_TESTS=OFF ^
    -DCMAKE_PREFIX_PATH=C:/Qt/6.10.2/msvc2022_64 ^
    -DCMAKE_MAXIMUM_RECURSION_DEPTH=2000

# Build (uses all 20 threads on the machine)
cmake --build build --config Release --parallel 20

# Run tests
build\bin\Release\test_vkcompute.exe > logs\test_output.log 2>&1

# Run benchmarks
build\bin\Release\benchmark_pipeline.exe > logs\benchmark.log 2>&1

# Run the full app
build\bin\Release\hesiod.exe > logs\app_output.log 2>&1
```

### The cycle at build checkpoints

1. Build (`cmake --build build --config Release --parallel 20`)
2. If build fails → fix → rebuild
3. Run test binary → capture log → parse for errors
4. Run benchmark → capture log → send results via Telegram
5. Commit + push

---

## Benchmark Requirements

All benchmarks must be REAL numbers from ACTUAL runs on the RTX 5080. No estimates, no projections.

### Focus: High-Cost Terrain Nodes

The most interesting benchmarks are the **expensive nodes** — where GPU residency and parallelism actually matter. Simple per-pixel ops are already fast. What matters is:

- **Hydraulic particle erosion** (iterative, thousands of particles) — this is THE benchmark
- **Thermal erosion** (iterative, multi-pass)
- **Hydraulic stream** (flow accumulation + erosion)
- **Noise FBM** at high octave counts (8+ octaves)
- **Multi-node erosion chains** (noise → thermal → hydraulic → smooth → export)

### Benchmark test program

Create `tests/benchmark_pipeline/main.cpp` early in Phase A:

```cpp
// Benchmark: real terrain workflows on RTX 5080
// Run with: build\bin\Release\benchmark_pipeline.exe > logs\benchmark.log 2>&1

// === Single node benchmarks (high-cost operations) ===

// Test 1: hydraulic_particle erosion 4096x4096, 1M particles
//   - Old: with CPU<->GPU transfer per dispatch
//   - New: GPU-resident, no transfers

// Test 2: thermal erosion 4096x4096, 50 iterations

// Test 3: hydraulic_stream 4096x4096

// Test 4: noise_fbm 4096x4096, 8 octaves, Perlin

// === Pipeline benchmarks (realistic workflows) ===

// Test 5: Full terrain pipeline (THE benchmark):
//   noise_fbm -> thermal(30 iter) -> hydraulic_particle(200k) -> smooth -> export
//   Size: 4096x4096 ONLY. Always 4K. No smaller resolutions.

// Test 6: Parallel branch test:
//   noise_A --> thermal --> blend --> hydraulic -> out
//   noise_B --> warp    --/
//   Measure: sequential time vs parallel time

// Test 7: Compute-to-render latency:
//   Modify erosion parameter -> measure time to pixels on screen

// Each test: 5 runs, report median + min + max
// [BENCH] test_name | resolution | median_ms | min_ms | max_ms | speedup | notes
```

Run this after Phase A, after Phase B, and after Phase D. Send results via Telegram:

```bash
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d "text=$(type logs\benchmark.log | findstr BENCH)"
```

Save results to `docs/BENCHMARKS_PHASE2.md` and commit. This file should grow with each phase.

### What counts as success

| Metric | Target | How to verify |
|--------|--------|---------------|
| Zero-copy between GPU nodes | 0 download calls between dispatches | `findstr download logs\test_output.log` |
| Hydraulic erosion 4096² (1M particles) | <300ms with GPU residency | `[BENCH] hydraulic_particle` |
| 5-node terrain pipeline 4096² | >5x faster than transfer-every-node | `[BENCH] full_pipeline` |
| Parallel branch speedup | >1.5x for 2 independent noise gens | `[BENCH] parallel_branch` |
| Render frame time (4096²) | <16ms (60fps) | `[Renderer] frame_time` |
| Parameter-to-pixels latency | <200ms for erosion param change | `[BENCH] compute_to_render` |
| Memory stable | No VRAM growth over 100 frames | VMA stats in log |

---

## Start Command

**YOUR VERY FIRST ACTION — before reading ANY files:**

```bash
curl -s -X POST "https://api.telegram.org/bot%TELEGRAM_BOT_TOKEN%/sendMessage" -d chat_id=%TELEGRAM_CHAT_ID% -d "text=👋 Claude Code Phase 2 started — GPU-resident pipeline + parallel graph + Vulkan renderer"
```

Then:
1. Read `docs/PROGRESS_PHASE2.md` — if it exists, resume from where it left off
2. Read `docs/PROGRESS.md` for Phase 1 environment details (build flags, paths, etc.)
3. If no `PROGRESS_PHASE2.md`, start with Phase A (GPU-resident buffers)
4. Create `tests/benchmark_pipeline/main.cpp` early — you'll use it throughout
5. Work through A → B → C → D in order
6. Build + run + parse logs after EVERY change

**Go.**
