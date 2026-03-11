# Conversion Progress

## Current Phase: 3
## Current Step: 3.1 — Update *_gpu.cpp files to use vkcompute::Run
## Last Completed: Group E (11 shaders: thermal, thermal_with_bedrock, thermal_auto_bedrock, thermal_inflate, thermal_rib, thermal_ridge, thermal_scree, hydraulic_particle, hydraulic_schott, jump_flooding, mean_shift) — ALL 66 SHADERS COMPILE OK
## Next Up: Phase 3 — rewrite ~24 *_gpu.cpp files to use vkcompute::Run
## Blockers: none
## Skipped: none
## Last Updated: 2026-03-11 UTC

## Phase 1 Summary — DONE ✅
- VkCompute library: external/HighMap/external/VkCompute/
- GPU: NVIDIA GeForce RTX 5080 (Vulkan 1.4.328)
- API: DeviceManager, PipelineManager, Run (mirrors CLWrapper exactly)
- Validation: test_vkcompute_init PASSES — 65536 floats, multiply_by_two shader, all == 2.0
- Shader path: ${CMAKE_BINARY_DIR}/shaders/ (SPIR-V compiled at build time via glslc)
- HighMap CMakeLists updated: HIGHMAP_ENABLE_VULKAN=ON links vkcompute + compiles shaders
- gpu_vulkan.hpp: HighMap/include/highmap/vulkan/gpu_vulkan.hpp
- gpu_vulkan.cpp: HighMap/src/gpu_vulkan/gpu_vulkan.cpp

## Environment
- Vulkan SDK: /c/VulkanSDK/1.4.328.1/ (Vulkan 1.4, VMA in SDK Include/vma/)
- Shader compiler: Vulkan::glslc
- VkCompute standalone build: /c/Dev/HesiodVulkan/vkcompute-build/
- MSVC requires /utf-8 flag (for spdlog/fmt)
- Qt6: /c/Qt/6.10.2/msvc2022_64/

## Phase 2 Kernel Conversion Status
Group A — Common helpers (→ _common.glsl):   [x] _common_index  [x] _common_math  [x] _common_rand  [x] _common_sort
Group B — Simple per-pixel:                   [x] gradient_norm  [x] maximum_local  [x] maximum_smooth  [x] minimum_smooth
                                              [x] mean_local  [x] median_3x3  [x] ruggedness  [x] rugosity
                                              [x] smooth_cpulse  [x] expand  [x] rotate  [x] warp  [x] laplace
Group C — Noise generators:                   [x] noise  [x] gavoronoise  [x] gabor_wave  [x] voronoi_base
                                              [x] voronoi_main  [x] voronoi_fbm  [x] voronoi_edge_distance
                                              [x] vorolines  [x] voronoise  [x] vororand_main  [x] wavelet_noise
Group D — Complex terrain:                    [x] advection_particle  [x] advection_warp  [x] blend_poisson_bf
                                              [x] flow_direction_d8  [x] generate_riverbed  [x] hemisphere_field
                                              [x] interpolate_array  [x] mountain_range_radial  [x] normal_displacement
                                              [x] plateau  [x] polygon_field  [x] sdf_2d_polyline
                                              [x] skeleton  [x] strata  [x] rifts
Group E — Erosion (complex):                  [x] thermal  [x] thermal_inflate  [x] thermal_rib  [x] thermal_ridge
                                              [x] thermal_scree  [x] hydraulic_particle  [x] hydraulic_schott
                                              [x] jump_flooding  [x] mean_shift
