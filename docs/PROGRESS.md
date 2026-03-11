# Conversion Progress

## Current Phase: 2
## Current Step: 2.1 — Shader compilation setup + Group A (_common.glsl)
## Last Completed: Phase 1 complete — VkCompute wrapper, validation test PASSES
## Next Up: Convert Group A common helpers → _common.glsl, then Group B simple shaders
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

## Phase 2 Kernel Conversion Status
Group A — Common helpers (→ _common.glsl):   [ ] _common_index  [ ] _common_math  [ ] _common_rand  [ ] _common_sort
Group B — Simple per-pixel:                   [ ] gradient_norm  [ ] maximum_local  [ ] maximum_smooth  [ ] minimum_smooth
                                              [ ] mean_local  [ ] median_3x3  [ ] ruggedness  [ ] rugosity
                                              [ ] smooth_cpulse  [ ] expand  [ ] rotate  [ ] warp  [ ] laplace
Group C — Noise generators:                   [ ] noise  [ ] gavoronoise  [ ] gabor_wave  [ ] voronoi_base
                                              [ ] voronoi_main  [ ] voronoi_fbm  [ ] voronoi_edge_distance
                                              [ ] vorolines  [ ] voronoise  [ ] vororand_main  [ ] wavelet_noise
Group D — Complex terrain:                    [ ] advection_particle  [ ] advection_warp  [ ] blend_poisson_bf
                                              [ ] flow_direction_d8  [ ] generate_riverbed  [ ] hemisphere_field
                                              [ ] interpolate_array  [ ] mountain_range_radial  [ ] normal_displacement
                                              [ ] plateau  [ ] polygon_field  [ ] sdf_2d_polyline
                                              [ ] skeleton  [ ] strata  [ ] rifts
Group E — Erosion (complex):                  [ ] thermal  [ ] thermal_inflate  [ ] thermal_rib  [ ] thermal_ridge
                                              [ ] thermal_scree  [ ] hydraulic_particle  [ ] hydraulic_schott
                                              [ ] jump_flooding  [ ] mean_shift
