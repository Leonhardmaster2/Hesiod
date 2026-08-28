/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <utility>

#include "highmap/opencl/gpu_opencl.hpp"
#include "highmap/primitives.hpp"

#include "meta/metadata/keys.hpp"

#include "hesiod/app/enum_mappings.hpp"
#include "hesiod/logger.hpp"
#include "hesiod/model/graph/metal_graph_execution.hpp"
#include "hesiod/model/nodes/attributes.hpp"
#include "hesiod/model/nodes/base_node.hpp"
#include "hesiod/model/nodes/post_process.hpp"

namespace hesiod
{

// -----------------------------------------------------------------------------
// Ports & Attributes
// -----------------------------------------------------------------------------

constexpr const char *P_DX   = "dx";
constexpr const char *P_DY   = "dy";
constexpr const char *P_CTRL = "control";
constexpr const char *P_ENV  = "envelope";
constexpr const char *P_OUT  = "output";

constexpr const char *A_NOISE_TYPE     = "noise_type";
constexpr const char *A_KW             = "kw";
constexpr const char *A_SEED           = "seed";
constexpr const char *A_OCTAVES        = "octaves";
constexpr const char *A_WEIGHT         = "weight";
constexpr const char *A_PERSISTENCE    = "persistence";
constexpr const char *A_LACUNARITY     = "lacunarity";
constexpr const char *A_PERIODIC       = "periodic";
constexpr const char *A_K_SMOOTHING    = "k_smoothing";
constexpr const char *A_GRADIENT_SCALE = "gradient_scale";
constexpr const char *A_WARP0          = "warp0";
constexpr const char *A_DAMP0          = "damp0";
constexpr const char *A_WARP_SCALE     = "warp_scale";
constexpr const char *A_DAMP_SCALE     = "damp_scale";
constexpr const char *A_MU             = "mu";

constexpr const char *G_FBM      = "FBM";
constexpr const char *G_RIDGED   = "Ridged";
constexpr const char *G_IQ       = "IQ";
constexpr const char *G_JORDAN   = "Jordan";
constexpr const char *G_PARBERRY = "Parberry";
constexpr const char *G_PINGPONG = "PingPong";
constexpr const char *G_SWISS    = "Swiss";

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------

void setup_coherent_noise_node(BaseNode &node)
{
  Logger::log()->trace("setup node {}", node.get_label());

  // --- Ports

  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_DX);
  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_DY);
  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_CTRL);
  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_ENV);
  node.add_port<hmap::VirtualArray>(gnode::PortType::OUT, P_OUT, CONFIG(node));

  {
    node.set_current_group(G_FBM);

    // clang-format off
    node.set_current_category("Noise");

    add_enum(node, A_NOISE_TYPE, "Type", enum_mappings.noise_type_map_fbm);
    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);
    add_bool(node, A_PERIODIC, "Periodic (tileable)", false);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }

  {
    node.set_current_group(G_RIDGED);

    // clang-format off
    node.set_current_category("Noise");

    add_enum(node, A_NOISE_TYPE, "Type", enum_mappings.noise_type_map_fbm);
    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);
    add_float(node, A_K_SMOOTHING, "k_smoothing", 0.2f, 0.f, 1.f);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }

  {
    node.set_current_group(G_IQ);

    // clang-format off
    node.set_current_category("Noise");

    add_enum(node, A_NOISE_TYPE, "Type", enum_mappings.noise_type_map_fbm);
    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);
    add_float(node, A_GRADIENT_SCALE, "gradient_scale", 0.05f, 0.01f, 0.1f);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }

  {
    node.set_current_group(G_JORDAN);

    // clang-format off
    node.set_current_category("Noise");

    add_enum(node, A_NOISE_TYPE, "Type", enum_mappings.noise_type_map_fbm);
    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);

    node.set_current_category("Warp");

    add_float(node, A_WARP0, "warp0", 0.2f, 0.f, 1.f);
    add_float(node, A_DAMP0, "damp0", 1.f, 0.f, 1.f);
    add_float(node, A_WARP_SCALE, "warp_scale", 0.2f, 0.f, 1.f);
    add_float(node, A_DAMP_SCALE, "damp_scale", 1.f, 0.f, 1.f);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }

  {
    node.set_current_group(G_PARBERRY);

    // clang-format off
    node.set_current_category("Noise");

    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);
    add_float(node, A_MU, "mu", 1.02f, 1.f, 1.2f);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }

  {
    node.set_current_group(G_PINGPONG);

    // clang-format off
    node.set_current_category("Noise");

    add_enum(node, A_NOISE_TYPE, "Type", enum_mappings.noise_type_map_fbm);
    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }

  {
    node.set_current_group(G_SWISS);

    // clang-format off
    node.set_current_category("Noise");

    add_enum(node, A_NOISE_TYPE, "Type", enum_mappings.noise_type_map_fbm);
    add_wavenumber(node, A_KW, "Spatial Frequency");
    add_seed(node, A_SEED, "Seed");

    node.set_current_category("FBM Layers");

    add_int(node, A_OCTAVES, "Octaves", 8, 0, 32);
    add_float(node, A_WEIGHT, "Weight", 0.7f, 0.f, 1.f);
    add_float(node, A_PERSISTENCE, "Persistence", 0.5f, 0.f, 1.f);
    add_float(node, A_LACUNARITY, "Lacunarity", 2.f, 0.01f, 4.f);
    add_float(node, A_WARP_SCALE, "warp_scale", 0.1f, 0.f, 0.5f);
    // clang-format on

    setup_post_process_heightmap_attributes(
        node,
        {.add_mix = false, .remap_active_state = true});
  }
}

// -----------------------------------------------------------------------------
// Compute
// -----------------------------------------------------------------------------

namespace
{

bool resident_noise_post_process_is_supported(const BaseNode &node)
{
  return !node.val<bool>("post_inverse") &&
         node.val<float>("post_gamma") == 1.f &&
         node.val<float>("post_gain") == 1.f &&
         node.val<float>("post_smoothing_radius") == 0.f &&
         !node.state_val<bool>("post_saturate", meta::keys::state::active);
}

bool try_resident_coherent_noise(BaseNode                   &node,
                                 hmap::VirtualArray        *p_dx,
                                 hmap::VirtualArray        *p_dy,
                                 hmap::VirtualArray        *p_ctrl,
                                 hmap::VirtualArray        *p_env,
                                 hmap::VirtualArray        *p_out,
                                 const std::string         &current_group,
                                 hmap::NoiseType             noise_type,
                                 glm::vec2                   kw,
                                 std::uint32_t               seed,
                                 int                         octaves,
                                 float                       weight,
                                 float                       persistence,
                                 float                       lacunarity)
{
  auto *execution = MetalGraphExecution::current();
  if (!execution || !execution->can_encode() || !p_out || p_env ||
      current_group != G_FBM ||
      !resident_noise_post_process_is_supported(node) ||
      !hmap::gpu::metal::supports_noise_fbm(noise_type) ||
      p_out->get_max_tiles() != glm::ivec2(1, 1))
    return false;

  const bool periodic = node.val<bool>(A_PERIODIC);
  glm::vec2 kw_local = kw;
  glm::ivec2 period(0, 0);
  if (periodic)
  {
    kw_local = glm::vec2(float(int(kw_local.x + 0.5f)),
                         float(int(kw_local.y + 0.5f)));
    period = glm::ivec2(int(kw_local.x), int(kw_local.y));
  }

  auto dx = p_dx ? execution->device_for(p_dx)
                 : hmap::gpu::metal::DeviceArray{};
  auto dy = p_dy ? execution->device_for(p_dy)
                 : hmap::gpu::metal::DeviceArray{};
  auto ctrl = p_ctrl ? execution->device_for(p_ctrl)
                     : hmap::gpu::metal::DeviceArray{};
  const auto *p_dx_device = p_dx ? &dx : nullptr;
  const auto *p_dy_device = p_dy ? &dy : nullptr;
  const auto *p_ctrl_device = p_ctrl ? &ctrl : nullptr;
  auto result = execution->session().noise_fbm(noise_type,
                                               p_out->shape,
                                               kw_local,
                                               seed,
                                               octaves,
                                               weight,
                                               persistence,
                                               lacunarity,
                                               p_ctrl_device,
                                               p_dx_device,
                                               p_dy_device,
                                               {0.f, 1.f, 0.f, 1.f},
                                               period);

  if (node.state_val<bool>("post_remap", meta::keys::state::active))
  {
    const glm::vec2 range = node.val<glm::vec2>("post_remap");
    result = execution->session().normalize(std::move(result), range.x, range.y);
  }

  execution->bind(p_out,
                  std::move(result),
                  !node.is_port_connected(P_OUT));
  execution->record_resident(node, "noise_fbm");
  return true;
}

} // namespace

void compute_coherent_noise_node(BaseNode &node)
{
  Logger::log()->trace("computing node [{}]/[{}]", node.get_label(), node.get_id());

  // --- Inputs / Outputs

  auto *p_dx   = node.get_value_ref<hmap::VirtualArray>(P_DX);
  auto *p_dy   = node.get_value_ref<hmap::VirtualArray>(P_DY);
  auto *p_ctrl = node.get_value_ref<hmap::VirtualArray>(P_CTRL);
  auto *p_env  = node.get_value_ref<hmap::VirtualArray>(P_ENV);
  auto *p_out  = node.get_value_ref<hmap::VirtualArray>(P_OUT);

  // --- Current group

  const std::optional<std::string> current_group_name = node.get_meta_group()
                                                            .current_container_name();

  if (!current_group_name)
  {
    Logger::log()->error("compute_coherent_noise_node: no group selected");
    return;
  }

  const std::string current_group = *current_group_name;

  Logger::log()->trace("compute_coherent_noise_node: current_group {}", current_group);

  // --- Common parameters

  const auto kw          = node.val<glm::vec2>(A_KW);
  const auto seed        = node.val<int>(A_SEED);
  const auto octaves     = node.val<int>(A_OCTAVES);
  const auto weight      = node.val<float>(A_WEIGHT);
  const auto persistence = node.val<float>(A_PERSISTENCE);
  const auto lacunarity  = node.val<float>(A_LACUNARITY);

  const auto resident_noise_type = current_group == G_FBM
                                       ? hmap::NoiseType(node.val<int>(A_NOISE_TYPE))
                                       : hmap::NoiseType::SIMPLEX2;
  if (try_resident_coherent_noise(node,
                                  p_dx,
                                  p_dy,
                                  p_ctrl,
                                  p_env,
                                  p_out,
                                  current_group,
                                  resident_noise_type,
                                  kw,
                                  static_cast<std::uint32_t>(seed),
                                  octaves,
                                  weight,
                                  persistence,
                                  lacunarity))
    return;

  // --- Compute

  if (current_group == G_FBM)
  {
    const auto noise_type = hmap::NoiseType(node.val<int>(A_NOISE_TYPE));
    const auto periodic   = node.val<bool>(A_PERIODIC);

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          // When periodic, snap kw to integer cells so the lattice wrap
          // aligns with the noise frequency and the result tiles
          // seamlessly.

          glm::vec2  kw_local = kw;
          glm::ivec2 period(0, 0);

          if (periodic)
          {
            kw_local = glm::vec2(float(int(kw_local.x + 0.5f)),
                                 float(int(kw_local.y + 0.5f)));

            period = glm::ivec2(int(kw_local.x), int(kw_local.y));
          }

          *pa_out = hmap::gpu::noise_fbm(noise_type,
                                         region.shape,
                                         kw_local,
                                         seed,
                                         octaves,
                                         weight,
                                         persistence,
                                         lacunarity,
                                         pa_ctrl,
                                         pa_dx,
                                         pa_dy,
                                         region.bbox,
                                         period);
        },
        node.cfg().cm_gpu);
  }
  else if (current_group == G_RIDGED)
  {
    const auto noise_type  = hmap::NoiseType(node.val<int>(A_NOISE_TYPE));
    const auto k_smoothing = node.val<float>(A_K_SMOOTHING);

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          *pa_out = hmap::noise_ridged(noise_type,
                                       region.shape,
                                       kw,
                                       seed,
                                       octaves,
                                       weight,
                                       persistence,
                                       lacunarity,
                                       k_smoothing,
                                       pa_ctrl,
                                       pa_dx,
                                       pa_dy,
                                       region.bbox);
        },
        node.cfg().cm_cpu);
  }
  else if (current_group == G_IQ)
  {
    const auto noise_type     = hmap::NoiseType(node.val<int>(A_NOISE_TYPE));
    const auto gradient_scale = node.val<float>(A_GRADIENT_SCALE);

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          *pa_out = hmap::noise_iq(noise_type,
                                   region.shape,
                                   kw,
                                   seed,
                                   octaves,
                                   weight,
                                   persistence,
                                   lacunarity,
                                   gradient_scale,
                                   pa_ctrl,
                                   pa_dx,
                                   pa_dy,
                                   region.bbox);
        },
        node.cfg().cm_cpu);
  }
  else if (current_group == G_JORDAN)
  {
    const auto noise_type = hmap::NoiseType(node.val<int>(A_NOISE_TYPE));
    const auto warp0      = node.val<float>(A_WARP0);
    const auto damp0      = node.val<float>(A_DAMP0);
    const auto warp_scale = node.val<float>(A_WARP_SCALE);
    const auto damp_scale = node.val<float>(A_DAMP_SCALE);

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          *pa_out = hmap::noise_jordan(noise_type,
                                       region.shape,
                                       kw,
                                       seed,
                                       octaves,
                                       weight,
                                       persistence,
                                       lacunarity,
                                       warp0,
                                       damp0,
                                       warp_scale,
                                       damp_scale,
                                       pa_ctrl,
                                       pa_dx,
                                       pa_dy,
                                       region.bbox);
        },
        node.cfg().cm_cpu);
  }
  else if (current_group == G_PARBERRY)
  {
    const auto mu = node.val<float>(A_MU);

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          *pa_out = hmap::noise_parberry(region.shape,
                                         kw,
                                         seed,
                                         octaves,
                                         weight,
                                         persistence,
                                         lacunarity,
                                         mu,
                                         pa_ctrl,
                                         pa_dx,
                                         pa_dy,
                                         region.bbox);
        },
        node.cfg().cm_cpu);
  }
  else if (current_group == G_PINGPONG)
  {
    const auto noise_type = hmap::NoiseType(node.val<int>(A_NOISE_TYPE));

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          *pa_out = hmap::noise_pingpong(noise_type,
                                         region.shape,
                                         kw,
                                         seed,
                                         octaves,
                                         weight,
                                         persistence,
                                         lacunarity,
                                         pa_ctrl,
                                         pa_dx,
                                         pa_dy,
                                         region.bbox);
        },
        node.cfg().cm_cpu);
  }
  else if (current_group == G_SWISS)
  {
    const auto noise_type = hmap::NoiseType(node.val<int>(A_NOISE_TYPE));
    const auto warp_scale = node.val<float>(A_WARP_SCALE);

    hmap::for_each_tile(
        {p_dx, p_dy, p_ctrl},
        {p_out},
        [&](std::vector<const hmap::Array *> in,
            std::vector<hmap::Array *>       out,
            const hmap::TileRegion          &region)
        {
          auto [pa_dx, pa_dy, pa_ctrl] = unpack<3>(in);
          auto [pa_out]                = unpack<1>(out);

          *pa_out = hmap::noise_swiss(noise_type,
                                      region.shape,
                                      kw,
                                      seed,
                                      octaves,
                                      weight,
                                      persistence,
                                      lacunarity,
                                      warp_scale,
                                      pa_ctrl,
                                      pa_dx,
                                      pa_dy,
                                      region.bbox);
        },
        node.cfg().cm_cpu);
  }
  else
  {
    Logger::log()->error("compute_coherent_noise_node: group {} not implemented",
                         current_group);
  }

  // --- Post-process (common to all groups)

  post_apply_enveloppe(node, *p_out, p_env);
  post_process_heightmap(node, *p_out);
}

} // namespace hesiod
