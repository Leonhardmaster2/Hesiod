/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <utility>

#include "highmap/filters.hpp"
#include "highmap/opencl/gpu_opencl.hpp"

#include "meta/metadata/keys.hpp"

#include "hesiod/model/nodes/attributes.hpp"

#include "hesiod/logger.hpp"
#include "hesiod/model/graph/metal_graph_execution.hpp"
#include "hesiod/model/nodes/base_node.hpp"
#include "hesiod/model/nodes/post_process.hpp"

namespace hesiod
{

// -----------------------------------------------------------------------------
// Ports & Attributes
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Ports & Attributes
// -----------------------------------------------------------------------------

constexpr const char *P_INPUT  = "input";
constexpr const char *P_MASK   = "mask";
constexpr const char *P_OUTPUT = "output";

constexpr const char *A_EQ   = "eq";
constexpr const char *A_RMIN = "rmin";
constexpr const char *A_RMAX = "rmax";

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------

void setup_spectral_equalizer_node(BaseNode &node)
{
  Logger::log()->trace("setup node {}", node.get_label());

  // --- Ports

  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_INPUT);
  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_MASK);
  node.add_port<hmap::VirtualArray>(gnode::PortType::OUT, P_OUTPUT, CONFIG(node));

  // --- Attributes

  std::vector<float> weights(6, 1.f);

  // clang-format off
  add_curve(node, A_EQ, "Band Weights", weights, 0.f, 2.f);
  add_float(node, A_RMIN, "Radius Min.", 0.05f, 0.f, 0.5f);
  add_float(node, A_RMAX, "Radius Max.", 0.25f, 0.f, 0.5f);
  // clang-format on

  // --- Attribute(s) order

  setup_pre_process_mask_attributes(node);
  setup_post_process_heightmap_attributes(node,
                                          {.add_mix = true, .remap_active_state = false});
}

// -----------------------------------------------------------------------------
// Compute
// -----------------------------------------------------------------------------

namespace
{

bool spectral_post_process_is_identity(const BaseNode &node,
                                       const hmap::VirtualArray *p_in)
{
  const bool mix_identity =
      !p_in || (node.val<float>("post_mix") == 1.f &&
                node.val<int>("post_mix_method") == BlendingMethod::REPLACE);
  return mix_identity && !node.val<bool>("post_inverse") &&
         node.val<float>("post_gamma") == 1.f &&
         node.val<float>("post_gain") == 1.f &&
         node.val<float>("post_smoothing_radius") == 0.f &&
         !node.state_val<bool>("post_remap", meta::keys::state::active) &&
         !node.state_val<bool>("post_saturate", meta::keys::state::active);
}

bool try_resident_spectral_equalizer(BaseNode                  &node,
                                     hmap::VirtualArray        *p_in,
                                     hmap::VirtualArray        *p_mask,
                                     hmap::VirtualArray        *p_out,
                                     const std::vector<float>  &weights,
                                     int                        ir_min,
                                     int                        ir_max)
{
  auto *execution = MetalGraphExecution::current();
  if (!execution || !execution->enabled() || !p_in || !p_out || p_mask ||
      node.is_port_connected(P_MASK) ||
      p_out->get_max_tiles() != glm::ivec2(1, 1) ||
      !spectral_post_process_is_identity(node, p_in))
    return false;

  auto input = execution->device_for(p_in);
  auto result = execution->session().spectral_equalizer(std::move(input),
                                                         weights,
                                                         ir_min,
                                                         ir_max);
  execution->bind(p_out,
                  std::move(result),
                  !node.is_port_connected(P_OUTPUT));
  execution->record_resident(node, "spectral_equalizer");
  return true;
}

} // namespace

void compute_spectral_equalizer_node(BaseNode &node)
{
  Logger::log()->trace("computing node [{}]/[{}]", node.get_label(), node.get_id());

  // --- Inputs / Outputs

  auto *p_in   = node.get_value_ref<hmap::VirtualArray>(P_INPUT);
  auto *p_mask = node.get_value_ref<hmap::VirtualArray>(P_MASK);
  auto *p_out  = node.get_value_ref<hmap::VirtualArray>(P_OUTPUT);

  if (!p_in)
    return;

  // --- Params

  // clang-format off
  const auto weights = node.val<std::vector<float>>(A_EQ);
  const auto rmin    = node.val<float>(A_RMIN);
  const auto rmax    = node.val<float>(A_RMAX);
  // clang-format on

  int ir_min = std::max(1, (int)(rmin * p_out->shape.x));
  int ir_max = std::max(1, (int)(rmax * p_out->shape.x));

  if (try_resident_spectral_equalizer(node,
                                      p_in,
                                      p_mask,
                                      p_out,
                                      weights,
                                      ir_min,
                                      ir_max))
    return;

  // --- Prepare mask

  std::shared_ptr<hmap::VirtualArray> sp_mask = pre_process_mask(node, p_mask, *p_in);

  // --- Compute

  hmap::for_each_tile(
      {p_in, p_mask},
      {p_out},
      [&](std::vector<const hmap::Array *> p_arrays_in,
          std::vector<hmap::Array *>       p_arrays_out,
          const hmap::TileRegion &)
      {
        const auto [pa_in, pa_mask] = unpack<2>(p_arrays_in);
        auto [pa_out]               = unpack<1>(p_arrays_out);

        *pa_out = hmap::gpu::spectral_equalizer(*pa_in, weights, ir_min, ir_max, pa_mask);
      },
      node.cfg().cm_gpu);

  // --- Post-process

  p_out->smooth_overlap_buffers();
  post_process_heightmap(node, *p_out, p_in);
}

} // namespace hesiod
