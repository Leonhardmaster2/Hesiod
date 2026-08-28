/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <utility>

#include "highmap/morphology.hpp"
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
constexpr const char *P_IN  = "input";
constexpr const char *P_OUT = "output";

constexpr const char *A_RADIUS = "radius";

namespace
{

bool post_process_is_identity(const BaseNode &node)
{
  return !node.val<bool>("post_inverse") &&
         node.val<float>("post_gamma") == 1.f &&
         node.val<float>("post_gain") == 1.f &&
         node.val<float>("post_smoothing_radius") == 0.f &&
         !node.state_val<bool>("post_saturate", meta::keys::state::active);
}

std::string fallback_detail(const MetalGraphExecution *execution,
                            const hmap::VirtualArray *p_in,
                            const hmap::VirtualArray *p_out)
{
  if (!execution || !execution->enabled())
    return "MorphologicalGradient: Metal unavailable";
  if (!execution->can_encode())
    return "MorphologicalGradient: closed Metal session";
  if (!p_in || !p_out)
    return "MorphologicalGradient: missing input or output";
  if (p_in->get_max_tiles() != glm::ivec2(1, 1) ||
      p_out->get_max_tiles() != glm::ivec2(1, 1))
    return "MorphologicalGradient: multi-tile halo fallback";
  return "MorphologicalGradient: unsupported resident input/output state";
}

bool try_resident_morphological_gradient(BaseNode          &node,
                                         hmap::VirtualArray *p_in,
                                         hmap::VirtualArray *p_out,
                                         int                 ir)
{
  auto *execution = MetalGraphExecution::current();
  if (!execution || !execution->can_encode() || !p_in || !p_out ||
      p_in->get_max_tiles() != glm::ivec2(1, 1) ||
      p_out->get_max_tiles() != glm::ivec2(1, 1))
    return false;

  auto input = execution->device_for(p_in);
  auto result = execution->session().morphological_gradient(std::move(input), ir);

  const bool identity = post_process_is_identity(node);
  if (identity &&
      node.state_val<bool>("post_remap", meta::keys::state::active))
  {
    const glm::vec2 range = node.val<glm::vec2>("post_remap");
    result = execution->session().normalize(std::move(result), range.x, range.y);
  }

  execution->bind(p_out,
                  std::move(result),
                  !identity || !node.is_port_connected(P_OUT));

  if (!identity)
  {
    // The core neighborhood pass is resident, while the legacy node's
    // optional post-processing remains a deliberate host boundary. Materialize
    // before the host code mutates the VirtualArray so device state cannot
    // overwrite those edits later in the graph update.
    execution->materialize(p_out);
    post_process_heightmap(node, *p_out);
    execution->record_resident(node, "morphological_gradient + host post-process");
  }
  else
    execution->record_resident(node, "morphological_gradient");

  return true;
}

} // namespace

void setup_morphological_gradient_node(BaseNode &node)
{
  Logger::log()->trace("setup node {}", node.get_label());

  // port(s)
  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, P_IN);
  node.add_port<hmap::VirtualArray>(gnode::PortType::OUT, P_OUT, CONFIG(node));

  // attribute(s)
  add_float(node, A_RADIUS, "radius", 0.01f, 0.f, 0.05f);

  setup_post_process_heightmap_attributes(node,
                                          {.add_mix = true, .remap_active_state = true});
}

void compute_morphological_gradient_node(BaseNode &node)
{
  Logger::log()->error(
      "MorphologicalGradient node is deprecated, use MorphologyOperators node");

  Logger::log()->trace("computing node [{}]/[{}]", node.get_label(), node.get_id());

  hmap::VirtualArray *p_in = node.get_value_ref<hmap::VirtualArray>(P_IN);

  if (p_in)
  {
    hmap::VirtualArray *p_out = node.get_value_ref<hmap::VirtualArray>(P_OUT);
    if (!p_out)
      return;

    int ir = std::max(1, (int)(node.val<float>(A_RADIUS) * p_out->shape.x));

    if (try_resident_morphological_gradient(node, p_in, p_out, ir))
      return;

    if (auto *execution = MetalGraphExecution::current())
      execution->prepare_host_node(node, fallback_detail(execution, p_in, p_out));

    hmap::for_each_tile(
        {p_out, p_in},
        [ir](std::vector<hmap::Array *> p_arrays, const hmap::TileRegion &)
        {
          auto [pa_out, pa_in] = unpack<2>(p_arrays);
          *pa_out              = hmap::gpu::morphological_gradient(*pa_in, ir);
        },
        node.cfg().cm_gpu);

    p_out->smooth_overlap_buffers();

    // post-process
    post_process_heightmap(node, *p_out);
  }
}

} // namespace hesiod
