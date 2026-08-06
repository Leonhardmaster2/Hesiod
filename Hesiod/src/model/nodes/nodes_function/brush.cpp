/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include "meta/core/data_provider.hpp"
#include "meta/ext/array/array.hpp"
#include "meta/metadata/keys.hpp"

#include "hesiod/logger.hpp"
#include "hesiod/model/nodes/base_node.hpp"
#include "hesiod/model/nodes/compat_attributes.hpp"
#include "hesiod/model/nodes/post_process.hpp"

namespace hesiod
{

void setup_brush_node(BaseNode &node)
{
  Logger::log()->trace("setup node {}", node.get_label());

  // port(s)
  node.add_port<hmap::VirtualArray>(gnode::PortType::IN, "background");
  node.add_port<hmap::VirtualArray>(gnode::PortType::OUT, "out", CONFIG(node));

  // attribute(s)
  auto &c = node.meta_group().current();

  // the paint canvas resamples canvas->model on every stroke and model->canvas
  // on every sync; those transforms are only lossless while the canvas field
  // and the model share a shape, so keep them tied together (a smaller canvas
  // low-pass filters the painting and erodes its peaks stroke by stroke)
  constexpr int hmap_shape = 512;

  auto *a = c.add<meta::Array>("hmap",
                               meta::Array{glm::ivec2(hmap_shape, hmap_shape),
                                           std::vector<float>(hmap_shape * hmap_shape,
                                                              0.f)});
  a->metadata().try_add(meta::keys::ui::label, std::string("Heightmap"));
  a->metadata().try_add(meta::keys::ui::category, std::string("Main"));
  a->metadata().try_add(meta::keys::ui::width, hmap_shape);
  a->metadata().try_add(meta::keys::ui::height, hmap_shape);
  a->metadata().try_add(std::string(hsd::compat::keys::type_label),
                        std::string("Array"));

  // legacy .hsd files store the painting as attr::ArrayAttribute json:
  // {"shape.x": int, "shape.y": int, "vector": [float...]}
  node.register_legacy_decoder("hmap",
                               [a](const nlohmann::json &j)
                               {
                                 a->value().shape = glm::ivec2(
                                     j.at("shape.x").get<int>(),
                                     j.at("shape.y").get<int>());
                                 a->value().vector =
                                     j.at("vector").get<std::vector<float>>();
                               });

  setup_post_process_heightmap_attributes(node,
                                          {.add_mix = true, .remap_active_state = true});

  // background thumbnail behind the paint canvas (ImageData data_provider;
  // helper's meta path is attribute-generic despite the cloud name)
  setup_background_image_for_cloud_attribute(node, "hmap", "background");
}

void compute_brush_node(BaseNode &node)
{
  Logger::log()->trace("computing node [{}]/[{}]", node.get_label(), node.get_id());

  hmap::VirtualArray *p_out = node.get_value_ref<hmap::VirtualArray>("out");

  // retrieve raw data and convert them to an hmap::Array
  const auto arr = node.meta_group().current().value<meta::Array>("hmap");

  hmap::Array array(arr.shape);
  array.vector = arr.vector;
  array = array.resample_to_shape_bilinear(node.get_config_ref()->shape);

  // Array -> VirtualArray
  p_out->from_array(array, node.cfg().cm_cpu);

  // post-process
  post_process_heightmap(node, *p_out);
}

} // namespace hesiod
