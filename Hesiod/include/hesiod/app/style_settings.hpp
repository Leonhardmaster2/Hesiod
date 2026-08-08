/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General Public
   License. The full license is in the file LICENSE, distributed with this software. */
#pragma once
#include <string>

#include <QApplication>
#include <QColor>

#include "highmap/array.hpp"
#include "highmap/geometry/cloud.hpp"
#include "highmap/geometry/path.hpp"
#include "highmap/virtual_array/virtual_array.hpp"
#include "highmap/virtual_array/virtual_texture.hpp"

#include "nlohmann/json.hpp"

namespace hesiod
{

struct StyleSettings
{
  StyleSettings() = default;

  // --- Serialization
  void           json_from(nlohmann::json const &json);
  nlohmann::json json_to() const;

  // --- Data
  // Port / link colours, one distinct hue per data type so a pin's colour
  // says what it carries. Muted accents from the industrial design language
  // rather than saturated primaries: they sit on the #2e2e2e node body
  // without glowing, and stay separable at small port sizes.
  //
  // VirtualArray (the heightmap that most ports carry) used to be near-white,
  // which made the whole graph read monochrome and defeated the point of
  // colouring pins at all. It is the steel blue now.
  std::map<std::string, QColor> data_color_map = {
      {typeid(hmap::VirtualArray).name(), QColor("#7d9cc0")},   // heightmap
      {typeid(hmap::Array).name(), QColor("#3aa899")},          // array / mask
      {typeid(hmap::VirtualTexture).name(), QColor("#a08bb8")}, // texture
      {typeid(hmap::Cloud).name(), QColor("#cfa143")},          // point cloud
      {typeid(hmap::Path).name(), QColor("#c06478")},           // path
      {typeid(std::vector<float>).name(), QColor("#8fa96b")},   // scalar list
  };

  std::map<std::string, QColor> category_color_map = {
      {"Converter", QColor(188, 182, 163, 255)},
      {"Comment", QColor(170, 170, 170, 255)},
      {"Debug", QColor(200, 0, 0, 255)},
      {"Math", QColor(0, 43, 54, 255)},
      {"Geometry", QColor(101, 123, 131, 255)},
      {"Roads", QColor(147, 161, 161, 255)},
      {"Routing", QColor(188, 182, 163, 255)},
      {"IO", QColor(203, 196, 177, 255)},
      {"Features", QColor(181, 137, 0, 255)},
      {"Erosion", QColor(203, 75, 22, 255)},
      {"Mask", QColor(211, 54, 130, 255)},
      {"Filter", QColor(108, 113, 196, 255)},
      {"Operator", QColor(108, 113, 196, 255)},
      {"Hydrology", QColor(38, 139, 210, 255)},
      {"Primitive", QColor(42, 161, 152, 255)},
      {"Biomes", QColor(133, 153, 0, 255)},
      {"Texture", QColor(0, 0, 0, 255)},
      {"WIP", QColor(255, 255, 255, 255)},
  };
};

} // namespace hesiod