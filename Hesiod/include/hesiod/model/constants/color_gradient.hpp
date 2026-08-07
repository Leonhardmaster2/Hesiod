/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General Public
   License. The full license is in the file LICENSE, distributed with this software. */
#pragma once
#include <map>
#include <string>
#include <vector>

#include "meta/ext/color_gradient/color_gradient.hpp"

#define HSD_COLOR_GRADIENT_FILE "data/color_gradient.json"
#define HSD_COLOR_GRADIENTS_PATH "data/color_gradients"

namespace hesiod
{

struct ColorGradientData
{
  std::string                       name = "";
  /// Subdirectory the file was found in; empty for files at the root.
  std::string                       category = "";
  std::vector<float>                positions = {};
  std::vector<std::array<float, 4>> colors = {};
};

class ColorGradientManager
{
public:
  // get the singleton instance
  static ColorGradientManager &get_instance()
  {
    static ColorGradientManager instance;
    return instance;
  }

  /**
   * @brief Presets for a gradient attribute's ui.presets metadata.
   *
   * meta::Preset has no category field, and adding one would mean editing the
   * Meta submodule, so a categorised preset is named "category/name". The
   * properties panel splits it back apart; Meta's own renderer shows the full
   * path, which is harmless.
   */
  std::vector<meta::Preset> get_as_attr_presets() const;

  void update_data(bool append = false);

  /**
   * @brief Write a user gradient to data/color_gradients/<category>/<name>.json.
   *
   * Creates the category directory if needed. Returns false and logs if the
   * file cannot be written. Does NOT reload: call update_data() after.
   */
  bool save_preset(const std::string                       &category,
                   const std::string                       &name,
                   const std::vector<float>                &positions,
                   const std::vector<std::array<float, 4>> &colors);

private:
  // private constructor
  ColorGradientManager();

  // delete copy constructor and assignment operator to enforce singleton
  ColorGradientManager(const ColorGradientManager &) = delete;
  ColorGradientManager &operator=(const ColorGradientManager &) = delete;

  // store colormap
  std::vector<ColorGradientData> gradient_data;
};

} // namespace hesiod
