/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <fstream>

#include "hesiod/model/constants/color_gradient.hpp"
#include "hesiod/logger.hpp"
#include "hesiod/model/utils.hpp"

namespace hesiod
{

ColorGradientManager::ColorGradientManager()
{
  Logger::log()->info("ColorGradientManager::ColorGradientManager: initializing...");
  this->update_data();
}

std::vector<meta::Preset> ColorGradientManager::get_as_attr_presets() const
{
  std::vector<meta::Preset> presets;

  for (auto &data : this->gradient_data)
  {
    meta::Preset preset;
    // "category/name" - see the header for why the category rides along in
    // the name rather than in a field of its own
    preset.name = data.category.empty() ? data.name : data.category + "/" + data.name;
    preset.stops = {};

    for (size_t k = 0; k < data.positions.size(); ++k)
    {
      meta::Stop stop;
      stop.position = data.positions[k];
      stop.color = data.colors[k];
      preset.stops.push_back(stop);
    }

    presets.push_back(preset);
  }

  return presets;
}

void ColorGradientManager::update_data(bool append)
{
  Logger::log()->trace("ColorGradientManager::update_data");

  if (!append)
    this->gradient_data.clear();

  // list files
  const std::string path = HSD_COLOR_GRADIENTS_PATH;
  const std::string extension = ".json";

  try
  {
    // Recursive: a subdirectory of data/color_gradients IS a category, so
    // users file gradients under foresty/, rocky/, ... by moving the json
    // rather than by editing any index. Files at the root stay uncategorised.
    for (const auto &entry : std::filesystem::recursive_directory_iterator(path))
      if (entry.is_regular_file() && entry.path().extension() == extension)
      {
        const std::string fname = entry.path().string();
        const std::string label = entry.path().filename().replace_extension().string();

        Logger::log()->trace("ColorGradientManager::update_data: parsing {}", fname);

        nlohmann::json json = json_from_file(fname);

        if (json.empty())
        {
          // one unreadable preset must not cost the user the other 39
          Logger::log()->error(
              "ColorGradientManager::update_data: empty or invalid json in {}",
              fname);
          continue;
        }

        ColorGradientData grad;
        grad.name = label;

        // category = directory path relative to the root, "/"-joined
        const std::filesystem::path rel = std::filesystem::relative(
            entry.path().parent_path(),
            std::filesystem::path(path));

        if (!rel.empty() && rel != std::filesystem::path("."))
          grad.category = rel.generic_string();

        for (const auto &grad_json : json["value"])
        {
          if (!grad_json.contains("position") || !grad_json.contains("color"))
          {
            Logger::log()->error("ColorGradientManager::update_data: invalid stop in {}",
                                 grad.name);
            continue;
          }

          grad.positions.push_back(grad_json["position"].get<float>());

          const auto &color_array = grad_json["color"];
          if (!color_array.is_array() || color_array.size() < 4)
          {
            Logger::log()->error("ColorGradientManager::update_data: invalid color in {}",
                                 grad.name);
            continue;
          }

          std::array<float, 4> cf;
          for (size_t k = 0; k < 4; ++k)
            cf[k] = color_array[k].get<float>();

          grad.colors.push_back(std::move(cf));
        }

        this->gradient_data.push_back(std::move(grad));
      }
  }
  catch (const std::filesystem::filesystem_error &e)
  {
    Logger::log()->error("ColorGradientManager::update_data: folder error: {}", e.what());
    return;
  }

  // Keep categories together and alphabetical, so the panel's filter and grid
  // do not reorder themselves depending on how the filesystem enumerated.
  std::sort(this->gradient_data.begin(),
            this->gradient_data.end(),
            [](const ColorGradientData &a, const ColorGradientData &b)
            {
              if (a.category != b.category)
                return a.category < b.category;
              return a.name < b.name;
            });
}

bool ColorGradientManager::save_preset(const std::string        &category,
                                       const std::string        &name,
                                       const std::vector<float> &positions,
                                       const std::vector<std::array<float, 4>> &colors)
{
  if (name.empty() || positions.size() != colors.size() || positions.empty())
  {
    Logger::log()->error("ColorGradientManager::save_preset: nothing to write");
    return false;
  }

  // A name is about to become a filename, so anything that would escape the
  // gradients folder or break on Windows gets folded to an underscore.
  auto sanitize = [](const std::string &in)
  {
    std::string out;
    out.reserve(in.size());
    for (const char c : in)
      out += (std::string("\\/:*?\"<>|").find(c) != std::string::npos) ? '_' : c;
    return out;
  };

  try
  {
    std::filesystem::path dir = HSD_COLOR_GRADIENTS_PATH;
    if (!category.empty())
      dir /= sanitize(category);

    std::filesystem::create_directories(dir);

    const std::filesystem::path fname = dir / (sanitize(name) + ".json");

    nlohmann::json stops = nlohmann::json::array();
    for (size_t k = 0; k < positions.size(); ++k)
      stops.push_back({{"position", positions[k]},
                       {"color",
                        {colors[k][0], colors[k][1], colors[k][2], colors[k][3]}}});

    // same schema the shipped presets use, so a saved gradient is
    // indistinguishable from a built-in one on reload
    nlohmann::json json;
    json["label"] = "gradient";
    json["type"] = 3;
    json["type_string"] = "Color gradient";
    json["value"] = stops;

    std::ofstream file(fname);
    if (!file.is_open())
    {
      Logger::log()->error("ColorGradientManager::save_preset: cannot open {}",
                           fname.string());
      return false;
    }

    file << json.dump(4);
    file.close();

    Logger::log()->info("ColorGradientManager::save_preset: wrote {}", fname.string());
    return true;
  }
  catch (const std::exception &e)
  {
    Logger::log()->error("ColorGradientManager::save_preset: {}", e.what());
    return false;
  }
}

} // namespace hesiod
