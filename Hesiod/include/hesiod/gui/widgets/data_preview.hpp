/* Copyright (c) 2023 Otto Link. Distributed under the terms of the GNU General Public
   License. The full license is in the file LICENSE, distributed with this software. */
#pragma once
#include <memory>

#include <QContextMenuEvent>
#include <QLabel>

#include "hesiod/logger.hpp"
#include "hesiod/model/nodes/base_node.hpp"

namespace hesiod
{

enum PreviewType : int
{
  GRAYSCALE,
  MAGMA,
  TERRAIN,
  SLOPE_ELEVATION_HEATMAP,
  HISTOGRAM,
};

static std::map<std::string, PreviewType> preview_type_map = {
    {"cmap Grayscale", PreviewType::GRAYSCALE},
    {"cmap Magma", PreviewType::MAGMA},
    {"cmap Terrain (hillshade)", PreviewType::TERRAIN},
    {"Histogram view", PreviewType::HISTOGRAM},
    // {"Slope/elev. heatmap", PreviewType::SLOPE_ELEVATION_HEATMAP},
};

// =====================================
// DataPreview
// =====================================
class DataPreview : public QLabel
{
public:
  DataPreview() = default;
  DataPreview(std::weak_ptr<BaseNode> model, QWidget *parent = nullptr);

  const QPixmap &get_preview_pixmap() const;

  // the node context menu drives the same two choices as this widget's own
  // right-click menu, so they are reachable without hitting the thumbnail
  int  get_preview_port_index() const { return this->preview_port_index; }
  void set_preview_port_index(int port_index);

  PreviewType get_preview_type() const { return this->preview_type; }
  void        set_preview_type(PreviewType type);

public slots:
  void clear_preview();
  void update_preview();

protected:
  void contextMenuEvent(QContextMenuEvent *event) override;

private:
  std::weak_ptr<BaseNode> model;
  int                     preview_port_index;
  PreviewType             preview_type = PreviewType::GRAYSCALE;
  QPixmap                 preview_pixmap;
};

} // namespace hesiod