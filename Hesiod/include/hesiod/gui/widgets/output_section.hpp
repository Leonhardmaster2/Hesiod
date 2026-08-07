/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file output_section.hpp
 * @brief "OUTPUT" section for the properties panel: preview + export.
 *
 * One more PpSection appended below the attribute sections. It previews the
 * selected node's output ports and writes any of them to disk without needing
 * an Export* node wired into the graph.
 *
 * The industrial widget set now lives in MetaUI, which this builds on. This
 * section stays Hesiod-side because it knows about BaseNode, HighMap data
 * types and export formats - none of which belong in a general attribute
 * library. The dependency direction is Hesiod -> MetaUI, never the reverse.
 */
#pragma once
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <QImage>
#include <QWidget>

#include "hesiod/gui/widgets/data_preview.hpp" // PreviewType
#include "meta_qt/widgets/industrial/pp_section.hpp"
#include "hesiod/model/graph/graph_node.hpp"

class QLabel;
class QTimer;

namespace meta::qt
{
class HCombo;
class ModButton;
} // namespace meta::qt

namespace hesiod
{

class BaseNode;

// =====================================
// Export description
// =====================================

/// How to write one port's data. Which of these apply depends on the port's
/// data type - see export_options_for().
enum class ExportKind
{
  PNG8,  ///< 8-bit grayscale png (arrays) / 8-bit png (textures)
  PNG16, ///< 16-bit grayscale png (arrays) / 16-bit png (textures)
  EXR32, ///< 32-bit exr, preserves absolute values
  RAW16, ///< 16-bit raw, Unity heightmap layout
  CSV,   ///< point list (clouds, paths)
};

struct ExportOption
{
  QString    label;  ///< shown in the format combo
  ExportKind kind;
  QString    suffix; ///< ".png", forced onto the chosen filename
  QString    filter; ///< QFileDialog name filter
};

/// Formats valid for `data_type` (a typeid(T).name() string, as returned by
/// BaseNode::get_data_type). Empty when the type cannot be exported.
std::vector<ExportOption> export_options_for(const std::string &data_type);

/// Write port `port_index` of `p_node` to `fname`. Returns false and logs on
/// failure (no data, unsupported type, or the underlying writer threw).
bool export_port_data(BaseNode                    *p_node,
                      int                          port_index,
                      ExportKind                   kind,
                      const std::filesystem::path &fname);

/// Render one port's data to an image, or a null QImage when the port has no
/// data or its type is not previewable. Shared so the node-body DataPreview
/// can adopt it later instead of keeping its own copy of this dispatch.
QImage render_port_preview(BaseNode         *p_node,
                           int               port_index,
                           PreviewType       type,
                           const glm::ivec2 &shape);

// =====================================
// OutputPreview
// =====================================

/// Aspect-preserving preview surface. Unlike DataPreview (a fixed-size
/// QLabel), this follows the panel width - the section has to stay usable from
/// a 240px pane up to a wide docked panel.
class OutputPreview : public QWidget
{
  Q_OBJECT

public:
  explicit OutputPreview(QWidget *parent = nullptr);

  /// A null image renders the empty state rather than blanking the widget.
  void set_image(const QImage &image);

protected:
  void paintEvent(QPaintEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private:
  int height_for_width(int w) const;

  QImage image_;
  double aspect_ = 1.0; ///< w/h of the source image
};

// =====================================
// OutputSection
// =====================================

class OutputSection : public meta::qt::PpSection
{
  Q_OBJECT

public:
  /// True when the node exposes at least one OUT port, i.e. when building an
  /// OutputSection for it is worthwhile.
  static bool node_has_outputs(BaseNode *p_node);

  OutputSection(std::weak_ptr<GraphNode> p_graph_node,
                const std::string       &node_id,
                const QString           &index,
                const QColor            &accent,
                QWidget                 *parent = nullptr);

  /// Re-read the model and repaint. Called when the node finishes updating.
  void refresh();

private:
  BaseNode *node() const;

  void build_body();
  void on_port_changed();
  void rebuild_format_options();
  void do_export();
  void set_status(const QString &text, bool error);

  std::weak_ptr<GraphNode> p_graph_node;
  std::string              node_id;

  std::vector<int>     out_ports_; ///< port indices, in declaration order
  int                  current_port_ = -1;
  PreviewType          preview_type_ = PreviewType::GRAYSCALE;
  std::vector<ExportOption> formats_;

  meta::qt::HCombo    *port_combo_ = nullptr; ///< only built when >1 output
  meta::qt::HCombo    *view_combo_ = nullptr;
  meta::qt::HCombo    *format_combo_ = nullptr;
  meta::qt::ModButton *export_btn_ = nullptr;
  OutputPreview *preview_ = nullptr;
  QLabel        *caption_ = nullptr;
  QLabel        *status_ = nullptr;
  QTimer        *status_timer_ = nullptr;
  QWidget       *view_row_ = nullptr; ///< hidden for non-colormapped types
  QWidget       *format_row_ = nullptr;
};

} // namespace hesiod
