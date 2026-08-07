/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <cstring>
#include <stdexcept>
#include <typeinfo>

#include <QFileDialog>
#include <QLabel>
#include <QPainter>
#include <QResizeEvent>
#include <QTimer>
#include <QVBoxLayout>

#include "highmap/colorize.hpp"
#include "highmap/export.hpp"
#include "highmap/geometry/cloud.hpp"
#include "highmap/geometry/path.hpp"
#include "highmap/tensor.hpp"

#include "hesiod/app/hesiod_application.hpp"
#include "hesiod/gui/widgets/output_section.hpp"
#include "meta_qt/widgets/industrial/h_combo.hpp"
#include "meta_qt/widgets/industrial/mod_button.hpp"
#include "meta_qt/widgets/industrial/tokens.hpp"
#include "hesiod/logger.hpp"
#include "hesiod/model/nodes/base_node.hpp"
#include "hesiod/model/utils.hpp"

namespace hesiod
{

using namespace meta::qt;

namespace
{

/// Source resolution of the cached preview image. Kept independent of the
/// widget size so dragging the panel splitter only rescales a cached QImage
/// instead of re-running colorize() on every resize event.
constexpr int kPreviewSrcW = 384;

constexpr int kPreviewMinH = 72;
constexpr int kPreviewMaxH = 360;

/// Colormaps in a deliberate order. preview_type_map is a std::map, so
/// iterating it would sort the entries alphabetically instead.
struct ViewOption
{
  const char *label;
  PreviewType type;
};

const ViewOption kViewOptions[] = {
    {"Grayscale", PreviewType::GRAYSCALE},
    {"Magma", PreviewType::MAGMA},
    {"Terrain (hillshade)", PreviewType::TERRAIN},
    {"Histogram", PreviewType::HISTOGRAM},
};

bool is_array_like(const std::string &data_type)
{
  return data_type == typeid(hmap::VirtualArray).name() ||
         data_type == typeid(hmap::Array).name();
}

/// Short, human-readable name for a port's data type.
QString pretty_type_name(const std::string &data_type)
{
  if (data_type == typeid(hmap::VirtualArray).name())
    return "Heightmap";
  if (data_type == typeid(hmap::Array).name())
    return "Array";
  if (data_type == typeid(hmap::VirtualTexture).name())
    return "Texture";
  if (data_type == typeid(hmap::Cloud).name())
    return "Cloud";
  if (data_type == typeid(hmap::Path).name())
    return "Path";
  return "Unknown";
}

/// Label above a control, matching PropertiesPanel::make_labeled.
QWidget *make_labeled(const QString &label, QWidget *control, QWidget *parent)
{
  auto *wrap = new QWidget(parent);
  auto *box = new QVBoxLayout(wrap);
  box->setContentsMargins(0, 0, 0, 0);
  box->setSpacing(6);

  auto *text = new QLabel(label, wrap);
  QFont f = text->font();
  f.setPixelSize(kLabelPx);
  f.setCapitalization(QFont::AllUppercase);
  f.setLetterSpacing(QFont::AbsoluteSpacing, 1.0);
  text->setFont(f);
  text->setStyleSheet(
      QString("color: %1; background: transparent;").arg(kInkDefault.name()));

  box->addWidget(text);
  box->addWidget(control);
  return wrap;
}

/// Resolve a VirtualArray/Array port to a concrete Array, or return false.
bool port_to_array(BaseNode *p_node, int port_index, hmap::Array &out)
{
  void             *blind_ptr = p_node->get_data_ref(port_index);
  const std::string data_type = p_node->get_data_type(port_index);

  if (!blind_ptr)
    return false;

  if (data_type == typeid(hmap::VirtualArray).name())
  {
    const auto *p_va = static_cast<const hmap::VirtualArray *>(blind_ptr);
    if (!p_va)
      return false;
    out = p_va->to_array(p_node->cfg().cm_cpu);
    return true;
  }

  if (data_type == typeid(hmap::Array).name())
  {
    const auto *p_a = static_cast<const hmap::Array *>(blind_ptr);
    if (!p_a)
      return false;
    out = *p_a;
    return true;
  }

  return false;
}

} // namespace

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

std::vector<ExportOption> export_options_for(const std::string &data_type)
{
  if (is_array_like(data_type))
    return {
        {"png (16 bit)", ExportKind::PNG16, ".png", "PNG image (*.png)"},
        {"png (8 bit)", ExportKind::PNG8, ".png", "PNG image (*.png)"},
        {"exr (32 bit)", ExportKind::EXR32, ".exr", "OpenEXR image (*.exr)"},
        {"raw (16 bit, Unity)", ExportKind::RAW16, ".raw", "Raw heightmap (*.raw)"},
    };

  if (data_type == typeid(hmap::VirtualTexture).name())
    return {
        {"png (16 bit)", ExportKind::PNG16, ".png", "PNG image (*.png)"},
        {"png (8 bit)", ExportKind::PNG8, ".png", "PNG image (*.png)"},
    };

  if (data_type == typeid(hmap::Cloud).name() ||
      data_type == typeid(hmap::Path).name())
    return {
        {"csv", ExportKind::CSV, ".csv", "Comma-separated values (*.csv)"},
    };

  return {};
}

bool export_port_data(BaseNode                    *p_node,
                      int                          port_index,
                      ExportKind                   kind,
                      const std::filesystem::path &fname)
{
  if (!p_node)
    return false;

  void             *blind_ptr = p_node->get_data_ref(port_index);
  const std::string data_type = p_node->get_data_type(port_index);

  if (!blind_ptr)
  {
    Logger::log()->error("export_port_data: port {} has no data", port_index);
    return false;
  }

  try
  {
    if (is_array_like(data_type))
    {
      hmap::Array array;
      if (!port_to_array(p_node, port_index, array))
        return false;

      switch (kind)
      {
      case ExportKind::PNG8: array.to_png_grayscale(fname.string(), CV_8U); break;
      case ExportKind::PNG16: array.to_png_grayscale(fname.string(), CV_16U); break;
      case ExportKind::EXR32: array.to_exr(fname.string()); break;
      case ExportKind::RAW16: array.to_raw_16bit(fname.string()); break;
      default:
        Logger::log()->error("export_port_data: format not valid for an array");
        return false;
      }
    }
    else if (data_type == typeid(hmap::VirtualTexture).name())
    {
      auto *p_tex = static_cast<hmap::VirtualTexture *>(blind_ptr);
      const int depth = (kind == ExportKind::PNG8) ? CV_8U : CV_16U;
      p_tex->to_png(fname.string(), p_node->cfg().cm_cpu, depth);
    }
    else if (data_type == typeid(hmap::Cloud).name())
    {
      static_cast<hmap::Cloud *>(blind_ptr)->to_csv(fname.string());
    }
    else if (data_type == typeid(hmap::Path).name())
    {
      static_cast<hmap::Path *>(blind_ptr)->to_csv(fname.string());
    }
    else
    {
      Logger::log()->error("export_port_data: unsupported data type {}", data_type);
      return false;
    }
  }
  catch (const std::exception &e)
  {
    Logger::log()->error("export_port_data: writing {} failed: {}",
                         fname.string(),
                         e.what());
    return false;
  }

  Logger::log()->info("export_port_data: wrote {}", fname.string());
  return true;
}

// ---------------------------------------------------------------------------
// Preview rendering
// ---------------------------------------------------------------------------

QImage render_port_preview(BaseNode         *p_node,
                           int               port_index,
                           PreviewType       type,
                           const glm::ivec2 &shape)
{
  if (!p_node || shape.x <= 0 || shape.y <= 0)
    return {};

  void             *blind_ptr = p_node->get_data_ref(port_index);
  const std::string data_type = p_node->get_data_type(port_index);

  if (!blind_ptr)
    return {};

  QImage::Format       img_format = QImage::Format_Grayscale8;
  std::vector<uint8_t> img;

  auto build_colored = [&](hmap::Array &array, hmap::Cmap cmap, bool normalize = true)
  {
    const float minv = array.min();
    const float maxv = array.max();
    return hmap::colorize(array, minv, maxv, cmap, normalize).to_img_8bit();
  };

  try
  {
    if (is_array_like(data_type))
    {
      hmap::Array array;

      if (data_type == typeid(hmap::VirtualArray).name())
        array = static_cast<const hmap::VirtualArray *>(blind_ptr)
                    ->to_array(shape, p_node->cfg().cm_cpu);
      else
        array = static_cast<const hmap::Array *>(blind_ptr)
                    ->resample_to_shape_nearest(shape);

      switch (type)
      {
      case PreviewType::GRAYSCALE:
        img = hmap::colorize_grayscale(array).to_img_8bit();
        img_format = QImage::Format_Grayscale8;
        break;
      case PreviewType::MAGMA:
        img = build_colored(array, hmap::Cmap::MAGMA);
        img_format = QImage::Format_RGB888;
        break;
      case PreviewType::TERRAIN:
        img = build_colored(array, hmap::Cmap::TERRAIN);
        img_format = QImage::Format_RGB888;
        break;
      case PreviewType::HISTOGRAM:
        img = hmap::colorize_histogram(array).to_img_8bit();
        img_format = QImage::Format_Grayscale8;
        break;
      case PreviewType::SLOPE_ELEVATION_HEATMAP:
        img = hmap::colorize_slope_height_heatmap(array, hmap::Cmap::HOT).to_img_8bit();
        img_format = QImage::Format_RGB888;
        break;
      }
    }
    else if (data_type == typeid(hmap::VirtualTexture).name())
    {
      img = static_cast<const hmap::VirtualTexture *>(blind_ptr)
                ->to_img_8bit(shape, p_node->cfg().cm_cpu);
      img_format = QImage::Format_RGBA8888;
    }
    else if (data_type == typeid(hmap::Cloud).name())
    {
      const auto *p_cloud = static_cast<const hmap::Cloud *>(blind_ptr);
      if (!p_cloud || p_cloud->size() == 0)
        return {};
      hmap::Array array(shape);
      p_cloud->to_array(array);
      img = build_colored(array, hmap::Cmap::MAGMA, /* normalize */ false);
      img_format = QImage::Format_RGB888;
    }
    else if (data_type == typeid(hmap::Path).name())
    {
      const auto *p_path = static_cast<const hmap::Path *>(blind_ptr);
      if (!p_path || p_path->size() == 0)
        return {};
      hmap::Array array(shape);
      glm::vec4   bbox(0.f, 1.f, 0.f, 1.f);
      hmap::Path  path = *p_path;
      path.remap_values(0.1f, 1.f);
      path.to_array(array, bbox);
      img = build_colored(array, hmap::Cmap::MAGMA, /* normalize */ false);
      img_format = QImage::Format_RGB888;
    }
    else
      return {};
  }
  catch (const std::exception &e)
  {
    Logger::log()->error("render_port_preview: {}", e.what());
    return {};
  }

  if (img.empty())
    return {};

  const size_t nchannels = (img_format == QImage::Format_Grayscale8) ? 1
                           : (img_format == QImage::Format_RGB888)   ? 3
                                                                     : 4;

  if (img.size() != static_cast<size_t>(shape.x) * shape.y * nchannels)
  {
    Logger::log()->error("render_port_preview: inconsistent image buffer size");
    return {};
  }

  QImage image(shape.x, shape.y, img_format);
  std::memcpy(image.bits(), img.data(), img.size());

  // HighMap's origin is bottom-left, Qt's is top-left.
  return image.mirrored(false, true);
}

// ---------------------------------------------------------------------------
// OutputPreview
// ---------------------------------------------------------------------------

OutputPreview::OutputPreview(QWidget *parent) : QWidget(parent)
{
  this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  this->setMinimumHeight(kPreviewMinH);
  // paintEvent fills the whole rect, so Qt need not clear it first
  this->setAttribute(Qt::WA_OpaquePaintEvent, true);
}

void OutputPreview::set_image(const QImage &image)
{
  this->image_ = image;

  if (!image.isNull() && image.height() > 0)
    this->aspect_ = static_cast<double>(image.width()) / image.height();

  // The aspect may have changed with the port, so re-derive the height.
  const int h = this->height_for_width(this->width());
  if (h != this->height())
    this->setFixedHeight(h);

  this->update();
}

int OutputPreview::height_for_width(int w) const
{
  if (w <= 0 || this->aspect_ <= 0.0)
    return kPreviewMinH;

  const int h = static_cast<int>(std::lround(w / this->aspect_));
  return qBound(kPreviewMinH, h, kPreviewMaxH);
}

void OutputPreview::resizeEvent(QResizeEvent *event)
{
  QWidget::resizeEvent(event);

  // Only a width change can change the derived height. Reacting to a
  // height-only resize calls setFixedHeight from inside the layout pass that
  // just resized us, which invalidates that layout and makes it run again -
  // so one animation frame turns into several full layout passes of the whole
  // panel, and the collapse animation drops frames.
  if (event->oldSize().width() == event->size().width())
    return;

  // Height follows width so the preview grows with the panel. setFixedHeight
  // re-triggers a resize, but the next pass computes the same value and stops.
  const int h = this->height_for_width(this->width());
  if (h != this->height())
    this->setFixedHeight(h);
}

void OutputPreview::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::SmoothPixmapTransform, true);

  const QRect r = this->rect().adjusted(0, 0, -1, -1);

  p.fillRect(this->rect(), kPadSurface);

  if (this->image_.isNull())
  {
    QFont f = this->font();
    f.setPixelSize(kLabelPx);
    f.setCapitalization(QFont::AllUppercase);
    f.setLetterSpacing(QFont::AbsoluteSpacing, 1.0);
    p.setFont(f);
    p.setPen(kInkLocked);
    p.drawText(this->rect(), Qt::AlignCenter, "no data");
  }
  else
  {
    const QImage scaled = this->image_.scaled(this->size(),
                                              Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation);
    const QPoint at((this->width() - scaled.width()) / 2,
                    (this->height() - scaled.height()) / 2);
    p.drawImage(at, scaled);
  }

  p.setPen(kFieldBorder);
  p.setBrush(Qt::NoBrush);
  p.drawRect(r);
}

// ---------------------------------------------------------------------------
// OutputSection
// ---------------------------------------------------------------------------

bool OutputSection::node_has_outputs(BaseNode *p_node)
{
  if (!p_node)
    return false;

  for (int k = 0; k < p_node->get_nports(); ++k)
    if (p_node->get_port_type(k) == gngui::PortType::OUT)
      return true;

  return false;
}

OutputSection::OutputSection(std::weak_ptr<GraphNode> p_graph_node,
                             const std::string       &node_id,
                             const QString           &index,
                             const QColor            &accent,
                             QWidget                 *parent)
    : PpSection("Output", index, accent, parent), p_graph_node(p_graph_node),
      node_id(node_id)
{
  Logger::log()->trace("OutputSection::OutputSection: node {}", node_id);

  this->build_body();
}

BaseNode *OutputSection::node() const
{
  auto gno = this->p_graph_node.lock();
  if (!gno)
    return nullptr;

  return gno->get_node_ref_by_id<BaseNode>(this->node_id);
}

void OutputSection::build_body()
{
  BaseNode *p_node = this->node();
  if (!p_node)
    return;

  for (int k = 0; k < p_node->get_nports(); ++k)
    if (p_node->get_port_type(k) == gngui::PortType::OUT)
      this->out_ports_.push_back(k);

  if (this->out_ports_.empty())
    return;

  this->current_port_ = this->out_ports_.front();

  QVBoxLayout *body = this->body_layout();

  // --- output selector, only when there is a choice to make

  if (this->out_ports_.size() > 1)
  {
    this->port_combo_ = new HCombo(this);

    QStringList names;
    for (int k : this->out_ports_)
      names << QString::fromStdString(p_node->get_port_caption(k));

    this->port_combo_->set_options(names);
    this->port_combo_->set_current(0);

    this->connect(this->port_combo_,
                  &HCombo::activated,
                  this,
                  [this](int i)
                  {
                    if (i >= 0 && i < static_cast<int>(this->out_ports_.size()))
                    {
                      this->current_port_ = this->out_ports_[i];
                      this->on_port_changed();
                    }
                  });

    body->addWidget(make_labeled("Output", this->port_combo_, this));
  }

  // --- preview

  this->preview_ = new OutputPreview(this);
  body->addWidget(this->preview_);

  this->caption_ = new QLabel(this);
  {
    QFont f = this->caption_->font();
    f.setPixelSize(kIndexPx);
    f.setFamily(mono_family());
    this->caption_->setFont(f);
    this->caption_->setStyleSheet(
        QString("color: %1; background: transparent;").arg(kInkLocked.name()));
  }
  body->addWidget(this->caption_);

  // --- colormap

  this->view_combo_ = new HCombo(this);
  {
    QStringList names;
    for (const auto &v : kViewOptions)
      names << v.label;
    this->view_combo_->set_options(names);
    this->view_combo_->set_current(0);
  }

  this->connect(this->view_combo_,
                &HCombo::activated,
                this,
                [this](int i)
                {
                  const int count = static_cast<int>(std::size(kViewOptions));
                  if (i >= 0 && i < count)
                  {
                    this->preview_type_ = kViewOptions[i].type;
                    this->refresh();
                  }
                });

  this->view_row_ = make_labeled("View", this->view_combo_, this);
  body->addWidget(this->view_row_);

  // --- export

  this->format_combo_ = new HCombo(this);
  this->format_row_ = make_labeled("Format", this->format_combo_, this);
  body->addWidget(this->format_row_);

  this->export_btn_ = new ModButton("Export...", /* checkable */ false, kAccent, this);
  this->export_btn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  this->export_btn_->setMinimumHeight(30);
  this->connect(this->export_btn_, &ModButton::clicked, this, &OutputSection::do_export);
  body->addWidget(this->export_btn_);

  this->status_ = new QLabel(this);
  this->status_->setWordWrap(true);
  {
    QFont f = this->status_->font();
    f.setPixelSize(kIndexPx);
    this->status_->setFont(f);
  }
  this->status_->hide();
  body->addWidget(this->status_);

  this->status_timer_ = new QTimer(this);
  this->status_timer_->setSingleShot(true);
  this->connect(this->status_timer_,
                &QTimer::timeout,
                this,
                [this]() { this->status_->hide(); });

  this->on_port_changed();
}

void OutputSection::on_port_changed()
{
  this->rebuild_format_options();
  this->refresh();
}

void OutputSection::rebuild_format_options()
{
  BaseNode *p_node = this->node();
  if (!p_node || this->current_port_ < 0 || !this->format_combo_)
    return;

  const std::string data_type = p_node->get_data_type(this->current_port_);

  this->formats_ = export_options_for(data_type);

  QStringList names;
  for (const auto &opt : this->formats_)
    names << opt.label;

  this->format_combo_->set_options(names);
  this->format_combo_->set_current(0);

  // A type with no writer gets no format row and no button, rather than a
  // button that always fails.
  const bool exportable = !this->formats_.empty();
  this->format_row_->setVisible(exportable);
  this->export_btn_->setVisible(exportable);

  // The colormap choice only means anything for scalar fields; a texture is
  // already RGBA and clouds/paths are rasterised one fixed way.
  this->view_row_->setVisible(is_array_like(data_type));
}

void OutputSection::refresh()
{
  BaseNode *p_node = this->node();
  if (!p_node || !this->preview_ || this->current_port_ < 0)
    return;

  const glm::ivec2 &data_shape = p_node->cfg().shape;
  const double      aspect = (data_shape.y > 0)
                                 ? static_cast<double>(data_shape.x) / data_shape.y
                                 : 1.0;

  const glm::ivec2 shape(kPreviewSrcW,
                         std::max(1, static_cast<int>(std::lround(kPreviewSrcW / aspect))));

  this->preview_->set_image(
      render_port_preview(p_node, this->current_port_, this->preview_type_, shape));

  if (this->caption_)
  {
    const std::string data_type = p_node->get_data_type(this->current_port_);
    this->caption_->setText(QString("%1  %2 x %3")
                                .arg(pretty_type_name(data_type))
                                .arg(data_shape.x)
                                .arg(data_shape.y));
  }
}

void OutputSection::set_status(const QString &text, bool error)
{
  if (!this->status_)
    return;

  this->status_->setText(text);
  this->status_->setStyleSheet(QString("color: %1; background: transparent;")
                                   .arg(error ? QString("#c06478") : kAccent.name()));
  this->status_->show();
  this->status_timer_->start(6000);
}

void OutputSection::do_export()
{
  BaseNode *p_node = this->node();
  if (!p_node || this->current_port_ < 0 || this->formats_.empty())
    return;

  const int idx = this->format_combo_ ? this->format_combo_->current() : 0;
  if (idx < 0 || idx >= static_cast<int>(this->formats_.size()))
    return;

  const ExportOption &opt = this->formats_[idx];

  const QString suggested = QString::fromStdString(p_node->get_label()) + opt.suffix;

  const QString picked = QFileDialog::getSaveFileName(this,
                                                      "Export output",
                                                      suggested,
                                                      opt.filter);
  if (picked.isNull() || picked.isEmpty())
    return;

  const std::filesystem::path fname = ensure_extension(
      std::filesystem::path(picked.toStdString()),
      opt.suffix.toStdString());

  if (export_port_data(p_node, this->current_port_, opt.kind, fname))
    this->set_status(
        QString("Exported to %1")
            .arg(QString::fromStdString(fname.filename().string())),
        /* error */ false);
  else
    this->set_status("Export failed - see the log for details", /* error */ true);
}

} // namespace hesiod
