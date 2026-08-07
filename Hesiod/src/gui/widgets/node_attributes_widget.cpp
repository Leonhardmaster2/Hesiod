/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <fstream>

#include <QDesktopServices>
#include <QFileDialog>
#include <QLayout>
#include <QStyle>
#include <QToolButton>

#include "hesiod/app/hesiod_application.hpp"
#include "hesiod/gui/widgets/documentation_popup.hpp"
#include "hesiod/gui/widgets/node_attributes_widget.hpp"
#include "hesiod/gui/widgets/output_section.hpp"
#include "hesiod/logger.hpp"
#include "hesiod/model/constants/color_gradient.hpp"
#include "hesiod/model/nodes/base_node.hpp"

namespace hesiod
{

NodeAttributesWidget::NodeAttributesWidget(std::weak_ptr<GraphNode>  p_graph_node,
                                           const std::string        &node_id,
                                           QPointer<GraphNodeWidget> p_graph_node_widget,
                                           bool                      add_toolbar,
                                           QWidget                  *parent)
    : QWidget(parent), p_graph_node(p_graph_node), node_id(node_id),
      p_graph_node_widget(p_graph_node_widget), add_toolbar(add_toolbar)
{
  Logger::log()->trace("NodeAttributesWidget::NodeAttributesWidget: node {}", node_id);

  this->setAttribute(Qt::WA_DeleteOnClose);

  this->setup_layout();
}

QWidget *NodeAttributesWidget::create_toolbar()
{
  Logger::log()->trace("NodeAttributesWidget::create_toolbar");

  QWidget     *toolbar = new QWidget(this);
  QHBoxLayout *layout = new QHBoxLayout(toolbar);
  layout->setContentsMargins(0, 0, 0, 6);
  layout->setSpacing(8);

  auto make_button = [&](const QIcon &icon, const QString &tooltip)
  {
    QToolButton *btn = new QToolButton;
    // #ppToolButton is styled by properties_panel_style.cpp; without the
    // objectName the toolbar renders stock next to the restyled panel.
    btn->setObjectName("ppToolButton");
    btn->setToolTip(tooltip);
    btn->setIcon(icon);
    btn->setIconSize(QSize(16, 16));
    btn->setFixedSize(34, 34);
    return btn;
  };

  auto *update_btn = make_button(HSD_ICON("refresh"), "Force Update");
  auto *info_btn = make_button(HSD_ICON("info"), "Node Information");
  auto *bckp_btn = make_button(HSD_ICON("bookmark"), "Backup State");
  auto *revert_btn = make_button(HSD_ICON("u_turn_left"), "Revert State");
  auto *load_btn = make_button(HSD_ICON("file_open"), "Load Preset");
  auto *save_btn = make_button(HSD_ICON("save"), "Save Preset");
  auto *reset_btn = make_button(HSD_ICON("settings_backup_restore"), "Reset Settings");
  auto *help_btn = make_button(HSD_ICON("help"), "Help!");
  auto *doc_btn = make_button(HSD_ICON("link"), "Online Documentation");

  for (auto *btn : {update_btn,
                    info_btn,
                    bckp_btn,
                    revert_btn,
                    load_btn,
                    save_btn,
                    reset_btn,
                    help_btn,
                    doc_btn})
    layout->addWidget(btn);

  // layout->addStretch();

  // --- connections

  // use node id + graph_node instead of the node pointer for safety
  // (no lifetime warranty on p_node)
  this->connect(update_btn,
                &QToolButton::pressed,
                [this]()
                {
                  auto gno = this->p_graph_node.lock();
                  if (!gno)
                    return;

                  gno->update(this->node_id);
                });

  this->connect(info_btn,
                &QToolButton::pressed,
                [this]()
                {
                  auto gno = this->p_graph_node.lock();
                  if (!gno)
                    return;

                  if (this->p_graph_node_widget)
                    this->p_graph_node_widget->on_node_info(this->node_id);
                });

  // State/preset buttons operate on the Meta container json (snapshot manager
  // for state, json_to/json_from for presets).
  auto meta_container = [this]() -> meta::AttributeContainer *
  {
    auto gno = this->p_graph_node.lock();
    if (!gno)
      return nullptr;
    BaseNode *p_node = gno->get_node_ref_by_id<BaseNode>(this->node_id);
    if (!p_node)
      return nullptr;
    return &p_node->get_meta_group().current();
  };

  this->connect(bckp_btn,
                &QToolButton::pressed,
                [this, meta_container]()
                {
                  if (auto *c = meta_container())
                    c->snapshot_manager().save("user_state", c->json_to());
                });

  this->connect(revert_btn,
                &QToolButton::pressed,
                [this, meta_container]()
                {
                  if (auto *c = meta_container())
                  {
                    if (c->snapshot_manager().has("user_state"))
                    {
                      c->json_from(c->snapshot_manager().load("user_state"), true);
                      this->sync_from_model();
                      if (auto gno = this->p_graph_node.lock())
                        gno->update(this->node_id);
                    }
                  }
                });

  this->connect(load_btn,
                &QToolButton::pressed,
                [this, meta_container]()
                {
                  auto *c = meta_container();
                  if (!c)
                    return;

                  QString fname = QFileDialog::getOpenFileName(nullptr,
                                                               "preset.json",
                                                               ".",
                                                               "json file (*.json)");

                  if (!fname.isNull() && !fname.isEmpty())
                  {
                    std::ifstream file(fname.toStdString());

                    if (file.is_open())
                    {
                      try
                      {
                        nlohmann::json json;
                        file >> json;
                        file.close();
                        Logger::log()->trace("JSON successfully loaded from {}",
                                             fname.toStdString());

                        c->json_from(json, true);
                        this->sync_from_model();
                        if (auto gno = this->p_graph_node.lock())
                          gno->update(this->node_id);
                      }
                      catch (const std::exception &e)
                      {
                        Logger::log()->error("Failed to load preset {}: {}",
                                             fname.toStdString(),
                                             e.what());
                      }
                    }
                    else
                      Logger::log()->error("Could not open file {} to load JSON",
                                           fname.toStdString());
                  }
                });

  this->connect(save_btn,
                &QToolButton::pressed,
                [this, meta_container]()
                {
                  auto *c = meta_container();
                  if (!c)
                    return;

                  QString fname = QFileDialog::getSaveFileName(nullptr,
                                                               "preset.json",
                                                               ".",
                                                               "json file (*.json)");

                  if (!fname.isNull() && !fname.isEmpty())
                  {
                    std::ofstream file(fname.toStdString());

                    if (file.is_open())
                    {
                      file << c->json_to().dump(4);
                      file.close();
                    }
                    else
                      Logger::log()->error("Could not open file {} to save JSON",
                                           fname.toStdString());
                  }
                });

  this->connect(reset_btn,
                &QToolButton::pressed,
                [this, meta_container]()
                {
                  auto gno = this->p_graph_node.lock();
                  if (!gno)
                    return;
                  BaseNode *p_node = gno->get_node_ref_by_id<BaseNode>(this->node_id);
                  if (!p_node)
                    return;

                  auto *c = meta_container();
                  if (c && !p_node->iinitial_meta_state().empty())
                  {
                    c->json_from(p_node->iinitial_meta_state(), true);
                    this->sync_from_model();
                    gno->update(this->node_id);
                  }
                });

  this->connect(help_btn,
                &QToolButton::pressed,
                [this]()
                {
                  auto gno = this->p_graph_node.lock();
                  if (!gno)
                    return;

                  if (auto *p_node = gno->get_node_ref_by_id<BaseNode>(this->node_id))
                  {
                    auto *popup = new DocumentationPopup(
                        p_node->get_label(),
                        p_node->get_documentation_html());
                    popup->setAttribute(Qt::WA_DeleteOnClose);
                    popup->show();
                  }
                });

  this->connect(
      doc_btn,
      &QToolButton::pressed,
      [this]()
      {
        auto gno = this->p_graph_node.lock();
        if (!gno)
          return;

        if (auto *p_node = gno->get_node_ref_by_id<BaseNode>(this->node_id))
        {
          std::string
              url = "https://hesioddoc.readthedocs.io/en/latest/node_reference/nodes/" +
                    p_node->get_label();
          QDesktopServices::openUrl(QUrl(url.c_str()));
        }
      });

  return toolbar;
}

void NodeAttributesWidget::sync_from_model()
{
  if (this->props_panel)
    this->props_panel->sync_from_model();

  // the node just recomputed, so the preview is stale
  if (this->output_section)
    this->output_section->refresh();
}

bool NodeAttributesWidget::is_meta_backed() const { return this->props_panel != nullptr; }

void NodeAttributesWidget::setup_layout()
{
  Logger::log()->trace("NodeAttributesWidget::setup_layout");

  auto gno = this->p_graph_node.lock();
  if (!gno)
    return;

  BaseNode *p_node = gno->get_node_ref_by_id<BaseNode>(this->node_id);
  if (!p_node)
    return;

  // --- main layout (built once)
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setSpacing(8);
  main_layout->setContentsMargins(0, 0, 0, 0);

  if (this->add_toolbar)
    main_layout->addWidget(this->create_toolbar());

  // --- Hesiod industrial properties panel
  //
  // Replaces meta::qt::ContainerGroupWidget. The Meta widgets are palette- and
  // stylesheet-driven, which cannot express the design language: the rail fill,
  // the machined thumb, per-group accents and the modified/default/locked text
  // rule are all painted, not styled. PropertiesPanel walks the same attribute
  // container and builds custom-painted rows instead, falling back to
  // meta::qt::render() for attribute types not ported yet.
  //
  // iinitial_meta_state() supplies each row's default, so the white/grey
  // "modified" text actually means changed-from-default rather than
  // changed-since-the-panel-opened.
  // Gradient presets live in data/color_gradients/<category>/. The panel is
  // kept ignorant of that: it gets a save and a reload callback, so the
  // properties widgets stay free of hesiod/model includes.
  meta::qt::GradientPresetStore preset_store;

  preset_store.save = [](const QString                     &category,
                         const QString                     &name,
                         const QVector<meta::qt::GradientStop>   &stops)
  {
    std::vector<float>                positions;
    std::vector<std::array<float, 4>> colors;
    positions.reserve(static_cast<size_t>(stops.size()));
    colors.reserve(static_cast<size_t>(stops.size()));

    for (const auto &s : stops)
    {
      positions.push_back(static_cast<float>(s.pos));
      colors.push_back({static_cast<float>(s.color.redF()),
                        static_cast<float>(s.color.greenF()),
                        static_cast<float>(s.color.blueF()),
                        static_cast<float>(s.color.alphaF())});
    }

    auto &mgr = ColorGradientManager::get_instance();
    if (!mgr.save_preset(category.toStdString(), name.toStdString(), positions, colors))
      return false;

    mgr.update_data();
    return true;
  };

  preset_store.reload = []()
  {
    QVector<meta::qt::GradientPreset> out;

    for (const auto &p : ColorGradientManager::get_instance().get_as_attr_presets())
    {
      QString   qname = QString::fromStdString(p.name);
      QString   category;
      const int slash = qname.lastIndexOf('/');

      if (slash >= 0)
      {
        category = qname.left(slash);
        qname = qname.mid(slash + 1);
      }

      QVector<meta::qt::GradientStop> stops;
      stops.reserve(static_cast<int>(p.stops.size()));
      for (const auto &s : p.stops)
        stops.push_back({static_cast<double>(s.position),
                         QColor::fromRgbF(s.color[0], s.color[1], s.color[2],
                                          s.color[3])});

      out.push_back({category, qname, stops});
    }

    return out;
  };

  this->props_panel = new meta::qt::PropertiesPanel(&p_node->get_meta_group().current(),
                                              p_node->iinitial_meta_state(),
                                              preset_store,
                                              this);

  // "Enable live update during editing", from the app settings. A node update
  // runs synchronously on the GUI thread, so forwarding every incremental
  // change makes a drag stutter: the panel's debounce only waits for the drag
  // to PAUSE, and a real drag pauses constantly. With this off the recompute
  // happens once, on release.
  this->props_panel->set_live_update(HSD_CTX.app_settings.node_editor.live_update);

  // The section restyling hack that used to live here is gone: PpSection paints
  // its own header, so there is nothing left to patch up after the fact.

  // Recompute continuously on value_changed: the panel syncs from the model
  // (sync_from_model()) instead of being rebuilt on update_finished, so
  // recomputing on every value_changed no longer destroys a live-dragged widget
  // mid-drag.
  this->connect(this->props_panel,
                &meta::qt::PropertiesPanel::value_changed,
                this,
                [this]()
                {
                  auto gno = this->p_graph_node.lock();
                  if (!gno)
                    return;
                  gno->update(this->node_id);
                });

  // --- OUTPUT section
  //
  // Appended into the panel's own stack so it reads as one more category,
  // continuing the index and accent cycle. It previews each of the node's
  // output ports and can write any of them to disk directly, which saves
  // wiring an Export* node just to look at or dump a result.
  if (OutputSection::node_has_outputs(p_node))
  {
    const int     index = this->props_panel->section_count() + 1;
    const QString idx = QString("%1").arg(index, 2, 10, QChar('0'));

    this->output_section = new OutputSection(this->p_graph_node,
                                             this->node_id,
                                             idx,
                                             meta::qt::group_accent(index - 1),
                                             this->props_panel);

    this->props_panel->add_section(this->output_section);
  }

  main_layout->addWidget(this->props_panel);
}

} // namespace hesiod
