/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file properties_panel.hpp
 * @brief Builds the industrial properties panel from a Meta attribute container.
 *
 * This replaces meta::qt::ContainerGroupWidget for the node settings panel.
 * Attributes are grouped by their Meta category into PpSection cards, and each
 * attribute is rendered with a Hesiod-side widget from this directory.
 *
 * Types that have not been ported yet fall back to meta::qt::render(), so the
 * panel stays fully functional while the port proceeds one widget at a time.
 */
#pragma once
#include <functional>
#include <vector>

#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include "meta/core/attribute_container.hpp"
#include "meta/core/event.hpp"

#include "hesiod/gui/widgets/properties/h_gradient.hpp"

namespace hesiod::pp
{

/**
 * @brief Host-supplied storage for user gradient presets.
 *
 * The panel has no idea where presets live - Hesiod keeps them as json under
 * data/color_gradients/<category>/ - so saving and reloading are injected.
 * Leave the callbacks empty to get a read-only preset library.
 */
struct GradientPresetStore
{
  std::function<bool(const QString               &category,
                     const QString               &name,
                     const QVector<GradientStop> &stops)>
      save;

  std::function<QVector<GradientPreset>()> reload;
};

class PropertiesPanel : public QWidget
{
  Q_OBJECT

public:
  /**
   * @param p_container  attributes to render.
   * @param initial_state  the node's initial Meta state, as produced by
   *   AttributeContainer::json_to(): `{ name: { type, value, ... } }`. Supplies
   *   each row's default, which is what the modified/default text colour is
   *   measured against. Pass an empty json to anchor defaults at the values
   *   present when the panel is built.
   */
  explicit PropertiesPanel(meta::AttributeContainer  *p_container,
                           const nlohmann::json      &initial_state = {},
                           const GradientPresetStore &preset_store = {},
                           QWidget                   *parent = nullptr);

  /// Push model values into the widgets (after a preset load, reset, undo).
  void sync_from_model();

  /// Number of attribute sections built. Callers appending their own section
  /// use this to continue the index and accent cycle.
  int section_count() const { return this->section_count_; }

  /// Append a caller-built section below the attribute sections, above the
  /// trailing stretch. The panel reparents and takes ownership. Kept type
  /// agnostic (a plain QWidget) so the panel stays free of any knowledge of
  /// what the caller is adding.
  void add_section(QWidget *section);

signals:
  void edit_started();
  void value_changed();
  void edit_ended();

private:
  void build();

  /// Build a row for one attribute, or nullptr if it should be skipped.
  QWidget *make_row(meta::AbstractAttribute *p_attr);

  QWidget *make_float_row(meta::AbstractAttribute *p_attr);
  QWidget *make_choice_row(meta::AbstractAttribute *p_attr); ///< string + allowed_values
  QWidget *make_enum_row(meta::AbstractAttribute *p_attr);   ///< int + enum_items
  QWidget *make_range_row(meta::AbstractAttribute *p_attr);  ///< glm::vec2 + RangeBar
  QWidget *make_gradient_row(meta::AbstractAttribute *p_attr); ///< meta::ColorGradient

  /// Wrap a control under an uppercase label, the way the reference stacks a
  /// label above a rail. Takes ownership of `control`.
  QWidget *make_labeled(const QString &label, QWidget *control);
  QWidget *make_int_row(meta::AbstractAttribute *p_attr);
  QWidget *make_bool_row(meta::AbstractAttribute *p_attr);

  /// Default for `name` from the initial state, or `fallback` if absent.
  double default_for(const std::string &name, double fallback) const;

  /**
   * @brief Ask for a downstream recompute, coalescing bursts.
   *
   * Attribute writes are cheap and happen immediately, but value_changed
   * drives a full node update. A 260ms glide ticks ~16 times, and a drag
   * ticks on every mouse move, so forwarding each one stalls the UI and makes
   * the animation stutter. Requests inside one window collapse into a single
   * update; flush_recompute() forces the trailing one.
   */
  void request_recompute();
  void flush_recompute();

  meta::AttributeContainer          *p_container_ = nullptr;
  nlohmann::json                     initial_state_;
  GradientPresetStore                preset_store_;
  QVBoxLayout                       *outer_ = nullptr;
  int                                section_count_ = 0;
  QTimer                            *recompute_timer_ = nullptr;
  std::vector<std::function<void()>> syncers_;
  std::vector<meta::EventConnection> connections_;
};

} // namespace hesiod::pp
