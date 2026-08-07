/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file h_gradient.hpp
 * @brief Gradient editor: preview bar, stop controls, categorised preset grid.
 *
 * A redesign of HGradient.qml rather than a straight port. Three things the
 * reference did are deliberately gone:
 *
 *   - the row of 9 interpolated dots: it restated the bar directly above it,
 *   - the hex captions under each swatch: those are preset *names*, not
 *     colours, so they read as noise,
 *   - the sideways-scrolling swatch strip: presets wrap into a grid instead.
 *
 * The bar is taller than the reference's 34px, and selection is an accent
 * border - never a fill, per the design language.
 *
 * Deliberately Qt-only: it speaks GradientStop, not meta::ColorGradient, and
 * it reaches preset storage through callbacks rather than knowing where
 * presets live. properties_panel.cpp does the conversion, the host supplies
 * the storage, and this file stays portable.
 */
#pragma once
#include <functional>

#include <QColor>
#include <QString>
#include <QVector>
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

class HCombo;
class ModButton;

struct GradientStop
{
  double pos = 0.0; ///< in [0, 1]
  QColor color;
};

struct GradientPreset
{
  QString               category; ///< "" groups under General
  QString               name;     ///< tooltip only - never painted as a caption
  QVector<GradientStop> stops;
};

class ColorChip;         // internal, defined in the .cpp
class GradientBar;       // internal
class GradientPresetGrid; // internal

class HGradient : public QWidget
{
  Q_OBJECT

public:
  explicit HGradient(QWidget *parent = nullptr);

  void                  set_stops(const QVector<GradientStop> &stops);
  QVector<GradientStop> stops() const;

  /// Presets shown in the grid, grouped by category. Names are tooltips only.
  void set_presets(const QVector<GradientPreset> &presets);

  /**
   * @brief Persist the current gradient as a named preset.
   *
   * Supplied by the host, because where presets live is not this widget's
   * business. Return false to report the write failed.
   */
  std::function<bool(const QString &category,
                     const QString &name,
                     const QVector<GradientStop> &stops)>
      on_save_preset;

  /// Re-read the preset library after a save. Host-supplied, like on_save_preset.
  std::function<QVector<GradientPreset>()> on_reload_presets;

signals:
  /// Incremental - fires continuously while a stop is dragged.
  void value_changed();
  /// Committed: drag released, colour picked, stop added/removed, preset applied.
  void edit_ended();

private:
  void refresh_categories();
  void apply_filter();
  void sync_stop_controls();
  void edit_selected_color();
  void save_current_as_preset();

  GradientBar        *bar_ = nullptr;
  GradientPresetGrid *grid_ = nullptr;
  ColorChip          *chip_ = nullptr;
  ModButton          *color_btn_ = nullptr;
  ModButton          *add_btn_ = nullptr;
  ModButton          *remove_btn_ = nullptr;
  HCombo             *category_combo_ = nullptr;
  QWidget            *category_row_ = nullptr;

  QVector<GradientPreset> presets_;  ///< everything the host handed over
  QVector<GradientPreset> filtered_; ///< what the grid is currently showing
  QStringList             categories_; ///< "All" first, then each category
};

} // namespace hesiod::pp
