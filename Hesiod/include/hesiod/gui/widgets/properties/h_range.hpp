/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file h_range.hpp
 * @brief Dual-thumb range row with enable box, readouts and preset chips.
 *
 * Ported from HRange.qml. The property is `active`, never `enabled` - in QML
 * that name shadows Item.enabled, and keeping the same name here keeps the two
 * implementations talking about the same thing.
 *
 * Note the model quirk this has to cooperate with: Meta stores a disabled
 * range as the sentinel value (-1, 0) rather than as a separate flag, so the
 * last meaningful range must be remembered and restored on re-enable. That
 * lives in the panel binding, not here - this widget just reports state.
 */
#pragma once
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

class ModButton;

class HRange : public QWidget
{
  Q_OBJECT

public:
  explicit HRange(QWidget *parent = nullptr);

  void set_bounds(double min, double max, int decimals);
  void set_range(double lo, double hi);
  void set_active(bool active);

  double lo() const { return this->lo_; }
  double hi() const { return this->hi_; }
  bool   is_active() const { return this->active_; }

  QSize sizeHint() const override;

signals:
  /// Live during a thumb drag - repaint only, do not write the model.
  void range_changed();
  /// Gesture finished, or a preset applied: safe to commit.
  void committed();
  void active_toggled(bool active);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private:
  QRect  box_rect() const;
  QRect  rail_rect() const;
  double value_at(int x) const;
  double ratio(double v) const;
  void   layout_chips();
  void   apply_preset(double lo, double hi);

  double min_ = 0.0;
  double max_ = 1.0;
  double lo_ = 0.0;
  double hi_ = 1.0;
  int    decimals_ = 3;
  bool   active_ = true;

  int  drag_thumb_ = -1; ///< 0 = lo, 1 = hi, -1 = none
  bool box_hovered_ = false;

  ModButton *on_chip_ = nullptr;
  ModButton *full_chip_ = nullptr;
  ModButton *center_chip_ = nullptr;
  ModButton *unit_chip_ = nullptr;
};

} // namespace hesiod::pp
