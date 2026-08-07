/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file param_slider.hpp
 * @brief Industrial parameter row: label left, accent-filled rail, value box.
 *
 * Ported from ParamSlider.qml in `.claude/skills/hesiod-ui/assets/`. The
 * interaction contract is part of the design, not decoration:
 *
 *   - a click glides 260ms OutCubic to the clicked value, it never snaps;
 *   - a press only becomes a drag after 6px of travel, so plain clicks always
 *     glide and double-click resets survive small hand jitter;
 *   - double-click anywhere on the row glides back to the default;
 *   - the wheel steps by `step` (x10 with Shift);
 *   - clicking the value box types a number; Enter commits, Esc cancels.
 *
 * The rail fill is ALWAYS the accent. Only text encodes state.
 */
#pragma once
#include <QColor>
#include <QElapsedTimer>
#include <QLineEdit>
#include <QPointer>
#include <QVariantAnimation>
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

class ParamSlider : public QWidget
{
  Q_OBJECT

public:
  explicit ParamSlider(QWidget *parent = nullptr);

  double value() const { return this->value_; }
  double default_value() const { return this->default_value_; }
  bool   is_locked() const { return this->locked_; }

  /// True when the value differs from its default - drives the text colour.
  bool is_modified() const;

  void set_label(const QString &label);
  void set_range(double from, double to, double step);

  /**
   * @brief Round dragged/typed values to whole `step` increments.
   *
   * Off by default. In Meta, `constraints.step` is the SPIN BOX increment, not
   * a quantisation grid - a float in [0,1] can carry step 1.0, and snapping to
   * that leaves the rail able to reach only its endpoints. Integers want it on;
   * floats do not. The wheel and arrow keys use `step` either way.
   */
  void set_quantized(bool quantized);
  void set_decimals(int decimals);
  void set_suffix(const QString &suffix);
  void set_accent(const QColor &accent);
  void set_default_value(double v);
  void set_locked(bool locked);

  /// Assign without animating. Use for model -> widget sync so an external
  /// change never fights a running glide.
  void set_value(double v);

  /// Assign with the 260ms glide. Use for every user-driven change.
  void glide_to(double v);

  QSize sizeHint() const override;

signals:
  void edit_started();
  void value_changed();
  void edit_ended();

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseDoubleClickEvent(QMouseEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void keyPressEvent(QKeyEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

private:
  QRect  label_rect() const;
  QRect  rail_rect() const;   ///< full interactive rail zone
  QRect  groove_rect() const; ///< the 6px well inside the rail zone
  QRect  field_rect() const;
  double ratio() const;
  double value_at(int x) const;
  void   set_value_snapped(double v); ///< rounds to step, stops any glide
  void   begin_edit();
  void   commit_edit();
  QColor ink() const;
  QString display_text() const;

  QString label_;
  QString suffix_{"%"};
  double  value_ = 50.0;
  double  from_ = 0.0;
  double  to_ = 100.0;
  double  step_ = 1.0;
  double  default_value_ = 50.0;
  int     decimals_ = 0;
  bool    quantized_ = false;
  QColor  accent_ = QColor("#cfa143");
  bool    locked_ = false;

  bool          dragging_ = false;
  bool          pressed_ = false;
  int           press_x_ = 0;
  bool          hovered_ = false;
  bool          field_hovered_ = false;
  QElapsedTimer click_timer_;
  int           last_click_x_ = -1000;

  QVariantAnimation  *glide_ = nullptr;
  QPointer<QLineEdit> editor_;
};

} // namespace hesiod::pp
