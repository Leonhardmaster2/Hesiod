/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file check_row.hpp
 * @brief Boolean rows: a square switch for modes, a square box for flags.
 *
 * Ported from CheckRow.qml and CheckBoxRow.qml. Two shapes on purpose - a
 * switch reads as "this changes how the node behaves", a check box reads as
 * "this is one more option". Both are square; the design language has no
 * pills.
 */
#pragma once
#include <QColor>
#include <QVariantAnimation>
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

/// Label left, 36x18 switch right. Use for modes.
class CheckRow : public QWidget
{
  Q_OBJECT

public:
  explicit CheckRow(const QString &label,
                    bool           checked = false,
                    const QColor  &accent = kAccent,
                    QWidget       *parent = nullptr);

  bool is_checked() const { return this->checked_; }
  void set_checked(bool checked, bool animate = true);

  QSize sizeHint() const override;

signals:
  void toggled(bool checked);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  QRect track_rect() const;

  QString            label_;
  bool               checked_ = false;
  QColor             accent_ = kAccent;
  bool               hovered_ = false;
  double             knob_t_ = 0.0; ///< 0 = off position, 1 = on position
  QVariantAnimation *slide_ = nullptr;
};

/// 18x18 box left, label right. Use for flags.
class CheckBoxRow : public QWidget
{
  Q_OBJECT

public:
  explicit CheckBoxRow(const QString &label,
                       bool           checked = false,
                       const QColor  &accent = kAccent,
                       QWidget       *parent = nullptr);

  bool is_checked() const { return this->checked_; }
  void set_checked(bool checked);

  QSize sizeHint() const override;

signals:
  void toggled(bool checked);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  QRect box_rect() const;

  QString label_;
  bool    checked_ = false;
  QColor  accent_ = kAccent;
  bool    hovered_ = false;
};

} // namespace hesiod::pp
