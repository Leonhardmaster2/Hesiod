/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file mod_button.hpp
 * @brief Beveled chip / action button.
 *
 * Ported from ModButton.qml. Two jobs in one widget, exactly as in the
 * reference: a small toggling modifier chip, and the wide action button used
 * by the pad and path editors (Center / Random / Clear / ...).
 *
 * Set `checkable` false and drive `active` externally to build an exclusive
 * group (512 / 1K / 2K / 4K). Active state is an accent border plus accent
 * text - never a coloured fill.
 */
#pragma once
#include <QColor>
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

class ModButton : public QWidget
{
  Q_OBJECT

public:
  explicit ModButton(const QString &label,
                     bool           checkable = true,
                     const QColor  &accent = kAccent,
                     QWidget       *parent = nullptr);

  bool is_active() const { return this->active_; }
  void set_active(bool active);
  void set_accent(const QColor &accent);

  QSize sizeHint() const override;

signals:
  void clicked();
  void toggled(bool active);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  QString label_;
  bool    checkable_ = true;
  bool    active_ = false;
  QColor  accent_ = kAccent;
  bool    hovered_ = false;
  bool    pressed_ = false;
};

} // namespace hesiod::pp
