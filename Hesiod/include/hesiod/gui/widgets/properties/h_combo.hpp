/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file h_combo.hpp
 * @brief Industrial combo box: field + raised chevron + option popup.
 *
 * Ported from HCombo.qml. Serves both Meta combo shapes - a std::string
 * attribute with `allowed_values`, and an int attribute with `enum_items`.
 *
 * The QML version needed a z-lift to escape its Column, and could never sit
 * inside a clipped Section body. Here the popup is a real Qt::Popup top-level
 * window, so neither problem exists - it cannot be clipped by an ancestor.
 */
#pragma once
#include <QElapsedTimer>
#include <QFrame>
#include <QStringList>
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

class HComboPopup;

class HCombo : public QWidget
{
  Q_OBJECT

public:
  explicit HCombo(QWidget *parent = nullptr);

  void        set_options(const QStringList &options);
  void        set_current(int index);
  int         current() const { return this->current_; }
  QString     current_text() const;

  QSize sizeHint() const override;

signals:
  /// Emitted only on a real user pick, never on set_current().
  void activated(int index);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;

private:
  QRect chevron_rect() const;
  void  toggle_popup();
  void  close_popup();

  QStringList  options_;
  int          current_ = 0;
  bool         hovered_ = false;
  bool         chevron_hovered_ = false;
  bool          open_ = false;
  HComboPopup  *popup_ = nullptr;
  QElapsedTimer since_closed_; ///< guards the close-then-reopen race
};

} // namespace hesiod::pp
