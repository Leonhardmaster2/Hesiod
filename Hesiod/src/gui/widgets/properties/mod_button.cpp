/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <QEnterEvent>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>

#include "hesiod/gui/widgets/properties/mod_button.hpp"

namespace hesiod::pp
{

ModButton::ModButton(const QString &label,
                     bool           checkable,
                     const QColor  &accent,
                     QWidget       *parent)
    : QWidget(parent), label_(label), checkable_(checkable), accent_(accent)
{
  this->setCursor(Qt::PointingHandCursor);
  this->setAttribute(Qt::WA_Hover, true);
  this->setFixedHeight(kChipHeight);
}

QSize ModButton::sizeHint() const
{
  QFont f = this->font();
  f.setPixelSize(kChipPx);
  f.setBold(true);
  const QFontMetrics fm(f);
  return QSize(fm.horizontalAdvance(this->label_) + 26, kChipHeight);
}

void ModButton::set_active(bool active)
{
  if (active == this->active_)
    return;
  this->active_ = active;
  this->update();
  Q_EMIT this->toggled(active);
}

void ModButton::set_accent(const QColor &accent)
{
  this->accent_ = accent;
  this->update();
}

void ModButton::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  // Qt already stops a disabled widget receiving clicks, but this button
  // paints itself, so without a disabled state it looks live and silently does
  // nothing - which is worse than either being absent or being obviously off.
  const bool off = !this->isEnabled();

  QColor top, bottom;
  if (off)
  {
    top = kChipTopPress;
    bottom = kChipBottomPress;
  }
  else if (this->active_)
  {
    top = kChipTopActive;
    bottom = kChipBottomActive;
  }
  else if (this->pressed_)
  {
    top = kChipTopPress;
    bottom = kChipBottomPress;
  }
  else if (this->hovered_)
  {
    top = kChipTopHover;
    bottom = kChipBottomHover;
  }
  else
  {
    top = kChipTop;
    bottom = kChipBottom;
  }

  const QRectF r = QRectF(this->rect()).adjusted(0.5, 0.5, -0.5, -0.5);

  QLinearGradient g(r.topLeft(), r.bottomLeft());
  g.setColorAt(0.0, top);
  g.setColorAt(1.0, bottom);

  p.setPen(QPen(off ? kChipBorder
                    : (this->active_ ? this->accent_
                                     : (this->hovered_ ? kChipBorderHover
                                                       : kChipBorder)),
                1));
  p.setBrush(g);
  p.drawRoundedRect(r, 2, 2);

  // 1px top bevel highlight, inset so it does not fight the border
  p.setPen(Qt::NoPen);
  QColor bevel(255, 255, 255);
  bevel.setAlphaF(0.07);
  p.setBrush(bevel);
  p.drawRect(QRectF(r.left() + 1.0, r.top() + 1.0, r.width() - 2.0, 1.0));

  QFont f = this->font();
  f.setPixelSize(kChipPx);
  f.setBold(true);
  p.setFont(f);
  // text is what carries state in this design language, so the off state is
  // read primarily from the ink
  p.setPen(off ? kInkLocked
               : (this->active_ ? this->accent_
                                : (this->hovered_ ? kInkPrimary : QColor("#b0b0b0"))));
  p.drawText(this->rect(), Qt::AlignCenter, this->label_);
}

void ModButton::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton)
  {
    this->pressed_ = true;
    this->update();
  }
}

void ModButton::mouseReleaseEvent(QMouseEvent *event)
{
  if (event->button() != Qt::LeftButton || !this->pressed_)
    return;

  this->pressed_ = false;
  this->update();

  if (!this->rect().contains(event->pos()))
    return;

  if (this->checkable_)
    this->set_active(!this->active_);

  Q_EMIT this->clicked();
}

void ModButton::enterEvent(QEnterEvent *event)
{
  this->hovered_ = true;
  this->update();
  QWidget::enterEvent(event);
}

void ModButton::leaveEvent(QEvent *event)
{
  this->hovered_ = false;
  this->pressed_ = false;
  this->update();
  QWidget::leaveEvent(event);
}

} // namespace hesiod::pp
