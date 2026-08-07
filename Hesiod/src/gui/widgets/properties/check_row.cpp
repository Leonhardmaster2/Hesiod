/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <QEnterEvent>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include "hesiod/gui/widgets/properties/check_row.hpp"

namespace hesiod::pp
{

// ---------------------------------------------------------------------------
// CheckRow - switch
// ---------------------------------------------------------------------------

CheckRow::CheckRow(const QString &label,
                   bool           checked,
                   const QColor  &accent,
                   QWidget       *parent)
    : QWidget(parent), label_(label), checked_(checked), accent_(accent)
{
  this->setCursor(Qt::PointingHandCursor);
  this->setAttribute(Qt::WA_Hover, true);
  this->setMinimumHeight(kCheckRowHeight);

  this->knob_t_ = checked ? 1.0 : 0.0;

  this->slide_ = new QVariantAnimation(this);
  this->slide_->setDuration(kSwitchMs);
  this->slide_->setEasingCurve(QEasingCurve::OutCubic);
  this->connect(this->slide_,
                &QVariantAnimation::valueChanged,
                this,
                [this](const QVariant &v)
                {
                  this->knob_t_ = v.toDouble();
                  this->update();
                });
}

QSize CheckRow::sizeHint() const { return QSize(220, kCheckRowHeight); }

QRect CheckRow::track_rect() const
{
  return QRect(this->width() - kSwitchWidth,
               (this->height() - kSwitchHeight) / 2,
               kSwitchWidth,
               kSwitchHeight);
}

void CheckRow::set_checked(bool checked, bool animate)
{
  if (checked == this->checked_)
    return;

  this->checked_ = checked;

  this->slide_->stop();
  if (animate)
  {
    this->slide_->setStartValue(this->knob_t_);
    this->slide_->setEndValue(checked ? 1.0 : 0.0);
    this->slide_->start();
  }
  else
  {
    this->knob_t_ = checked ? 1.0 : 0.0;
    this->update();
  }

  Q_EMIT this->toggled(checked);
}

void CheckRow::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  QFont f = this->font();
  f.setPixelSize(kLabelPx);
  p.setFont(f);
  p.setPen(this->hovered_ ? QColor("#d8d8d8") : QColor("#a0a0a0"));
  p.drawText(QRect(0, 0, this->width() - kSwitchWidth - kGap, this->height()),
             Qt::AlignVCenter | Qt::AlignLeft,
             p.fontMetrics().elidedText(this->label_,
                                        Qt::ElideRight,
                                        this->width() - kSwitchWidth - kGap));

  // square track - the design language has no pills
  const QRect  track = this->track_rect();
  const QColor track_bg = this->checked_ ? this->accent_.darker(170) : kField;
  const QColor track_border = this->checked_ ? this->accent_ : kFieldBorder;

  p.setPen(QPen(track_border, 1));
  p.setBrush(track_bg);
  p.drawRoundedRect(QRectF(track).adjusted(0.5, 0.5, -0.5, -0.5), 2, 2);

  const int  travel = track.width() - kKnobSize - 6;
  const QRect knob(track.left() + 3 + int(std::round(this->knob_t_ * travel)),
                   track.center().y() - kKnobSize / 2 + 1,
                   kKnobSize,
                   kKnobSize);

  QLinearGradient g(knob.topLeft(), knob.bottomLeft());
  g.setColorAt(0.0, this->checked_ ? kKnobOnTop : kKnobOffTop);
  g.setColorAt(1.0, this->checked_ ? kKnobOnBottom : kKnobOffBottom);
  p.setPen(Qt::NoPen);
  p.setBrush(g);
  p.drawRoundedRect(knob, 2, 2);
}

void CheckRow::mouseReleaseEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton && this->rect().contains(event->pos()))
    this->set_checked(!this->checked_);
  QWidget::mouseReleaseEvent(event);
}

void CheckRow::enterEvent(QEnterEvent *event)
{
  this->hovered_ = true;
  this->update();
  QWidget::enterEvent(event);
}

void CheckRow::leaveEvent(QEvent *event)
{
  this->hovered_ = false;
  this->update();
  QWidget::leaveEvent(event);
}

// ---------------------------------------------------------------------------
// CheckBoxRow - square box
// ---------------------------------------------------------------------------

CheckBoxRow::CheckBoxRow(const QString &label,
                         bool           checked,
                         const QColor  &accent,
                         QWidget       *parent)
    : QWidget(parent), label_(label), checked_(checked), accent_(accent)
{
  this->setCursor(Qt::PointingHandCursor);
  this->setAttribute(Qt::WA_Hover, true);
  this->setMinimumHeight(kCheckBoxRowHeight);
}

QSize CheckBoxRow::sizeHint() const { return QSize(200, kCheckBoxRowHeight); }

QRect CheckBoxRow::box_rect() const
{
  return QRect(0, (this->height() - kBoxSize) / 2, kBoxSize, kBoxSize);
}

void CheckBoxRow::set_checked(bool checked)
{
  if (checked == this->checked_)
    return;
  this->checked_ = checked;
  this->update();
  Q_EMIT this->toggled(checked);
}

void CheckBoxRow::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  const QRect box = this->box_rect();

  p.setPen(QPen(this->checked_ ? this->accent_
                               : (this->hovered_ ? kFieldBorderHover : kFieldBorder),
                1));
  p.setBrush(this->hovered_ ? kFieldHover : kField);
  p.drawRoundedRect(QRectF(box).adjusted(0.5, 0.5, -0.5, -0.5), 2, 2);

  if (this->checked_)
  {
    // drawn, not a glyph - no icon file to go missing
    QPainterPath tick;
    tick.moveTo(box.left() + 4.0, box.center().y() + 0.5);
    tick.lineTo(box.center().x() - 0.5, box.bottom() - 4.5);
    tick.lineTo(box.right() - 3.5, box.top() + 4.5);
    p.setPen(QPen(this->accent_, 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);
    p.drawPath(tick);
  }

  QFont f = this->font();
  f.setPixelSize(kLabelPx);
  p.setFont(f);
  p.setPen(this->hovered_ ? kInkPrimary : kInkIcon);
  const int x = box.right() + 1 + kGap;
  p.drawText(QRect(x, 0, this->width() - x, this->height()),
             Qt::AlignVCenter | Qt::AlignLeft,
             p.fontMetrics().elidedText(this->label_, Qt::ElideRight, this->width() - x));
}

void CheckBoxRow::mouseReleaseEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton && this->rect().contains(event->pos()))
    this->set_checked(!this->checked_);
  QWidget::mouseReleaseEvent(event);
}

void CheckBoxRow::enterEvent(QEnterEvent *event)
{
  this->hovered_ = true;
  this->update();
  QWidget::enterEvent(event);
}

void CheckBoxRow::leaveEvent(QEvent *event)
{
  this->hovered_ = false;
  this->update();
  QWidget::leaveEvent(event);
}

} // namespace hesiod::pp
