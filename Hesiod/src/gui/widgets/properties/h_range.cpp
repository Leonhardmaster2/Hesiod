/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>

#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include "hesiod/gui/widgets/properties/h_range.hpp"
#include "hesiod/gui/widgets/properties/mod_button.hpp"

namespace hesiod::pp
{

namespace
{
constexpr int kRailTop = 4;      ///< rail zone y
constexpr int kRailZoneH = 26;   ///< rail zone height
constexpr int kReadoutY = 30;    ///< mono min/max baseline strip
constexpr int kChipY = 46;       ///< preset chip row
constexpr int kRailLeft = 30;    ///< clears the enable box
constexpr int kThumbW = 10;
constexpr int kThumbH = 16;
constexpr int kOnChipW = 44;
constexpr int kChipGap = 6;
constexpr int kTotalH = kChipY + kChipHeight;
constexpr double kMinGap = 0.01; ///< thumbs may not cross
} // namespace

HRange::HRange(QWidget *parent) : QWidget(parent)
{
  this->setMouseTracking(true);
  this->setFixedHeight(kTotalH);

  this->on_chip_ = new ModButton("On", true, kAccent, this);
  this->on_chip_->set_active(true);
  this->connect(this->on_chip_,
                &ModButton::clicked,
                this,
                [this]()
                {
                  this->active_ = this->on_chip_->is_active();
                  this->update();
                  Q_EMIT this->active_toggled(this->active_);
                });

  this->full_chip_ = new ModButton("Full", false, kAccent, this);
  this->center_chip_ = new ModButton("Center", false, kAccent, this);
  this->unit_chip_ = new ModButton("[0, 1]", false, kAccent, this);

  this->connect(this->full_chip_,
                &ModButton::clicked,
                this,
                [this]() { this->apply_preset(this->min_, this->max_); });
  this->connect(this->center_chip_,
                &ModButton::clicked,
                this,
                [this]()
                {
                  const double span = this->max_ - this->min_;
                  this->apply_preset(this->min_ + 0.25 * span,
                                     this->min_ + 0.75 * span);
                });
  this->connect(this->unit_chip_,
                &ModButton::clicked,
                this,
                [this]()
                {
                  this->apply_preset(std::max(0.0, this->min_),
                                     std::min(1.0, this->max_));
                });
}

QSize HRange::sizeHint() const { return QSize(320, kTotalH); }

void HRange::set_bounds(double min, double max, int decimals)
{
  this->min_ = min;
  this->max_ = (max > min) ? max : min + 1.0;
  this->decimals_ = std::max(0, decimals);
  this->update();
}

void HRange::set_range(double lo, double hi)
{
  this->lo_ = std::clamp(lo, this->min_, this->max_);
  this->hi_ = std::clamp(hi, this->lo_, this->max_);
  this->update();
}

void HRange::set_active(bool active)
{
  this->active_ = active;
  this->on_chip_->set_active(active);
  this->update();
}

void HRange::apply_preset(double lo, double hi)
{
  this->set_range(lo, hi);
  Q_EMIT this->range_changed();
  Q_EMIT this->committed();
}

// ---------------------------------------------------------------- geometry

QRect HRange::box_rect() const { return QRect(0, kRailTop, kBoxSize, kBoxSize); }

QRect HRange::rail_rect() const
{
  return QRect(kRailLeft, kRailTop, std::max(0, this->width() - kRailLeft), kRailZoneH);
}

double HRange::ratio(double v) const
{
  if (this->max_ <= this->min_)
    return 0.0;
  return std::clamp((v - this->min_) / (this->max_ - this->min_), 0.0, 1.0);
}

double HRange::value_at(int x) const
{
  const QRect r = this->rail_rect();
  if (r.width() <= 0)
    return this->min_;
  const double t = std::clamp(double(x - r.left()) / double(r.width()), 0.0, 1.0);
  return this->min_ + t * (this->max_ - this->min_);
}

void HRange::layout_chips()
{
  const int y = kChipY;
  const int rest = this->width() - kOnChipW - 3 * kChipGap;
  const int each = std::max(30, rest / 3);

  this->on_chip_->setGeometry(0, y, kOnChipW, kChipHeight);
  this->full_chip_->setGeometry(kOnChipW + kChipGap, y, each, kChipHeight);
  this->center_chip_->setGeometry(kOnChipW + kChipGap + each + kChipGap,
                                  y,
                                  each,
                                  kChipHeight);
  this->unit_chip_->setGeometry(kOnChipW + 2 * kChipGap + 2 * each + kChipGap,
                                y,
                                each,
                                kChipHeight);
}

void HRange::resizeEvent(QResizeEvent *event)
{
  this->layout_chips();
  QWidget::resizeEvent(event);
}

// ---------------------------------------------------------------- painting

void HRange::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  // enable box
  const QRect box = this->box_rect();
  p.setPen(QPen(this->active_ ? kAccent
                              : (this->box_hovered_ ? kFieldBorderHover : kFieldBorder),
                1));
  p.setBrush(this->box_hovered_ ? kFieldHover : kField);
  p.drawRoundedRect(QRectF(box).adjusted(0.5, 0.5, -0.5, -0.5), 2, 2);

  if (this->active_)
  {
    QPainterPath tick;
    tick.moveTo(box.left() + 4.0, box.center().y() + 0.5);
    tick.lineTo(box.center().x() - 0.5, box.bottom() - 4.5);
    tick.lineTo(box.right() - 3.5, box.top() + 4.5);
    p.setPen(QPen(kAccent, 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);
    p.drawPath(tick);
  }

  // rail well
  const QRect rail = this->rail_rect();
  const QRect groove(rail.left(),
                     rail.center().y() - kRailHeight / 2,
                     rail.width(),
                     kRailHeight);
  p.setPen(QPen(kRailBorder, 1));
  p.setBrush(kRangeTrack);
  p.drawRoundedRect(QRectF(groove).adjusted(0.5, 0.5, -0.5, -0.5), 1, 1);

  // span between the thumbs
  const double t_lo = this->ratio(this->lo_);
  const double t_hi = this->ratio(this->hi_);
  const int    x_lo = groove.left() + int(std::round(t_lo * groove.width()));
  const int    x_hi = groove.left() + int(std::round(t_hi * groove.width()));

  if (x_hi > x_lo)
  {
    p.save();
    p.setOpacity(this->active_ ? kFillOpacity : kRangeSpanIdle);
    p.setPen(Qt::NoPen);
    p.setBrush(kRangeSpan);
    p.drawRoundedRect(QRectF(x_lo, groove.top(), x_hi - x_lo, groove.height()), 1, 1);
    p.restore();
  }

  // thumbs
  for (int i = 0; i < 2; ++i)
  {
    const int  cx = (i == 0) ? x_lo : x_hi;
    const QRect thumb(std::clamp(cx - kThumbW / 2,
                                 rail.left(),
                                 rail.right() - kThumbW),
                      rail.center().y() - kThumbH / 2,
                      kThumbW,
                      kThumbH);

    QLinearGradient g(thumb.topLeft(), thumb.bottomLeft());
    g.setColorAt(0.0, kThumbTop);
    g.setColorAt(1.0, kThumbBottom);
    p.setPen(QPen(kThumbBorder, 1));
    p.setBrush(g);
    p.setOpacity(this->active_ ? 1.0 : kThumbOpacityLocked);
    p.drawRoundedRect(QRectF(thumb).adjusted(0.5, 0.5, -0.5, -0.5), 2, 2);
    p.setOpacity(1.0);
  }

  // mono readouts, lo left / hi right
  QFont f = this->font();
  f.setFamily(mono_family());
  f.setPixelSize(10);
  p.setFont(f);
  p.setPen(this->active_ ? kInkDim : kInkLocked);
  p.drawText(QRect(rail.left(), kReadoutY, rail.width() / 2, 14),
             Qt::AlignLeft | Qt::AlignVCenter,
             QString::number(this->lo_, 'f', this->decimals_));
  p.drawText(QRect(rail.center().x(), kReadoutY, rail.width() / 2, 14),
             Qt::AlignRight | Qt::AlignVCenter,
             QString::number(this->hi_, 'f', this->decimals_));
}

// ------------------------------------------------------------- interaction

void HRange::mousePressEvent(QMouseEvent *event)
{
  if (event->button() != Qt::LeftButton)
    return;

  if (this->box_rect().adjusted(-2, -2, 2, 2).contains(event->pos()))
  {
    this->active_ = !this->active_;
    this->on_chip_->set_active(this->active_);
    this->update();
    Q_EMIT this->active_toggled(this->active_);
    return;
  }

  const QRect rail = this->rail_rect();
  if (!rail.adjusted(0, -4, 0, 4).contains(event->pos()))
    return;

  // grab whichever thumb is nearer, so overlapping thumbs stay separable
  const double v = this->value_at(event->pos().x());
  this->drag_thumb_ = (std::abs(v - this->lo_) <= std::abs(v - this->hi_)) ? 0 : 1;
  this->mouseMoveEvent(event);
}

void HRange::mouseMoveEvent(QMouseEvent *event)
{
  const bool over_box = this->box_rect().adjusted(-2, -2, 2, 2).contains(event->pos());
  if (over_box != this->box_hovered_)
  {
    this->box_hovered_ = over_box;
    this->update();
  }

  if (this->drag_thumb_ < 0)
  {
    this->setCursor(this->rail_rect().contains(event->pos()) ? Qt::SplitHCursor
                                                             : Qt::ArrowCursor);
    return;
  }

  const double span = this->max_ - this->min_;
  const double v = this->value_at(event->pos().x());

  if (this->drag_thumb_ == 0)
    this->lo_ = std::clamp(v, this->min_, this->hi_ - kMinGap * span);
  else
    this->hi_ = std::clamp(v, this->lo_ + kMinGap * span, this->max_);

  this->update();
  Q_EMIT this->range_changed();
}

void HRange::mouseReleaseEvent(QMouseEvent *event)
{
  if (this->drag_thumb_ >= 0)
  {
    this->drag_thumb_ = -1;
    Q_EMIT this->committed(); // model is written here, never mid-drag
  }
  QWidget::mouseReleaseEvent(event);
}

} // namespace hesiod::pp
