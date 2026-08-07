/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>

#include <QEnterEvent>
#include <QKeyEvent>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QWheelEvent>

#include "hesiod/gui/widgets/properties/param_slider.hpp"

namespace hesiod::pp
{

ParamSlider::ParamSlider(QWidget *parent) : QWidget(parent)
{
  this->setMouseTracking(true);
  this->setFocusPolicy(Qt::StrongFocus);
  this->setAttribute(Qt::WA_Hover, true);
  // pinned, not a minimum: in a stretching QVBoxLayout a minimum lets rows
  // grow and the 36px row rhythm drifts
  this->setFixedHeight(kRowHeight);

  this->click_timer_.start();

  this->glide_ = new QVariantAnimation(this);
  this->glide_->setDuration(kGlideMs);
  this->glide_->setEasingCurve(QEasingCurve::OutCubic);

  this->connect(this->glide_,
                &QVariantAnimation::valueChanged,
                this,
                [this](const QVariant &v)
                {
                  this->value_ = v.toDouble();
                  this->update();
                  Q_EMIT this->value_changed();
                });

  this->connect(this->glide_,
                &QVariantAnimation::finished,
                this,
                [this]() { Q_EMIT this->edit_ended(); });
}

// ----------------------------------------------------------------- setters

void ParamSlider::set_label(const QString &label)
{
  this->label_ = label;
  this->update();
}

void ParamSlider::set_range(double from, double to, double step)
{
  this->from_ = from;
  this->to_ = to;
  this->step_ = (step > 0.0) ? step : 1.0;
  this->value_ = std::clamp(this->value_, from, to);
  this->update();
}

void ParamSlider::set_quantized(bool quantized)
{
  this->quantized_ = quantized;
}

void ParamSlider::set_decimals(int decimals)
{
  this->decimals_ = std::max(0, decimals);
  this->update();
}

void ParamSlider::set_suffix(const QString &suffix)
{
  this->suffix_ = suffix;
  this->update();
}

void ParamSlider::set_accent(const QColor &accent)
{
  this->accent_ = accent;
  this->update();
}

void ParamSlider::set_default_value(double v)
{
  this->default_value_ = v;
  this->update();
}

void ParamSlider::set_locked(bool locked)
{
  this->locked_ = locked;
  this->setCursor(locked ? Qt::ArrowCursor : Qt::PointingHandCursor);
  this->update();
}

void ParamSlider::set_value(double v)
{
  // model -> widget: never animate, and stop any glide that is mid-flight so
  // the two do not fight over the value
  this->glide_->stop();
  this->value_ = std::clamp(v, this->from_, this->to_);
  this->update();
}

void ParamSlider::glide_to(double v)
{
  // A running QVariantAnimation ignores a retargeted end value, exactly like
  // QML's NumberAnimation - it must be stopped first or the reset silently
  // does nothing.
  this->glide_->stop();
  this->glide_->setStartValue(this->value_);
  this->glide_->setEndValue(std::clamp(v, this->from_, this->to_));
  this->glide_->start();
  Q_EMIT this->edit_started();
}

bool ParamSlider::is_modified() const
{
  return std::abs(this->value_ - this->default_value_) > 1e-4;
}

// ---------------------------------------------------------------- geometry

QRect ParamSlider::label_rect() const
{
  return QRect(0, 0, label_width(this->width()), this->height());
}

QRect ParamSlider::field_rect() const
{
  const int w = field_width(this->width());
  return QRect(this->width() - w,
               (this->height() - kFieldHeight) / 2,
               w,
               kFieldHeight);
}

QRect ParamSlider::rail_rect() const
{
  const int left = this->label_rect().right() + 1 + kGap;
  const int right = this->field_rect().left() - kGap;
  return QRect(left, 0, std::max(0, right - left), this->height());
}

QRect ParamSlider::groove_rect() const
{
  const QRect r = this->rail_rect();
  return QRect(r.left(), r.center().y() - kRailHeight / 2, r.width(), kRailHeight);
}

double ParamSlider::ratio() const
{
  if (this->to_ <= this->from_)
    return 0.0;
  return std::clamp((this->value_ - this->from_) / (this->to_ - this->from_),
                    0.0,
                    1.0);
}

double ParamSlider::value_at(int x) const
{
  const QRect r = this->rail_rect();
  if (r.width() <= 0)
    return this->from_;
  const double t = std::clamp(double(x - r.left()) / double(r.width()), 0.0, 1.0);
  return this->from_ + t * (this->to_ - this->from_);
}

void ParamSlider::set_value_snapped(double v)
{
  this->glide_->stop();
  const double snapped = this->quantized_
                             ? std::round(v / this->step_) * this->step_
                             : v;
  this->value_ = std::clamp(snapped, this->from_, this->to_);
  this->update();
  Q_EMIT this->value_changed();
}

QColor ParamSlider::ink() const
{
  if (this->locked_)
    return kInkLocked;
  return this->is_modified() ? kInkModified : kInkDefault;
}

QString ParamSlider::display_text() const
{
  QString s = QString::number(this->value_, 'f', this->decimals_);
  if (!this->suffix_.isEmpty())
    s += " " + this->suffix_;
  return s;
}

QSize ParamSlider::sizeHint() const { return QSize(320, kRowHeight); }

// ---------------------------------------------------------------- painting

void ParamSlider::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  const QColor  ink_colour = this->ink();
  const QRect   groove = this->groove_rect();
  const QRect   rail = this->rail_rect();
  const double  t = this->ratio();

  // --- label, uppercase with tracking, never bold
  QFont label_font = this->font();
  label_font.setPixelSize(kLabelPx);
  label_font.setCapitalization(QFont::AllUppercase);
  label_font.setLetterSpacing(QFont::AbsoluteSpacing, 1.0);
  label_font.setBold(false);
  p.setFont(label_font);
  p.setPen(ink_colour);
  p.drawText(this->label_rect().adjusted(0, 0, -4, 0),
             Qt::AlignVCenter | Qt::AlignLeft,
             p.fontMetrics().elidedText(this->label_,
                                        Qt::ElideRight,
                                        this->label_rect().width() - 4));

  // --- rail well
  p.setPen(QPen(kRailBorder, 1));
  p.setBrush(kRailWell);
  p.drawRoundedRect(QRectF(groove).adjusted(0.5, 0.5, -0.5, -0.5),
                    kRailRadius,
                    kRailRadius);

  // --- fill: ALWAYS the accent, only its opacity reacts to lock
  if (t > 0.0)
  {
    const int fill_w = int(std::round(t * groove.width()));
    if (fill_w > 0)
    {
      p.save();
      p.setOpacity(this->locked_ ? kFillOpacityLocked : kFillOpacity);
      p.setPen(Qt::NoPen);
      p.setBrush(this->accent_);
      p.drawRoundedRect(QRectF(groove.left(), groove.top(), fill_w, groove.height()),
                        kRailRadius,
                        kRailRadius);
      p.restore();
    }
  }

  // --- machined thumb with grip notch
  if (rail.width() > kThumbWidth)
  {
    const int  x = rail.left() + int(std::round(t * (rail.width() - kThumbWidth)));
    const QRect thumb(x,
                      rail.center().y() - kThumbHeight / 2,
                      kThumbWidth,
                      kThumbHeight);

    p.save();
    p.setOpacity(this->locked_ ? kThumbOpacityLocked : 1.0);

    QLinearGradient g(thumb.topLeft(), thumb.bottomLeft());
    g.setColorAt(0.0, kThumbTop);
    g.setColorAt(1.0, kThumbBottom);
    p.setPen(QPen(kThumbBorder, 1));
    p.setBrush(g);
    p.drawRoundedRect(QRectF(thumb).adjusted(0.5, 0.5, -0.5, -0.5),
                      kThumbRadius,
                      kThumbRadius);

    p.setPen(Qt::NoPen);
    p.setBrush(kThumbNotch);
    p.drawRect(QRect(thumb.center().x(),
                     thumb.center().y() - kNotchHeight / 2 + 1,
                     kNotchWidth,
                     kNotchHeight));
    p.restore();
  }

  // --- value box
  if (!this->editor_ || !this->editor_->isVisible())
  {
    const QRect f = this->field_rect();
    p.setPen(QPen(this->field_hovered_ ? kFieldBorderHover : kFieldBorder, 1));
    p.setBrush(this->field_hovered_ ? kFieldHover : kField);
    p.drawRoundedRect(QRectF(f).adjusted(0.5, 0.5, -0.5, -0.5),
                      kFieldRadius,
                      kFieldRadius);

    QFont value_font = this->font();
    value_font.setFamily(mono_family());
    value_font.setPixelSize(kValuePx);
    p.setFont(value_font);
    p.setPen(ink_colour);
    p.drawText(f.adjusted(8, 0, -6, 0),
               Qt::AlignVCenter | Qt::AlignLeft,
               this->display_text());
  }
}

// -------------------------------------------------------------- interaction

void ParamSlider::mousePressEvent(QMouseEvent *event)
{
  if (this->locked_ || event->button() != Qt::LeftButton)
  {
    QWidget::mousePressEvent(event);
    return;
  }

  const QPoint pos = event->pos();

  if (this->field_rect().contains(pos))
  {
    this->begin_edit();
    return;
  }

  this->setFocus(Qt::MouseFocusReason);

  // manual double-click detection alongside the native signal: two quick
  // clicks near the same point reset, and this path also covers presses that
  // land outside the rail
  const qint64 now = this->click_timer_.elapsed();
  static qint64 last = -100000;
  if (now - last < kDoubleClickMs && std::abs(pos.x() - this->last_click_x_) < 8)
  {
    last = -100000;
    this->dragging_ = false;
    this->pressed_ = false;
    this->glide_to(this->default_value_);
    return;
  }
  last = now;
  this->last_click_x_ = pos.x();

  this->pressed_ = true;
  this->dragging_ = false;
  this->press_x_ = pos.x();

  if (this->rail_rect().adjusted(-4, 0, 4, 0).contains(pos))
    this->glide_to(this->value_at(pos.x()));
}

void ParamSlider::mouseMoveEvent(QMouseEvent *event)
{
  const bool over_field = this->field_rect().contains(event->pos());
  if (over_field != this->field_hovered_)
  {
    this->field_hovered_ = over_field;
    this->setCursor(over_field && !this->locked_ ? Qt::IBeamCursor
                                                 : Qt::PointingHandCursor);
    this->update();
  }

  if (!this->pressed_ || this->locked_)
  {
    QWidget::mouseMoveEvent(event);
    return;
  }

  // below the threshold the gesture stays a click, so it keeps gliding
  if (!this->dragging_ &&
      std::abs(event->pos().x() - this->press_x_) > kDragThresholdPx)
    this->dragging_ = true;

  if (this->dragging_)
    this->set_value_snapped(this->value_at(event->pos().x()));
}

void ParamSlider::mouseReleaseEvent(QMouseEvent *event)
{
  if (this->pressed_)
  {
    this->pressed_ = false;
    if (this->dragging_)
    {
      this->dragging_ = false;
      Q_EMIT this->edit_ended();
    }
  }
  QWidget::mouseReleaseEvent(event);
}

void ParamSlider::mouseDoubleClickEvent(QMouseEvent *event)
{
  if (this->locked_ || event->button() != Qt::LeftButton)
    return;
  if (this->field_rect().contains(event->pos()))
    return;

  this->dragging_ = false;
  this->pressed_ = false;
  this->glide_to(this->default_value_);
}

void ParamSlider::wheelEvent(QWheelEvent *event)
{
  // Only a focused row responds to the wheel. Otherwise scrolling the panel
  // nudges whatever value happens to be under the pointer.
  if (this->locked_ || !this->hasFocus())
  {
    event->ignore();
    return;
  }

  const double mult = (event->modifiers() & Qt::ShiftModifier) ? 10.0 : 1.0;
  const double delta = (event->angleDelta().y() > 0 ? 1.0 : -1.0) * this->step_ * mult;
  this->set_value_snapped(this->value_ + delta);
  Q_EMIT this->edit_ended();
  event->accept();
}

void ParamSlider::keyPressEvent(QKeyEvent *event)
{
  if (this->locked_)
  {
    QWidget::keyPressEvent(event);
    return;
  }

  if (event->key() == Qt::Key_Left)
    this->set_value_snapped(this->value_ - this->step_);
  else if (event->key() == Qt::Key_Right)
    this->set_value_snapped(this->value_ + this->step_);
  else
  {
    QWidget::keyPressEvent(event);
    return;
  }
  Q_EMIT this->edit_ended();
}

void ParamSlider::enterEvent(QEnterEvent *event)
{
  this->hovered_ = true;
  this->update();
  QWidget::enterEvent(event);
}

void ParamSlider::leaveEvent(QEvent *event)
{
  this->hovered_ = false;
  this->field_hovered_ = false;
  this->update();
  QWidget::leaveEvent(event);
}

void ParamSlider::resizeEvent(QResizeEvent *event)
{
  if (this->editor_)
    this->editor_->setGeometry(this->field_rect());
  QWidget::resizeEvent(event);
}

// -------------------------------------------------------------- text entry

void ParamSlider::begin_edit()
{
  this->glide_->stop();

  if (!this->editor_)
  {
    this->editor_ = new QLineEdit(this);
    this->editor_->setFrame(false);
    QFont f = this->font();
    f.setFamily(mono_family());
    f.setPixelSize(kValuePx);
    this->editor_->setFont(f);
    this->editor_->setStyleSheet(
        QString("QLineEdit { background: %1; color: #ffffff; border: 1px solid %2;"
                " border-radius: %3px; padding-left: 7px;"
                " selection-background-color: %2; selection-color: %1; }")
            .arg(kFieldEditing.name(), this->accent_.name())
            .arg(kFieldRadius));

    this->connect(this->editor_,
                  &QLineEdit::returnPressed,
                  this,
                  [this]() { this->commit_edit(); });
    this->connect(this->editor_,
                  &QLineEdit::editingFinished,
                  this,
                  [this]()
                  {
                    if (this->editor_ && this->editor_->isVisible())
                      this->commit_edit();
                  });
  }

  this->editor_->setGeometry(this->field_rect());
  this->editor_->setText(QString::number(this->value_, 'f', this->decimals_));
  this->editor_->show();
  this->editor_->setFocus(Qt::MouseFocusReason);
  this->editor_->selectAll();
  this->update();
}

void ParamSlider::commit_edit()
{
  if (!this->editor_)
    return;

  const QString text = this->editor_->text().trimmed();

  // Accept both decimal separators regardless of locale: QString::toDouble is
  // C-locale only, so "1,5" silently fails on a German system and the typed
  // value is thrown away without a word.
  bool   ok = false;
  double v = text.toDouble(&ok);
  if (!ok)
    v = QLocale().toDouble(text, &ok);
  if (!ok)
  {
    QString normalized = text;
    normalized.replace(QLatin1Char(','), QLatin1Char('.'));
    v = normalized.toDouble(&ok);
  }

  this->editor_->hide();
  this->update();

  if (!ok)
    return; // leave the value untouched rather than clamping to nonsense

  // Deliberately no edit_ended here. glide_to() only STARTS the animation, so
  // value() is still the old number at this instant - emitting now would make
  // listeners commit the value the user just replaced. The glide emits
  // edit_ended itself when it finishes, carrying the real value.
  this->glide_to(v);
}

} // namespace hesiod::pp
