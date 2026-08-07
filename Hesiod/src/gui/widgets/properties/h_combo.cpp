/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <QApplication>
#include <QEnterEvent>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QHideEvent>
#include <QVariantAnimation>
#include <QScreen>
#include <QWheelEvent>

#include "hesiod/gui/widgets/properties/h_combo.hpp"

namespace hesiod::pp
{

namespace
{
constexpr int kComboHeight = 30;
constexpr int kChevronSize = 24;
constexpr int kOptionHeight = 26;
constexpr int kPopupPad = 4;
} // namespace

// ---------------------------------------------------------------------------
// popup
// ---------------------------------------------------------------------------

class HComboPopup : public QFrame
{
public:
  HComboPopup(const QStringList &options, int current, QWidget *anchor)
      : QFrame(anchor, Qt::Popup), options_(options), current_(current)
  {
    this->setAttribute(Qt::WA_DeleteOnClose);
    this->setMouseTracking(true);
    this->setFrameShape(QFrame::NoFrame);
    this->full_height_ = options.size() * kOptionHeight + 2 * kPopupPad;
    this->resize(anchor->width(), this->full_height_);

    // Reveal: grow from the top edge rather than appearing whole. Matches the
    // 220ms section collapse in feel, but quicker - a menu should not make you
    // wait for it.
    this->reveal_ = new QVariantAnimation(this);
    this->reveal_->setDuration(130);
    this->reveal_->setEasingCurve(QEasingCurve::OutCubic);
    this->connect(this->reveal_,
                  &QVariantAnimation::valueChanged,
                  this,
                  [this](const QVariant &v) { this->set_revealed(v.toInt()); });
  }

  int full_height() const { return this->full_height_; }

  /**
   * @brief Grow upward from a fixed bottom edge instead of downward from a
   * fixed top.
   *
   * A plain resize() keeps the top-left pinned, so a popup that flipped above
   * its field appeared at its final top edge and grew DOWN towards the field -
   * reading as the menu falling from the ceiling rather than opening out of
   * the control. When flipped, the bottom is the edge that must stay put.
   *
   * @param bottom_y Global y the popup's bottom edge is pinned to.
   */
  void set_grow_up(int bottom_y)
  {
    this->grow_up_ = true;
    this->bottom_y_ = bottom_y;
  }

  std::function<void()> on_closing;

  void animate_close()
  {
    if (this->closing_)
      return;
    this->closing_ = true;

    // Arm the reopen guard NOW, not in hideEvent: the dismissing click is
    // being processed at this instant, and the guard has to already exist.
    if (this->on_closing)
      this->on_closing();

    this->reveal_->stop();
    this->reveal_->setStartValue(this->height());
    this->reveal_->setEndValue(1);
    this->connect(this->reveal_,
                  &QVariantAnimation::finished,
                  this,
                  [this]() { QFrame::close(); });
    this->reveal_->start();
  }

  void animate_open()
  {
    this->set_revealed(1);
    this->reveal_->setStartValue(1);
    this->reveal_->setEndValue(this->full_height_);
    this->reveal_->start();
  }

  /// Apply one animation frame, keeping whichever edge is anchored fixed.
  void set_revealed(int h)
  {
    if (this->grow_up_)
      this->setGeometry(this->x(), this->bottom_y_ - h, this->width(), h);
    else
      this->resize(this->width(), h);
  }

  std::function<void(int)> on_pick;
  std::function<void()>    on_hidden;

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRectF r = QRectF(this->rect()).adjusted(0.5, 0.5, -0.5, -0.5);
    p.setPen(QPen(kHairline, 1));
    p.setBrush(kPopup);
    p.drawRoundedRect(r, 2, 2);

    QFont f = this->font();
    f.setPixelSize(kLabelPx);
    p.setFont(f);

    for (int i = 0; i < this->options_.size(); ++i)
    {
      const QRect row(kPopupPad,
                      kPopupPad + i * kOptionHeight,
                      this->width() - 2 * kPopupPad,
                      kOptionHeight);

      if (i == this->hover_)
      {
        p.setPen(Qt::NoPen);
        p.setBrush(kSectionHeader);
        p.drawRoundedRect(row, 2, 2);
      }

      p.setPen(i == this->current_ ? kAccent : kInkIcon);
      p.drawText(row.adjusted(8, 0, -8, 0),
                 Qt::AlignVCenter | Qt::AlignLeft,
                 p.fontMetrics().elidedText(this->options_[i],
                                            Qt::ElideRight,
                                            row.width() - 16));
    }
  }

  void mouseMoveEvent(QMouseEvent *e) override
  {
    const int idx = this->index_at(e->pos());
    if (idx != this->hover_)
    {
      this->hover_ = idx;
      this->update();
    }
  }

  void mousePressEvent(QMouseEvent *e) override
  {
    this->saw_press_ = true;

    // Overriding this at all suppressed Qt's built-in "press outside a popup
    // dismisses it" behaviour, so clicking the combo field did nothing: the
    // popup stayed up and the click never reached the field either. Close it
    // explicitly, then let hideEvent arm the reopen guard.
    if (!this->rect().contains(e->pos()))
    {
      this->animate_close();
      return;
    }

    QFrame::mousePressEvent(e);
  }

  void mouseReleaseEvent(QMouseEvent *e) override
  {
    // The release that finishes the click which OPENED this popup arrives
    // here, outside any row. Acting on it closed the menu instantly.
    if (!this->saw_press_ && !this->rect().contains(e->pos()))
      return;

    const int idx = this->index_at(e->pos());
    if (idx < 0)
      return; // Qt::Popup closes itself on a genuine outside click

    if (this->on_pick)
      this->on_pick(idx);
    this->animate_close();
  }

  void leaveEvent(QEvent *) override
  {
    this->hover_ = -1;
    this->update();
  }

  void hideEvent(QHideEvent *e) override
  {
    // destroyed() is too late: WA_DeleteOnClose defers deletion to the next
    // event-loop pass, by which time the click that dismissed this popup has
    // already reached the combo and reopened it.
    if (this->on_hidden)
      this->on_hidden();
    QFrame::hideEvent(e);
  }

private:
  int index_at(const QPoint &pos) const
  {
    if (pos.y() < kPopupPad)
      return -1;
    const int idx = (pos.y() - kPopupPad) / kOptionHeight;
    return (idx >= 0 && idx < this->options_.size()) ? idx : -1;
  }

  QStringList        options_;
  int                current_ = 0;
  int                hover_ = -1;
  bool               saw_press_ = false;
  bool               closing_ = false;
  int                full_height_ = 0;
  QVariantAnimation *reveal_ = nullptr;
  bool               grow_up_ = false;
  int                bottom_y_ = 0;
};

// ---------------------------------------------------------------------------
// combo
// ---------------------------------------------------------------------------

HCombo::HCombo(QWidget *parent) : QWidget(parent)
{
  this->setCursor(Qt::PointingHandCursor);
  this->setAttribute(Qt::WA_Hover, true);
  this->setMouseTracking(true);
  this->setFixedHeight(kComboHeight);
}

QSize HCombo::sizeHint() const { return QSize(200, kComboHeight); }

void HCombo::set_options(const QStringList &options)
{
  this->options_ = options;
  this->current_ = qBound(0, this->current_, qMax(0, options.size() - 1));
  this->update();
}

void HCombo::set_current(int index)
{
  if (index < 0 || index >= this->options_.size() || index == this->current_)
    return;
  this->current_ = index;
  this->update();
}

QString HCombo::current_text() const
{
  if (this->current_ < 0 || this->current_ >= this->options_.size())
    return {};
  return this->options_[this->current_];
}

QRect HCombo::chevron_rect() const
{
  return QRect(this->width() - kChevronSize - 3,
               (this->height() - kChevronSize) / 2,
               kChevronSize,
               kChevronSize);
}

void HCombo::paintEvent(QPaintEvent *)
{
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  // field
  const QRectF r = QRectF(this->rect()).adjusted(0.5, 0.5, -0.5, -0.5);
  p.setPen(QPen(this->open_      ? kAccent
                : this->hovered_ ? kFieldBorderHover
                                 : kFieldBorder,
                1));
  p.setBrush(this->hovered_ ? kFieldHover : kField);
  p.drawRoundedRect(r, 2, 2);

  QFont f = this->font();
  f.setPixelSize(kLabelPx);
  p.setFont(f);
  p.setPen(kInkTitle);
  const int text_right = this->chevron_rect().left() - 6;
  p.drawText(QRect(10, 0, text_right - 10, this->height()),
             Qt::AlignVCenter | Qt::AlignLeft,
             p.fontMetrics().elidedText(this->current_text(),
                                        Qt::ElideRight,
                                        text_right - 10));

  // raised chevron button
  const QRect  cr = this->chevron_rect();
  const QRectF crf = QRectF(cr).adjusted(0.5, 0.5, -0.5, -0.5);
  QLinearGradient g(crf.topLeft(), crf.bottomLeft());
  g.setColorAt(0.0, this->chevron_hovered_ ? kButtonTopPress : kButtonTop);
  g.setColorAt(1.0, this->chevron_hovered_ ? kButtonBottomPress : kButtonBottom);
  p.setPen(QPen(kButtonBorder, 1));
  p.setBrush(g);
  p.drawRoundedRect(crf, 2, 2);

  // chevron glyph, rotated about its own centre so it never pivots off-axis
  p.save();
  p.translate(QPointF(cr.center()) + QPointF(0.5, 0.5));
  p.rotate(this->open_ ? 180.0 : 0.0);
  QPainterPath chev;
  chev.moveTo(-4.0, -2.0);
  chev.lineTo(0.0, 2.0);
  chev.lineTo(4.0, -2.0);
  p.setPen(QPen(kInkIcon, 1.6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  p.setBrush(Qt::NoBrush);
  p.drawPath(chev);
  p.restore();
}

void HCombo::toggle_popup()
{
  if (this->open_)
  {
    this->close_popup();
    return;
  }

  // Clicking the field while the popup is open goes to the POPUP first (it
  // holds a mouse grab), which closes it - and then the very same click
  // reaches us and would reopen it. Net effect: the menu never appears to
  // collapse. Swallow any click that lands in the wake of a close.
  if (this->since_closed_.isValid() && this->since_closed_.elapsed() < 200)
    return;

  if (this->options_.isEmpty())
    return;

  this->popup_ = new HComboPopup(this->options_, this->current_, this);
  this->popup_->on_pick = [this](int idx)
  {
    const bool changed = (idx != this->current_);
    this->current_ = idx;
    this->update();
    if (changed)
      Q_EMIT this->activated(idx);
  };

  auto mark_closed = [this]()
  {
    this->open_ = false;
    this->since_closed_.restart();
    this->update();
  };
  this->popup_->on_hidden = mark_closed;
  this->popup_->on_closing = mark_closed;

  this->connect(this->popup_,
                &QObject::destroyed,
                this,
                [this]()
                {
                  this->popup_ = nullptr;
                  this->open_ = false;
                  this->since_closed_.restart();
                  this->update();
                });

  // flip above the field when there is no room below
  QPoint      pos = this->mapToGlobal(QPoint(0, this->height() + 2));
  const QRect avail = this->screen() ? this->screen()->availableGeometry() : QRect();

  if (avail.isValid() && pos.y() + this->popup_->full_height() > avail.bottom())
  {
    pos = this->mapToGlobal(QPoint(0, -this->popup_->full_height() - 2));

    // Anchor the bottom edge to just above the field so the reveal opens
    // upward out of the control, instead of dropping in from its final top
    // edge and growing back down towards it.
    this->popup_->set_grow_up(this->mapToGlobal(QPoint(0, -2)).y());
  }

  this->popup_->move(pos);
  this->popup_->show();
  this->popup_->animate_open();

  this->open_ = true;
  this->update();
}

void HCombo::close_popup()
{
  if (this->popup_)
    this->popup_->close();
  this->open_ = false;
  this->update();
}

void HCombo::mousePressEvent(QMouseEvent *event)
{
  // deliberately does nothing but accept: opening happens on release
  if (event->button() != Qt::LeftButton)
    QWidget::mousePressEvent(event);
}

void HCombo::mouseReleaseEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton && this->rect().contains(event->pos()))
    this->toggle_popup();
}

void HCombo::enterEvent(QEnterEvent *event)
{
  this->hovered_ = true;
  this->update();
  QWidget::enterEvent(event);
}

void HCombo::leaveEvent(QEvent *event)
{
  this->hovered_ = false;
  this->chevron_hovered_ = false;
  this->update();
  QWidget::leaveEvent(event);
}

void HCombo::wheelEvent(QWheelEvent *event)
{
  // Scrolling the panel must never change a value under the pointer.
  if (!this->hasFocus() || this->options_.isEmpty())
  {
    event->ignore();
    return;
  }

  const int step = event->angleDelta().y() > 0 ? -1 : 1;
  const int next = qBound(0, this->current_ + step, this->options_.size() - 1);
  if (next != this->current_)
  {
    this->current_ = next;
    this->update();
    Q_EMIT this->activated(next);
  }
  event->accept();
}

} // namespace hesiod::pp
