/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <QEnterEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QShowEvent>
#include <QTimer>

#include "hesiod/gui/widgets/properties/pp_section.hpp"

namespace hesiod::pp
{

// ---------------------------------------------------------------------------
// header strip
// ---------------------------------------------------------------------------

class SectionHeader : public QWidget
{
public:
  SectionHeader(const QString &title,
                const QString &index,
                const QColor  &accent,
                QWidget       *parent)
      : QWidget(parent), title_(title.toUpper()), index_(index), accent_(accent)
  {
    this->setFixedHeight(kSectionHeaderHeight);
    this->setCursor(Qt::PointingHandCursor);
    this->setAttribute(Qt::WA_Hover, true);
  }

  void set_expanded(bool e)
  {
    this->expanded_ = e;
    this->update();
  }

  std::function<void()> on_click;

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QColor bg = this->pressed_    ? kSectionHeaderPress
                      : this->hovered_  ? kSectionHeaderHover
                                        : kSectionHeader;
    p.fillRect(this->rect(), bg);

    // bevel: light top edge, dark bottom edge
    p.fillRect(QRect(0, 0, this->width(), 1), kBevelTop);
    p.fillRect(QRect(0, this->height() - 1, this->width(), 1), kBevelBottom);

    // disclosure triangle, rotated about its own centre
    const QPointF centre(16 + 7, this->height() / 2.0);
    p.save();
    p.translate(centre);
    p.rotate(this->expanded_ ? 90.0 : 0.0);
    QPainterPath tri;
    tri.moveTo(-3.0, -4.5);
    tri.lineTo(4.0, 0.0);
    tri.lineTo(-3.0, 4.5);
    tri.closeSubpath();
    p.setPen(Qt::NoPen);
    p.setBrush(kInkDefault);
    p.drawPath(tri);
    p.restore();

    int x = 16 + 14 + kGap;

    if (!this->index_.isEmpty())
    {
      QFont f = this->font();
      f.setFamily(mono_family());
      f.setPixelSize(kIndexPx);
      p.setFont(f);
      p.setPen(this->accent_);
      const int w = p.fontMetrics().horizontalAdvance(this->index_);
      p.drawText(QRect(x, 0, w, this->height()),
                 Qt::AlignVCenter | Qt::AlignLeft,
                 this->index_);
      x += w + kGap;
    }

    QFont tf = this->font();
    tf.setPixelSize(kTitlePx);
    tf.setBold(true);
    tf.setLetterSpacing(QFont::AbsoluteSpacing, 1.2);
    p.setFont(tf);
    p.setPen(kInkTitle);
    p.drawText(QRect(x, 0, this->width() - x - 12, this->height()),
               Qt::AlignVCenter | Qt::AlignLeft,
               p.fontMetrics().elidedText(this->title_,
                                          Qt::ElideRight,
                                          this->width() - x - 12));
  }

  void enterEvent(QEnterEvent *e) override
  {
    this->hovered_ = true;
    this->update();
    QWidget::enterEvent(e);
  }

  void leaveEvent(QEvent *e) override
  {
    this->hovered_ = false;
    this->pressed_ = false;
    this->update();
    QWidget::leaveEvent(e);
  }

  void mousePressEvent(QMouseEvent *e) override
  {
    if (e->button() == Qt::LeftButton)
    {
      this->pressed_ = true;
      this->update();
    }
  }

  void mouseReleaseEvent(QMouseEvent *e) override
  {
    if (e->button() == Qt::LeftButton && this->pressed_)
    {
      this->pressed_ = false;
      this->update();
      if (this->rect().contains(e->pos()) && this->on_click)
        this->on_click();
    }
  }

private:
  QString title_;
  QString index_;
  QColor  accent_;
  bool    expanded_ = true;
  bool    hovered_ = false;
  bool    pressed_ = false;
};

// ---------------------------------------------------------------------------
// section
// ---------------------------------------------------------------------------

PpSection::PpSection(const QString &title,
                     const QString &index,
                     const QColor  &accent,
                     QWidget       *parent)
    : QWidget(parent)
{
  // The section must never be handed more height than its content needs.
  // With a Preferred policy the parent can stretch it, and QVBoxLayout then
  // redistributes that slack between two fixed-height children - which walks
  // the HEADER down and back as the body animates. That is the up-and-down.
  this->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

  auto *outer = new QVBoxLayout(this);
  outer->setContentsMargins(0, 0, 0, 0);
  outer->setSpacing(0);
  outer->setSizeConstraint(QLayout::SetNoConstraint);

  this->header_ = new SectionHeader(title, index, accent, this);
  this->header_->on_click = [this]() { this->set_expanded(!this->expanded_); };
  outer->addWidget(this->header_);

  this->body_ = new QWidget(this);
  this->body_layout_ = new QVBoxLayout(this->body_);
  // reference: body height = content + 22, split 12 top / 10 bottom
  this->body_layout_->setContentsMargins(kBodyPadX, kBodyPadY, kBodyPadX, 10);
  this->body_layout_->setSpacing(kRowSpacing);
  // A widget cannot be animated below its layout's minimum: Qt clamps
  // maximumHeight up to minimumHeight, so the body springs to full size for a
  // frame before snapping away. Releasing both constraints lets maximumHeight
  // alone drive the collapse.
  this->body_layout_->setSizeConstraint(QLayout::SetNoConstraint);
  this->body_->setMinimumHeight(0);
  this->body_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
  outer->addWidget(this->body_);

  // Any leftover height lands here rather than being shared out around the
  // header and body.
  outer->addStretch(0);

  // Drive setFixedHeight rather than maximumHeight. setFixedHeight pins
  // minimum AND maximum together, so there is no second constraint left for
  // Qt to clamp against - which is what made the body spring to full size for
  // a frame before collapsing.
  this->collapse_ = new QVariantAnimation(this);
  this->collapse_->setDuration(kCollapseMs);
  // InOutCubic eases in AND out; OutCubic starts at full speed, which is
  // what makes the first frame read as a jump.
  this->collapse_->setEasingCurve(QEasingCurve::InOutCubic);

  this->connect(this->collapse_,
                &QVariantAnimation::valueChanged,
                this,
                [this](const QVariant &v)
                { this->body_->setFixedHeight(v.toInt()); });

  this->connect(this->collapse_,
                &QVariantAnimation::finished,
                this,
                [this]()
                {
                  // the squeeze is over, so the rows may position themselves
                  // from their own metrics again
                  this->body_layout_->setEnabled(true);

                  if (this->expanded_)
                  {
                    // hand height back to the layout so the body tracks its
                    // content again
                    this->body_->setMinimumHeight(0);
                    this->body_->setMaximumHeight(QWIDGETSIZE_MAX);
                    this->body_layout_->activate();
                    QTimer::singleShot(0,
                                       this,
                                       [this]()
                                       {
                                         if (this->expanded_ &&
                                             this->body_->height() > 0)
                                           this->measured_full_ =
                                               this->body_->height();
                                       });
                  }
                  else
                    this->body_->setVisible(false);
                });
}

void PpSection::showEvent(QShowEvent *event)
{
  QWidget::showEvent(event);

  // Sections start expanded, so the first real layout pass is our chance to
  // record the body's true height before anything ever collapses it.
  if (this->expanded_ && this->measured_full_ < 0)
    QTimer::singleShot(0,
                       this,
                       [this]()
                       {
                         if (this->expanded_ && this->body_->height() > 0)
                           this->measured_full_ = this->body_->height();
                       });
}

int PpSection::body_full_height() const
{
  // Prefer a height we have actually MEASURED. sizeHint() on a hidden,
  // never-laid-out body overestimates, and animating to that value expands the
  // section past its real height before the layout snaps it back - which is
  // the stretch of the whole panel.
  if (this->measured_full_ > 0)
    return this->measured_full_;
  return this->body_layout_->sizeHint().height();
}

void PpSection::set_expanded(bool expanded, bool animate)
{
  if (expanded == this->expanded_)
    return;

  this->expanded_ = expanded;
  this->header_->set_expanded(expanded);

  this->collapse_->stop();

  if (!animate)
  {
    this->body_layout_->setEnabled(true);
    if (expanded)
    {
      this->body_->setMinimumHeight(0);
      this->body_->setMaximumHeight(QWIDGETSIZE_MAX);
    }
    this->body_->setVisible(expanded);
    Q_EMIT this->expanded_changed(expanded);
    return;
  }

  // Freeze the body's layout for the whole animation.
  //
  // With the layout live, driving body_ below its content height does NOT clip
  // - QVBoxLayout treats it as a squeeze and redistributes the shortfall
  // across every row and every gap. So the rows and their padding visibly
  // compress, then spring back to full size when the layout is released. The
  // section is supposed to slide behind its own bottom edge, not deflate.
  //
  // A disabled layout leaves its children exactly where they are, and the body
  // clips them to its shrinking rect, which is the intended effect.
  this->body_layout_->setEnabled(false);

  if (expanded)
  {
    // pin to zero BEFORE showing, so the body is never visible at full height
    this->body_->setFixedHeight(0);
    this->body_->setVisible(true);

    // Force the layout to settle at zero height BEFORE the first animated
    // frame. Without this, showing the body queues a relayout that paints
    // once at its natural size, which is the flash of everything above
    // being shoved around.
    if (this->layout())
      this->layout()->activate();

    this->collapse_->setStartValue(0);
    this->collapse_->setEndValue(this->body_full_height());
  }
  else
  {
    // pin to where it currently is, so frame one is a no-op rather than a jump
    const int current = this->body_->height();
    this->body_->setFixedHeight(current);
    this->collapse_->setStartValue(current);
    this->collapse_->setEndValue(0);
  }

  this->collapse_->start();
  Q_EMIT this->expanded_changed(expanded);
}

} // namespace hesiod::pp
