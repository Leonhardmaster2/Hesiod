/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <cmath>

#include <QColorDialog>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QRandomGenerator>
#include <QResizeEvent>
#include <QVBoxLayout>

#include "hesiod/gui/widgets/properties/h_combo.hpp"
#include "hesiod/gui/widgets/properties/h_gradient.hpp"
#include "hesiod/gui/widgets/properties/mod_button.hpp"

namespace hesiod::pp
{

namespace
{

/// Alpha checkerboard, so a partly transparent gradient reads as transparent
/// instead of as a darker colour.
const QColor kCheckA{"#2a2a2a"};
const QColor kCheckB{"#343434"};

const QString kUncategorised = "General";
const QString kAllCategories = "All";

QColor lerp(const QColor &a, const QColor &b, double t)
{
  t = std::clamp(t, 0.0, 1.0);
  return QColor::fromRgbF(a.redF() + (b.redF() - a.redF()) * t,
                          a.greenF() + (b.greenF() - a.greenF()) * t,
                          a.blueF() + (b.blueF() - a.blueF()) * t,
                          a.alphaF() + (b.alphaF() - a.alphaF()) * t);
}

/**
 * @brief The checkerboard as a cached 2x2-cell tile.
 *
 * This used to be a nested loop of drawRect, which for the bar plus a full
 * preset grid ran to roughly a thousand calls on every single repaint - and a
 * repaint happens on every frame of the section collapse animation. One
 * tiled fillRect does the same job.
 */
const QBrush &checker_brush()
{
  static const QBrush brush = []()
  {
    const int s = kGradientCheckSize;
    QPixmap   tile(2 * s, 2 * s);
    QPainter  p(&tile);
    p.fillRect(tile.rect(), kCheckA);
    p.fillRect(0, 0, s, s, kCheckB);
    p.fillRect(s, s, s, s, kCheckB);
    return QBrush(tile);
  }();

  return brush;
}

void paint_checkerboard(QPainter &p, const QRect &r)
{
  p.save();
  // anchor the pattern to the rect rather than to the window origin
  p.setBrushOrigin(r.topLeft());
  p.fillRect(r, checker_brush());
  p.restore();
}

void paint_gradient(QPainter &p, const QRect &r, const QVector<GradientStop> &stops)
{
  if (stops.isEmpty())
    return;

  QLinearGradient g(r.topLeft(), r.topRight());

  if (stops.size() == 1)
  {
    g.setColorAt(0.0, stops.front().color);
    g.setColorAt(1.0, stops.front().color);
  }
  else
  {
    for (const auto &s : stops)
      g.setColorAt(std::clamp(s.pos, 0.0, 1.0), s.color);
  }

  p.fillRect(r, g);
}

} // namespace

// ---------------------------------------------------------------------------
// ColorChip
// ---------------------------------------------------------------------------

/// The selected stop's colour, as a clickable swatch. Exists because the only
/// way to recolour a stop used to be double-clicking its marker, which nothing
/// on screen advertised.
class ColorChip : public QWidget
{
public:
  explicit ColorChip(QWidget *parent = nullptr) : QWidget(parent)
  {
    this->setFixedSize(34, kChipHeight);
    this->setCursor(Qt::PointingHandCursor);
  }

  void set_color(const QColor &c)
  {
    this->color_ = c;
    this->update();
  }

  void set_active(bool active)
  {
    this->active_ = active;
    this->update();
  }

  std::function<void()> on_click;

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter    p(this);
    const QRect r = this->rect().adjusted(0, 0, -1, -1);

    if (this->active_)
    {
      paint_checkerboard(p, this->rect());
      p.fillRect(this->rect(), this->color_);
    }
    else
      p.fillRect(this->rect(), kField);

    p.setPen(this->active_ ? kFieldBorder : kChipBorder);
    p.setBrush(Qt::NoBrush);
    p.drawRect(r);
  }

  void mouseReleaseEvent(QMouseEvent *e) override
  {
    if (e->button() == Qt::LeftButton && this->active_ && this->on_click &&
        this->rect().contains(e->pos()))
      this->on_click();
  }

private:
  QColor color_ = Qt::black;
  bool   active_ = false;
};

// ---------------------------------------------------------------------------
// GradientBar
// ---------------------------------------------------------------------------

/// Preview bar plus the stop markers underneath it. No Q_OBJECT: it reports
/// through std::function so it needs no moc pass of its own.
class GradientBar : public QWidget
{
public:
  explicit GradientBar(QWidget *parent = nullptr) : QWidget(parent)
  {
    this->setFixedHeight(kGradientBarHeight + kGradientGutter);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  }

  /**
   * @brief Replace the stops from outside (model sync, preset, shuffle).
   *
   * Ignored mid-drag: a value_changed round-trip comes back through the model
   * as a sync while the mouse is still down, and re-seating the vector under
   * the cursor drops the stop being dragged.
   */
  void set_stops(const QVector<GradientStop> &stops)
  {
    if (this->dragging_)
      return;

    this->stops_ = stops;
    this->sort();

    // keep the selection where it can be kept, so the colour chip does not go
    // dead every time the node recomputes
    if (this->selected_ >= this->stops_.size())
      this->selected_ = this->stops_.isEmpty() ? -1 : this->stops_.size() - 1;

    this->update();
    this->notify_selection();
  }

  const QVector<GradientStop> &stops() const { return this->stops_; }

  int selected() const { return this->selected_; }

  QColor selected_color() const
  {
    return this->has_selection() ? this->stops_[this->selected_].color : QColor();
  }

  bool has_selection() const
  {
    return this->selected_ >= 0 && this->selected_ < this->stops_.size();
  }

  bool can_remove() const { return this->has_selection() && this->stops_.size() > 2; }

  void set_selected_color(const QColor &c)
  {
    if (!this->has_selection())
      return;

    this->stops_[this->selected_].color = c;
    this->update();
    this->notify_selection();
    this->emit_changed();
    this->emit_committed();
  }

  /// Add a stop halfway into the widest gap, so repeated presses spread out
  /// instead of stacking on one spot.
  void add_stop()
  {
    double pos = 0.5;

    if (this->stops_.size() >= 2)
    {
      double widest = -1.0;
      for (int i = 1; i < this->stops_.size(); ++i)
      {
        const double gap = this->stops_[i].pos - this->stops_[i - 1].pos;
        if (gap > widest)
        {
          widest = gap;
          pos = 0.5 * (this->stops_[i].pos + this->stops_[i - 1].pos);
        }
      }
    }

    const GradientStop added{pos, this->color_at(pos)};
    this->stops_.push_back(added);
    this->sort();
    this->reselect(added);

    this->update();
    this->notify_selection();
    this->emit_changed();
    this->emit_committed();
  }

  void remove_selected()
  {
    if (!this->can_remove())
      return;

    this->stops_.remove(this->selected_);
    this->selected_ = -1;

    this->update();
    this->notify_selection();
    this->emit_changed();
    this->emit_committed();
  }

  std::function<void()> on_changed;
  std::function<void()> on_committed;
  std::function<void()> on_selection_changed;

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);

    const QRect bar = this->bar_rect();

    paint_checkerboard(p, bar);
    paint_gradient(p, bar, this->stops_);

    p.setPen(kHairline);
    p.setBrush(Qt::NoBrush);
    p.drawRect(bar.adjusted(0, 0, -1, -1));

    for (int i = 0; i < this->stops_.size(); ++i)
    {
      const QRect  m = this->marker_rect(i);
      const bool   sel = (i == this->selected_);
      const QColor c = this->stops_[i].color;

      // tick joining the marker to the bar it belongs to
      p.setPen(sel ? kAccent : kThumbBorder);
      p.drawLine(m.center().x(), bar.bottom(), m.center().x(), m.top());

      p.setPen(QPen(sel ? kAccent : kThumbBorder, 1));
      p.setBrush(QColor(c.red(), c.green(), c.blue())); // alpha reads in the bar
      p.drawRoundedRect(m.adjusted(0, 0, -1, -1), 2, 2);
    }
  }

  void mousePressEvent(QMouseEvent *e) override
  {
    const int hit = this->hit_test(e->pos());

    if (e->button() == Qt::RightButton)
    {
      if (hit >= 0)
      {
        this->selected_ = hit;
        this->remove_selected();
      }
      return;
    }

    if (e->button() != Qt::LeftButton)
      return;

    this->selected_ = hit;
    this->dragging_ = (hit >= 0);
    this->update();
    this->notify_selection();
  }

  void mouseMoveEvent(QMouseEvent *e) override
  {
    if (!this->dragging_ || !this->has_selection())
      return;

    this->stops_[this->selected_].pos = this->pos_at(e->pos().x());
    this->update();
    this->emit_changed();
  }

  void mouseReleaseEvent(QMouseEvent *e) override
  {
    if (e->button() != Qt::LeftButton || !this->dragging_)
      return;

    this->dragging_ = false;

    // Sorting only on release keeps `selected_` valid for the whole drag; the
    // painter and QLinearGradient are both order-independent anyway.
    const GradientStop moved = this->has_selection() ? this->stops_[this->selected_]
                                                     : GradientStop{};
    this->sort();
    this->reselect(moved);

    this->update();
    this->notify_selection();
    this->emit_committed();
  }

  void mouseDoubleClickEvent(QMouseEvent *e) override
  {
    if (e->button() != Qt::LeftButton)
      return;

    const int hit = this->hit_test(e->pos());

    if (hit >= 0)
    {
      this->selected_ = hit;
      this->notify_selection();

      const QColor picked = QColorDialog::getColor(this->stops_[hit].color,
                                                   this,
                                                   "Stop colour",
                                                   QColorDialog::ShowAlphaChannel);
      if (picked.isValid())
        this->set_selected_color(picked);
      return;
    }

    if (this->bar_rect().contains(e->pos()))
    {
      const double       pos = this->pos_at(e->pos().x());
      const GradientStop added{pos, this->color_at(pos)};

      this->stops_.push_back(added);
      this->sort();
      this->reselect(added);

      this->update();
      this->notify_selection();
      this->emit_changed();
      this->emit_committed();
    }
  }

private:
  QRect bar_rect() const { return QRect(0, 0, this->width(), kGradientBarHeight); }

  QRect marker_rect(int i) const
  {
    const QRect  bar = this->bar_rect();
    const double x = bar.left() +
                     std::clamp(this->stops_[i].pos, 0.0, 1.0) * (bar.width() - 1);

    return QRect(static_cast<int>(std::lround(x)) - kGradientStopW / 2,
                 kGradientBarHeight + 2,
                 kGradientStopW,
                 kGradientStopH);
  }

  double pos_at(int x) const
  {
    const QRect bar = this->bar_rect();
    if (bar.width() <= 1)
      return 0.0;
    return std::clamp(static_cast<double>(x - bar.left()) / (bar.width() - 1), 0.0, 1.0);
  }

  int hit_test(const QPoint &pt) const
  {
    // reverse so the marker painted last (on top) wins an overlap
    for (int i = this->stops_.size() - 1; i >= 0; --i)
      if (this->marker_rect(i).adjusted(-3, -3, 3, 3).contains(pt))
        return i;
    return -1;
  }

  QColor color_at(double pos) const
  {
    if (this->stops_.isEmpty())
      return Qt::black;

    QVector<GradientStop> s = this->stops_;
    std::sort(s.begin(), s.end(), [](const auto &a, const auto &b)
              { return a.pos < b.pos; });

    if (pos <= s.front().pos)
      return s.front().color;
    if (pos >= s.back().pos)
      return s.back().color;

    for (int i = 1; i < s.size(); ++i)
      if (pos <= s[i].pos)
      {
        const double span = s[i].pos - s[i - 1].pos;
        const double t = (span > 0.0) ? (pos - s[i - 1].pos) / span : 0.0;
        return lerp(s[i - 1].color, s[i].color, t);
      }

    return s.back().color;
  }

  void sort()
  {
    std::sort(this->stops_.begin(),
              this->stops_.end(),
              [](const auto &a, const auto &b) { return a.pos < b.pos; });
  }

  /// Re-find `s` after a sort so the selection follows the stop, not the slot.
  void reselect(const GradientStop &s)
  {
    for (int i = 0; i < this->stops_.size(); ++i)
      if (this->stops_[i].pos == s.pos && this->stops_[i].color == s.color)
      {
        this->selected_ = i;
        return;
      }
    this->selected_ = -1;
  }

  void notify_selection()
  {
    if (this->on_selection_changed)
      this->on_selection_changed();
  }

  void emit_changed()
  {
    if (this->on_changed)
      this->on_changed();
  }

  void emit_committed()
  {
    if (this->on_committed)
      this->on_committed();
  }

  QVector<GradientStop> stops_;
  int                   selected_ = -1;
  bool                  dragging_ = false;
};

// ---------------------------------------------------------------------------
// GradientPresetGrid
// ---------------------------------------------------------------------------

/// Presets as a wrapping grid. The strip this replaces scrolled sideways, so
/// most of the library was permanently off-screen; here the row count follows
/// the width and the whole (filtered) set is visible at once.
class GradientPresetGrid : public QWidget
{
public:
  explicit GradientPresetGrid(QWidget *parent = nullptr) : QWidget(parent)
  {
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    this->setMouseTracking(true);
  }

  void set_presets(const QVector<GradientPreset> &presets)
  {
    this->presets_ = presets;
    this->current_ = -1;
    this->hovered_ = -1;
    this->cache_ = QPixmap();
    this->apply_height();
    this->update();
  }

  bool is_empty() const { return this->presets_.isEmpty(); }

  void clear_current()
  {
    if (this->current_ != -1)
    {
      this->current_ = -1;
      this->update();
    }
  }

  std::function<void(int)> on_picked;

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);

    // The fills never change unless the presets or the width do, but forty
    // QLinearGradients plus forty checkerboards per repaint is far too much to
    // redo on every animation frame. Cache them; only the state borders are
    // cheap enough to paint live.
    if (this->cache_.isNull())
      this->rebuild_cache();

    if (!this->cache_.isNull())
      p.drawPixmap(0, 0, this->cache_);

    for (int i = 0; i < this->presets_.size(); ++i)
    {
      // selection is a border, never a fill
      const QColor border = (i == this->current_)   ? kAccent
                            : (i == this->hovered_) ? kFieldBorderHover
                                                    : kHairline;

      p.setPen(QPen(border, 1));
      p.setBrush(Qt::NoBrush);
      p.drawRect(this->swatch_rect(i).adjusted(0, 0, -1, -1));
    }
  }

  void resizeEvent(QResizeEvent *event) override
  {
    QWidget::resizeEvent(event);

    // Only the width changes the layout. Recomputing on a height-only resize
    // re-invalidates the parent layout from inside a layout pass, which turns
    // one pass per animation frame into several.
    if (event->oldSize().width() == event->size().width())
      return;

    this->cache_ = QPixmap();
    this->apply_height();
  }

  void mouseMoveEvent(QMouseEvent *e) override
  {
    const int hit = this->hit_test(e->pos());
    if (hit == this->hovered_)
      return;

    this->hovered_ = hit;
    // the preset name is a tooltip, never a caption - the names are opaque ids
    this->setToolTip(hit >= 0 ? this->presets_[hit].name : QString());
    this->update();
  }

  void leaveEvent(QEvent *) override
  {
    this->hovered_ = -1;
    this->update();
  }

  void mousePressEvent(QMouseEvent *e) override
  {
    if (e->button() != Qt::LeftButton)
      return;

    const int hit = this->hit_test(e->pos());
    if (hit < 0)
      return;

    this->current_ = hit;
    this->update();

    if (this->on_picked)
      this->on_picked(hit);
  }

private:
  int columns() const
  {
    if (this->width() <= 0)
      return 1;
    return std::max(1, (this->width() + kSwatchGap) / (kSwatchMinWidth + kSwatchGap));
  }

  int rows() const
  {
    if (this->presets_.isEmpty())
      return 0;
    const int cols = this->columns();
    return (this->presets_.size() + cols - 1) / cols;
  }

  QRect swatch_rect(int i) const
  {
    const int cols = this->columns();
    const int col = i % cols;
    const int row = i / cols;

    // distribute the rounding remainder so the last column lands on the edge
    const double total = this->width() - (cols - 1) * kSwatchGap;
    const double w = total / cols;

    const int x = static_cast<int>(std::lround(col * (w + kSwatchGap)));
    const int right = static_cast<int>(std::lround(col * (w + kSwatchGap) + w));

    return QRect(x, row * (kSwatchHeight + kSwatchGap), right - x, kSwatchHeight);
  }

  int hit_test(const QPoint &pt) const
  {
    for (int i = 0; i < this->presets_.size(); ++i)
      if (this->swatch_rect(i).contains(pt))
        return i;
    return -1;
  }

  void apply_height()
  {
    const int r = this->rows();
    const int h = (r == 0) ? 0 : r * kSwatchHeight + (r - 1) * kSwatchGap;
    if (h != this->height())
      this->setFixedHeight(h);
  }

  /// Render every swatch fill once, at device resolution.
  void rebuild_cache()
  {
    if (this->presets_.isEmpty() || this->width() <= 0 || this->height() <= 0)
      return;

    const qreal dpr = this->devicePixelRatioF();

    this->cache_ = QPixmap(static_cast<int>(this->width() * dpr),
                           static_cast<int>(this->height() * dpr));
    this->cache_.setDevicePixelRatio(dpr);
    this->cache_.fill(Qt::transparent);

    QPainter p(&this->cache_);
    p.setRenderHint(QPainter::Antialiasing, false);

    for (int i = 0; i < this->presets_.size(); ++i)
    {
      const QRect r = this->swatch_rect(i);
      paint_checkerboard(p, r);
      paint_gradient(p, r, this->presets_[i].stops);
    }
  }

  QVector<GradientPreset> presets_;
  int                     current_ = -1;
  int                     hovered_ = -1;
  QPixmap                 cache_;
};

// ---------------------------------------------------------------------------
// HGradient
// ---------------------------------------------------------------------------

namespace
{

/// Uppercase caption above a control, matching the panel's row labels.
QLabel *make_caption(const QString &text, QWidget *parent)
{
  auto *l = new QLabel(text, parent);
  QFont f = l->font();
  f.setPixelSize(kLabelPx);
  f.setCapitalization(QFont::AllUppercase);
  f.setLetterSpacing(QFont::AbsoluteSpacing, 1.0);
  l->setFont(f);
  l->setStyleSheet(QString("color: %1; background: transparent;").arg(kInkDefault.name()));
  return l;
}

} // namespace

HGradient::HGradient(QWidget *parent) : QWidget(parent)
{
  auto *box = new QVBoxLayout(this);
  box->setContentsMargins(0, 0, 0, 0);
  box->setSpacing(10);

  // --- bar

  this->bar_ = new GradientBar(this);
  box->addWidget(this->bar_);

  // --- stop controls
  //
  // Double-click on a marker still opens the colour dialog, but nothing on
  // screen said so, so recolouring a stop was effectively undiscoverable.

  {
    auto *row = new QWidget(this);
    auto *h = new QHBoxLayout(row);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(6);

    this->chip_ = new ColorChip(row);
    this->chip_->on_click = [this]() { this->edit_selected_color(); };

    this->color_btn_ = new ModButton("Colour", false, kAccent, row);
    this->add_btn_ = new ModButton("Add", false, kAccent, row);
    this->remove_btn_ = new ModButton("Remove", false, kAccent, row);

    this->connect(this->color_btn_,
                  &ModButton::clicked,
                  this,
                  [this]() { this->edit_selected_color(); });

    this->connect(this->add_btn_,
                  &ModButton::clicked,
                  this,
                  [this]() { this->bar_->add_stop(); });

    this->connect(this->remove_btn_,
                  &ModButton::clicked,
                  this,
                  [this]() { this->bar_->remove_selected(); });

    h->addWidget(this->chip_);
    h->addWidget(this->color_btn_);
    h->addWidget(this->add_btn_);
    h->addWidget(this->remove_btn_);
    h->addStretch();

    box->addWidget(row);
  }

  // --- category filter

  {
    this->category_row_ = new QWidget(this);
    auto *v = new QVBoxLayout(this->category_row_);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(6);

    this->category_combo_ = new HCombo(this->category_row_);
    this->connect(this->category_combo_,
                  &HCombo::activated,
                  this,
                  [this](int) { this->apply_filter(); });

    v->addWidget(make_caption("Category", this->category_row_));
    v->addWidget(this->category_combo_);

    this->category_row_->hide(); // shown only once there is a choice to make
    box->addWidget(this->category_row_);
  }

  // --- preset grid

  this->grid_ = new GradientPresetGrid(this);
  this->grid_->on_picked = [this](int i)
  {
    if (i < 0 || i >= this->filtered_.size())
      return;

    this->bar_->set_stops(this->filtered_[i].stops);
    this->sync_stop_controls();
    Q_EMIT this->value_changed();
    Q_EMIT this->edit_ended();
  };
  box->addWidget(this->grid_);

  // --- library actions

  {
    auto *row = new QWidget(this);
    auto *h = new QHBoxLayout(row);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(6);

    auto *shuffle = new ModButton("Shuffle Colors", false, kAccent, row);
    auto *save = new ModButton("Save as Preset...", false, kAccent, row);

    this->connect(shuffle,
                  &ModButton::clicked,
                  this,
                  [this]()
                  {
                    QVector<GradientStop> s = this->bar_->stops();
                    if (s.size() < 2)
                      return;

                    // permute the colours, leave the positions alone: the
                    // shape of the ramp is the user's, only the palette is
                    // rerolled
                    for (int i = s.size() - 1; i > 0; --i)
                    {
                      const int j = QRandomGenerator::global()->bounded(i + 1);
                      std::swap(s[i].color, s[j].color);
                    }

                    this->bar_->set_stops(s);
                    this->grid_->clear_current();
                    this->sync_stop_controls();
                    Q_EMIT this->value_changed();
                    Q_EMIT this->edit_ended();
                  });

    this->connect(save,
                  &ModButton::clicked,
                  this,
                  [this]() { this->save_current_as_preset(); });

    h->addWidget(shuffle);
    h->addWidget(save);
    h->addStretch();

    box->addWidget(row);
  }

  // --- wiring

  this->bar_->on_changed = [this]()
  {
    this->grid_->clear_current();
    Q_EMIT this->value_changed();
  };

  this->bar_->on_committed = [this]() { Q_EMIT this->edit_ended(); };

  this->bar_->on_selection_changed = [this]() { this->sync_stop_controls(); };

  this->sync_stop_controls();
}

void HGradient::set_stops(const QVector<GradientStop> &stops)
{
  this->bar_->set_stops(stops);
  this->sync_stop_controls();
}

QVector<GradientStop> HGradient::stops() const { return this->bar_->stops(); }

void HGradient::sync_stop_controls()
{
  const bool has = this->bar_->has_selection();

  this->chip_->set_active(has);
  if (has)
    this->chip_->set_color(this->bar_->selected_color());

  // A disabled-looking button that still fires is worse than one that visibly
  // cannot apply, so gate on the same condition the action checks.
  this->color_btn_->setEnabled(has);
  this->remove_btn_->setEnabled(this->bar_->can_remove());
}

void HGradient::edit_selected_color()
{
  if (!this->bar_->has_selection())
    return;

  const QColor picked = QColorDialog::getColor(this->bar_->selected_color(),
                                               this,
                                               "Stop colour",
                                               QColorDialog::ShowAlphaChannel);
  if (picked.isValid())
  {
    this->bar_->set_selected_color(picked);
    this->grid_->clear_current();
  }
}

void HGradient::set_presets(const QVector<GradientPreset> &presets)
{
  this->presets_ = presets;
  this->refresh_categories();
  this->apply_filter();
}

void HGradient::refresh_categories()
{
  const QString previous = (this->category_combo_ &&
                            this->category_combo_->current() >= 0 &&
                            this->category_combo_->current() < this->categories_.size())
                               ? this->categories_[this->category_combo_->current()]
                               : QString();

  this->categories_.clear();
  this->categories_ << kAllCategories;

  for (const auto &p : this->presets_)
  {
    const QString cat = p.category.isEmpty() ? kUncategorised : p.category;
    if (!this->categories_.contains(cat))
      this->categories_ << cat;
  }

  // "All" plus a single category is not a choice, so do not spend a row on it
  const bool worth_showing = this->categories_.size() > 2;
  this->category_row_->setVisible(worth_showing);

  this->category_combo_->set_options(this->categories_);

  const int restored = this->categories_.indexOf(previous);
  this->category_combo_->set_current(restored >= 0 ? restored : 0);
}

void HGradient::apply_filter()
{
  const int idx = this->category_combo_ ? this->category_combo_->current() : 0;
  const QString want = (idx > 0 && idx < this->categories_.size())
                           ? this->categories_[idx]
                           : QString();

  this->filtered_.clear();

  for (const auto &p : this->presets_)
  {
    const QString cat = p.category.isEmpty() ? kUncategorised : p.category;
    if (want.isEmpty() || cat == want)
      this->filtered_ << p;
  }

  this->grid_->set_presets(this->filtered_);
  this->grid_->setVisible(!this->grid_->is_empty());
}

void HGradient::save_current_as_preset()
{
  if (!this->on_save_preset)
    return;

  // default to whatever category is being browsed, so saving while filtered to
  // "rocky" files the new gradient next to the ones it was derived from
  const int     idx = this->category_combo_ ? this->category_combo_->current() : 0;
  const QString default_cat = (idx > 0 && idx < this->categories_.size())
                                  ? this->categories_[idx]
                                  : QString("custom");

  bool          ok = false;
  const QString text = QInputDialog::getText(
      this,
      "Save gradient preset",
      "Name, as category/name:",
      QLineEdit::Normal,
      default_cat + "/my gradient",
      &ok);

  if (!ok)
    return;

  QString   name = text.trimmed();
  QString   category = default_cat;
  const int slash = name.lastIndexOf('/');

  if (slash >= 0)
  {
    category = name.left(slash).trimmed();
    name = name.mid(slash + 1).trimmed();
  }

  if (name.isEmpty())
    return;

  if (!this->on_save_preset(category, name, this->bar_->stops()))
    return;

  if (this->on_reload_presets)
    this->set_presets(this->on_reload_presets());
}

} // namespace hesiod::pp
