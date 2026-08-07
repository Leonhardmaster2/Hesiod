/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <map>
#include <typeindex>
#include <vector>

#include <QLabel>
#include <QVBoxLayout>

#include "meta_common.hpp"
#include "meta_qt/widget_renderer.hpp"

#include "meta/ext/color_gradient/color_gradient.hpp"

#include "hesiod/gui/widgets/properties/check_row.hpp"
#include "hesiod/gui/widgets/properties/h_combo.hpp"
#include "hesiod/gui/widgets/properties/h_gradient.hpp"
#include "hesiod/gui/widgets/properties/h_range.hpp"
#include "hesiod/gui/widgets/properties/param_slider.hpp"
#include "hesiod/gui/widgets/properties/pp_section.hpp"
#include "hesiod/gui/widgets/properties/properties_panel.hpp"

namespace hesiod::pp
{

PropertiesPanel::PropertiesPanel(meta::AttributeContainer  *p_container,
                                 const nlohmann::json      &initial_state,
                                 const GradientPresetStore &preset_store,
                                 QWidget                   *parent)
    : QWidget(parent), p_container_(p_container), initial_state_(initial_state),
      preset_store_(preset_store)
{
  // ~60ms: fast enough that a drag still feels live, slow enough that a
  // 260ms glide costs about four node updates instead of sixteen
  this->recompute_timer_ = new QTimer(this);
  this->recompute_timer_->setSingleShot(true);
  this->recompute_timer_->setInterval(150);
  this->connect(this->recompute_timer_,
                &QTimer::timeout,
                this,
                [this]() { Q_EMIT this->value_changed(); });

  this->build();
}

void PropertiesPanel::request_recompute()
{
  // Debounce, not throttle: restarting on every event means a continuous drag
  // emits nothing until the user actually stops. Needed because not every Meta
  // fallback widget emits edit_ended - relying on that alone left some
  // attributes (the remap range among them) unable to update at all.
  this->recompute_timer_->start();
}

void PropertiesPanel::flush_recompute()
{
  this->recompute_timer_->stop();
  Q_EMIT this->value_changed();
}

double PropertiesPanel::default_for(const std::string &name, double fallback) const
{
  // AttributeContainer::json_to() nests the value: { name: { type, value } }
  if (!this->initial_state_.is_object() || !this->initial_state_.contains(name))
    return fallback;

  const nlohmann::json &entry = this->initial_state_.at(name);
  if (entry.is_object() && entry.contains("value") && entry["value"].is_number())
    return entry["value"].get<double>();
  if (entry.is_number())
    return entry.get<double>();

  return fallback;
}

void PropertiesPanel::sync_from_model()
{
  for (const auto &fn : this->syncers_)
    fn();
}

// ---------------------------------------------------------------------------

void PropertiesPanel::add_section(QWidget *section)
{
  if (!section || !this->outer_)
    return;

  section->setParent(this);
  // insert before the trailing stretch so the section joins the stack rather
  // than being pushed to the bottom of the panel
  this->outer_->insertWidget(this->outer_->count() - 1, section);
  ++this->section_count_;
}

void PropertiesPanel::build()
{
  auto *outer = new QVBoxLayout(this);
  outer->setContentsMargins(0, 0, 0, 0);
  outer->setSpacing(0);
  this->outer_ = outer;

  // the stretch is added up front so add_section() has something to insert
  // before even when the container is empty or every attribute is hidden
  outer->addStretch();

  if (!this->p_container_)
    return;

  // group by Meta category, preserving first-seen order so the panel reads in
  // the order the node author declared its parameters
  std::vector<std::string>                        category_order;
  std::map<std::string, std::vector<std::string>> by_category;

  for (const std::string &name : this->p_container_->insertion_order())
  {
    meta::AbstractAttribute *p_attr = this->p_container_->find(name);
    if (!p_attr)
      continue;

    std::string cat = meta::common::category(*p_attr);
    if (cat.empty())
      cat = "General";

    if (!by_category.count(cat))
      category_order.push_back(cat);
    by_category[cat].push_back(name);
  }

  int index = 0;

  for (const std::string &cat : category_order)
  {
    const QString idx = QString("%1").arg(++index, 2, 10, QChar('0'));

    auto *section = new PpSection(QString::fromStdString(cat),
                                  idx,
                                  *group_accent(index - 1),
                                  this);

    int rows = 0;
    for (const std::string &name : by_category[cat])
    {
      if (QWidget *row = this->make_row(this->p_container_->find(name)))
      {
        section->body_layout()->addWidget(row);
        ++rows;
      }
    }

    if (rows == 0)
    {
      delete section;
      --index;
      continue;
    }

    outer->insertWidget(outer->count() - 1, section);
  }

  this->section_count_ = index;
}

QWidget *PropertiesPanel::make_row(meta::AbstractAttribute *p_attr)
{
  if (!p_attr)
    return nullptr;

  // widget_type() only has a typed overload, so read the metadata key directly
  // - the value is what Meta's own renderers switch on.
  if (meta::common::try_get<std::string>(*p_attr,
                                         meta::keys::ui::widget_type,
                                         std::string{}) == "None")
    return nullptr;

  const std::type_index t = p_attr->type();

  if (t == std::type_index(typeid(meta::ColorGradient)))
    if (QWidget *w = this->make_gradient_row(p_attr))
      return w;

  if (t == std::type_index(typeid(glm::vec2)))
    if (QWidget *w = this->make_range_row(p_attr))
      return w;
  if (t == std::type_index(typeid(std::string)))
    if (QWidget *w = this->make_choice_row(p_attr))
      return w;
  if (t == std::type_index(typeid(int)))
    if (QWidget *w = this->make_enum_row(p_attr))
      return w;

  if (t == std::type_index(typeid(float)))
    if (QWidget *w = this->make_float_row(p_attr))
      return w;
  if (t == std::type_index(typeid(int)))
    if (QWidget *w = this->make_int_row(p_attr))
      return w;
  if (t == std::type_index(typeid(bool)))
    if (QWidget *w = this->make_bool_row(p_attr))
      return w;

  // Not ported yet: fall back to Meta's own renderer so the attribute stays
  // editable. These rows keep the stock look until their widget lands.
  meta::qt::MetaWidget *fallback = meta::qt::render(p_attr, this);
  if (!fallback)
    return nullptr;

  this->syncers_.push_back([fallback]() { fallback->sync_widget_from_model(); });

  this->connect(fallback,
                &meta::qt::MetaWidget::value_changed,
                this,
                [this]()
                {
                  // Debounced: nothing fires until the drag pauses. edit_ended
                  // still flushes immediately when the widget reports one.
                  this->request_recompute();
                });
  this->connect(fallback,
                &meta::qt::MetaWidget::edit_ended,
                this,
                [this]()
                {
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  return fallback;
}

// ---------------------------------------------------------------------------

QWidget *PropertiesPanel::make_labeled(const QString &label, QWidget *control)
{
  if (label.isEmpty())
    return control;

  auto *wrap = new QWidget(this);
  auto *box = new QVBoxLayout(wrap);
  box->setContentsMargins(0, 0, 0, 0);
  box->setSpacing(6);

  auto *text = new QLabel(label, wrap);
  QFont f = text->font();
  f.setPixelSize(kLabelPx);
  f.setCapitalization(QFont::AllUppercase);
  f.setLetterSpacing(QFont::AbsoluteSpacing, 1.0);
  text->setFont(f);
  text->setStyleSheet(QString("color: %1; background: transparent;")
                          .arg(kInkDefault.name()));

  box->addWidget(text);
  box->addWidget(control);
  return wrap;
}

QWidget *PropertiesPanel::make_choice_row(meta::AbstractAttribute *p_attr)
{
  auto *typed = p_attr->try_cast<meta::Attribute<std::string>>();
  if (!typed)
    return nullptr;

  const std::vector<std::string> options = meta::common::allowed_values(*typed);
  if (options.empty())
    return nullptr; // free-form string, not a choice - let Meta render it

  QStringList qopts;
  int         current = 0;
  for (int i = 0; i < static_cast<int>(options.size()); ++i)
  {
    qopts << QString::fromStdString(options[i]);
    if (options[i] == typed->value())
      current = i;
  }

  auto *combo = new HCombo(this);
  combo->set_options(qopts);
  combo->set_current(current);

  this->syncers_.push_back(
      [combo, typed, options]()
      {
        for (int i = 0; i < static_cast<int>(options.size()); ++i)
          if (options[i] == typed->value())
            combo->set_current(i);
      });

  this->connect(combo,
                &HCombo::activated,
                this,
                [this, typed, options](int idx)
                {
                  if (idx < 0 || idx >= static_cast<int>(options.size()))
                    return;
                  typed->set_from_any(options[idx]);
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  return this->make_labeled(QString::fromStdString(meta::common::label(*typed)),
                            combo);
}

QWidget *PropertiesPanel::make_enum_row(meta::AbstractAttribute *p_attr)
{
  auto *typed = p_attr->try_cast<meta::Attribute<int>>();
  if (!typed)
    return nullptr;

  const auto items = meta::common::enum_items<int>(*typed);
  if (items.empty())
    return nullptr; // plain int, falls through to the rail

  QStringList qopts;
  int         current = 0;
  for (int i = 0; i < static_cast<int>(items.size()); ++i)
  {
    qopts << QString::fromStdString(items[i].second);
    if (items[i].first == typed->value())
      current = i;
  }

  auto *combo = new HCombo(this);
  combo->set_options(qopts);
  combo->set_current(current);

  this->syncers_.push_back(
      [combo, typed, items]()
      {
        for (int i = 0; i < static_cast<int>(items.size()); ++i)
          if (items[i].first == typed->value())
            combo->set_current(i);
      });

  this->connect(combo,
                &HCombo::activated,
                this,
                [this, typed, items](int idx)
                {
                  if (idx < 0 || idx >= static_cast<int>(items.size()))
                    return;
                  typed->set_from_any(items[idx].first);
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  return this->make_labeled(QString::fromStdString(meta::common::label(*typed)),
                            combo);
}

QWidget *PropertiesPanel::make_range_row(meta::AbstractAttribute *p_attr)
{
  // glm::vec2 also backs the XY pad; only the RangeBar flavour belongs here
  if (meta::common::try_get<std::string>(*p_attr,
                                         meta::keys::ui::widget_type,
                                         std::string{}) != "RangeBar")
    return nullptr;

  auto *typed = p_attr->try_cast<meta::Attribute<glm::vec2>>();
  if (!typed)
    return nullptr;

  const float min = meta::common::min<float>(*typed);
  const float max = meta::common::max<float>(*typed);
  if (!(max > min))
    return nullptr;

  const glm::vec2 v = typed->value();

  // Meta encodes "disabled" as the sentinel (-1, 0) rather than a flag, so the
  // last meaningful range has to be remembered here and restored on re-enable
  // - otherwise toggling off destroys the user's range.
  const bool active = !(v.x == -1.f && v.y == 0.f);
  auto  last = std::make_shared<glm::vec2>(active ? v : glm::vec2{min, max});

  auto *row = new HRange(this);
  row->set_bounds(min,
                  max,
                  meta::common::try_get_format_decimals(
                      meta::common::format(*typed)));
  row->set_range(last->x, last->y);
  row->set_active(active);

  this->syncers_.push_back(
      [row, typed, last]()
      {
        const glm::vec2 cur = typed->value();
        const bool      on = !(cur.x == -1.f && cur.y == 0.f);
        if (on)
        {
          *last = cur;
          row->set_range(cur.x, cur.y);
        }
        row->set_active(on);
      });

  auto commit = [this, row, typed, last]()
  {
    if (row->is_active())
    {
      *last = glm::vec2{static_cast<float>(row->lo()),
                        static_cast<float>(row->hi())};
      typed->set_from_any(*last);
    }
    else
      typed->set_from_any(glm::vec2{-1.f, 0.f});

    this->flush_recompute();
    Q_EMIT this->edit_ended();
  };

  this->connect(row, &HRange::committed, this, commit);
  this->connect(row, &HRange::active_toggled, this, [commit](bool) { commit(); });

  const std::string label = meta::common::label(*typed);
  return label.empty() ? row
                       : this->make_labeled(QString::fromStdString(label), row);
}

QWidget *PropertiesPanel::make_gradient_row(meta::AbstractAttribute *p_attr)
{
  auto *typed = p_attr->try_cast<meta::Attribute<meta::ColorGradient>>();
  if (!typed)
    return nullptr;

  // meta::Stop <-> GradientStop. HGradient is deliberately Qt-only, so the
  // translation lives here rather than in the widget.
  auto to_qt = [](const std::vector<meta::Stop> &in)
  {
    QVector<GradientStop> out;
    out.reserve(static_cast<int>(in.size()));
    for (const auto &s : in)
      out.push_back({static_cast<double>(s.position),
                     QColor::fromRgbF(s.color[0], s.color[1], s.color[2], s.color[3])});
    return out;
  };

  auto from_qt = [](const QVector<GradientStop> &in)
  {
    std::vector<meta::Stop> out;
    out.reserve(static_cast<size_t>(in.size()));
    for (const auto &s : in)
      out.push_back({static_cast<float>(s.pos),
                     {static_cast<float>(s.color.redF()),
                      static_cast<float>(s.color.greenF()),
                      static_cast<float>(s.color.blueF()),
                      static_cast<float>(s.color.alphaF())}});
    return out;
  };

  auto *row = new HGradient(this);
  row->set_stops(to_qt(typed->value().value()));

  // meta::Preset carries a name and stops, with nowhere to put a category, and
  // adding a field would mean editing the Meta submodule. So the host encodes
  // the category into the name as "category/name" and it is split back out
  // here. Meta's own renderer just shows the full path, which is harmless.
  auto split_preset = [](const meta::Preset &p)
  {
    QString       name = QString::fromStdString(p.name);
    QString       category;
    const int     slash = name.lastIndexOf('/');

    if (slash >= 0)
    {
      category = name.left(slash);
      name = name.mid(slash + 1);
    }
    return std::pair<QString, QString>{category, name};
  };

  // Presets are host configuration carried in metadata, not part of the value,
  // so an empty library just means no grid.
  if (const auto *p_presets = typed->metadata().try_value<meta::GradientPresets>(
          meta::keys::ui::presets))
  {
    QVector<GradientPreset> presets;
    presets.reserve(static_cast<int>(p_presets->presets.size()));
    for (const auto &p : p_presets->presets)
    {
      const auto [category, name] = split_preset(p);
      presets.push_back({category, name, to_qt(p.stops)});
    }

    row->set_presets(presets);
  }

  // Saving a user preset is the host's job; without a store the library stays
  // read-only and the save button is simply inert.
  if (this->preset_store_.save)
    row->on_save_preset = this->preset_store_.save;

  if (this->preset_store_.reload)
    row->on_reload_presets = this->preset_store_.reload;

  this->syncers_.push_back([row, typed, to_qt]()
                           { row->set_stops(to_qt(typed->value().value())); });

  auto write_back = [row, typed, from_qt]()
  {
    meta::ColorGradient cg;
    cg.set_value(from_qt(row->stops()));
    typed->set_from_any(cg);
  };

  this->connect(row,
                &HGradient::value_changed,
                this,
                [this, write_back]()
                {
                  write_back();
                  this->request_recompute();
                });

  this->connect(row,
                &HGradient::edit_ended,
                this,
                [this, write_back]()
                {
                  write_back();
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  const std::string label = meta::common::label(*typed);
  return label.empty() ? row
                       : this->make_labeled(QString::fromStdString(label), row);
}

QWidget *PropertiesPanel::make_float_row(meta::AbstractAttribute *p_attr)
{
  auto *typed = p_attr->try_cast<meta::Attribute<float>>();
  if (!typed)
    return nullptr;

  // A rail needs bounds. Without min/max metadata meta::common::min/max both
  // return T{} == 0, so the range collapses and EVERY value - including a
  // typed one - clamps to zero. Meta's own renderer refuses this case too;
  // hand it back so the attribute gets a spin box instead.
  if (!typed->metadata().contains_all_keys(
          {meta::keys::constraints::min, meta::keys::constraints::max}))
    return nullptr;

  const float min = meta::common::min(*typed);
  const float max = meta::common::max(*typed);
  const float step = meta::common::step(*typed);

  if (!(max > min))
    return nullptr;

  auto *row = new ParamSlider(this);
  row->set_label(QString::fromStdString(meta::common::label(*typed)));
  // step drives the wheel only; dragging a float stays continuous
  row->set_range(min, max, step > 0.f ? step : (max - min) / 200.f);
  row->set_quantized(false);
  row->set_decimals(
      meta::common::try_get_format_decimals(meta::common::format(*typed)));
  row->set_suffix("");
  row->set_default_value(this->default_for(p_attr->name(), typed->value()));
  row->set_value(typed->value());

  this->syncers_.push_back([row, typed]() { row->set_value(typed->value()); });

  this->connect(row,
                &ParamSlider::value_changed,
                this,
                [row]()
                {
                  // Intentionally empty: the row repaints itself. Writing the
                  // attribute here would fire Meta's own value_changed event
                  // and recompute the node on every drag tick.
                  Q_UNUSED(row);
                });
  this->connect(row,
                &ParamSlider::edit_ended,
                this,
                [this, row, typed]()
                {
                  typed->set_from_any(static_cast<float>(row->value()));
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  return row;
}

QWidget *PropertiesPanel::make_int_row(meta::AbstractAttribute *p_attr)
{
  auto *typed = p_attr->try_cast<meta::Attribute<int>>();
  if (!typed)
    return nullptr;

  // A rail needs bounds. Without min/max metadata meta::common::min/max both
  // return T{} == 0, so the range collapses and EVERY value - including a
  // typed one - clamps to zero. Meta's own renderer refuses this case too;
  // hand it back so the attribute gets a spin box instead.
  if (!typed->metadata().contains_all_keys(
          {meta::keys::constraints::min, meta::keys::constraints::max}))
    return nullptr;

  const int min = meta::common::min(*typed);
  const int max = meta::common::max(*typed);

  if (!(max > min))
    return nullptr;

  auto *row = new ParamSlider(this);
  row->set_label(QString::fromStdString(meta::common::label(*typed)));
  row->set_range(min, max, 1.0);
  row->set_quantized(true);
  row->set_decimals(0);
  row->set_suffix("");
  row->set_default_value(this->default_for(p_attr->name(), typed->value()));
  row->set_value(typed->value());

  this->syncers_.push_back([row, typed]() { row->set_value(typed->value()); });

  this->connect(row,
                &ParamSlider::value_changed,
                this,
                [row]() { Q_UNUSED(row); });
  this->connect(row,
                &ParamSlider::edit_ended,
                this,
                [this, row, typed]()
                {
                  typed->set_from_any(static_cast<int>(std::lround(row->value())));
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  return row;
}

QWidget *PropertiesPanel::make_bool_row(meta::AbstractAttribute *p_attr)
{
  auto *typed = p_attr->try_cast<meta::Attribute<bool>>();
  if (!typed)
    return nullptr;

  auto *row = new CheckRow(QString::fromStdString(meta::common::label(*typed)),
                           typed->value(),
                           kAccent,
                           this);

  this->syncers_.push_back([row, typed]()
                           { row->set_checked(typed->value(), false); });

  this->connect(row,
                &CheckRow::toggled,
                this,
                [this, typed](bool checked)
                {
                  typed->set_from_any(checked);
                  this->flush_recompute();
                  Q_EMIT this->edit_ended();
                });

  return row;
}

} // namespace hesiod::pp
