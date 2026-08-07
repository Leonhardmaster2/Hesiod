/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file pp_section.hpp
 * @brief Collapsible parameter group: beveled header strip + animated body.
 *
 * Ported from Section.qml. The header is a custom-painted 38px strip with a
 * light top bevel and dark bottom bevel, a disclosure triangle that rotates
 * 90 degrees when open, a monospace index in the group accent, and a bold
 * uppercase title.
 *
 * The body is clipped while it animates, which is why a combo popup must
 * never be parented inside one - put those at panel level.
 */
#pragma once
#include <QColor>
#include <QVBoxLayout>
#include <QVariantAnimation>
#include <QWidget>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

class SectionHeader;

class PpSection : public QWidget
{
  Q_OBJECT

public:
  explicit PpSection(const QString &title,
                     const QString &index,
                     const QColor  &accent,
                     QWidget       *parent = nullptr);

  /// Layout that owns the rows. Add widgets here, not to the section itself.
  QVBoxLayout *body_layout() const { return this->body_layout_; }

  bool is_expanded() const { return this->expanded_; }
  void set_expanded(bool expanded, bool animate = true);

signals:
  void expanded_changed(bool expanded);

protected:
  void showEvent(QShowEvent *event) override;

private:
  int body_full_height() const;

  /// Real body height, captured while expanded and correctly laid out.
  int measured_full_ = -1;

  SectionHeader     *header_ = nullptr;
  QWidget           *body_ = nullptr;
  QVBoxLayout       *body_layout_ = nullptr;
  QVariantAnimation *collapse_ = nullptr;
  bool               expanded_ = true;
};

} // namespace hesiod::pp
