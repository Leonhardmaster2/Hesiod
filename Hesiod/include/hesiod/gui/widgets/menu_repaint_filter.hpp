/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#pragma once
#include <cmath>

#include <QEvent>
#include <QMenu>
#include <QObject>

namespace hesiod
{

class MenuRepaintFilter : public QObject
{
public:
  using QObject::QObject;

protected:
  bool eventFilter(QObject *object, QEvent *event) override
  {
    if (event->type() == QEvent::UpdateRequest)
      if (auto *menu = qobject_cast<QMenu *>(object))
      {
        const qreal dpr = menu->devicePixelRatioF();
        if (!qFuzzyCompare(dpr, std::round(dpr)))
          // Fractional scaling can leave lines between previously hovered
          // items. Expand the dirty region before Qt syncs the backing store.
          // Doing this during Paint would be too late and risk a repaint loop.
          menu->update();
      }

    return false;
  }
};

} // namespace hesiod
