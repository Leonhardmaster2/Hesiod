/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <QFontDatabase>

#include "hesiod/gui/widgets/properties/properties_tokens.hpp"

namespace hesiod::pp
{

QString mono_family()
{
  // Resolved once. The reference implementation specifies Menlo, which only
  // exists on macOS - naming it directly elsewhere silently falls back to a
  // proportional face and every numeric readout loses its column alignment.
  static const QString family = []() -> QString
  {
    for (const QString &candidate :
         {QStringLiteral("Menlo"),
          QStringLiteral("Consolas"),
          QStringLiteral("DejaVu Sans Mono"),
          QStringLiteral("Liberation Mono")})
      if (QFontDatabase::families().contains(candidate))
        return candidate;

    return QFontDatabase::systemFont(QFontDatabase::FixedFont).family();
  }();

  return family;
}

} // namespace hesiod::pp
