/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General Public
   License. The full license is in the file LICENSE, distributed with this software. */
#pragma once
#include <QWidget>

namespace hesiod
{

// Apply the industrial "properties manager" look to the node settings panel
// (right-hand pane): a panel-scoped palette (read by the custom-painted Meta
// widgets: sliders, range bars, canvases) plus a stylesheet scoped to the
// panel subtree (buttons, sections, combos, fields, scrollbars). Nothing
// outside `panel` is affected.
void apply_properties_panel_style(QWidget *panel);

} // namespace hesiod
