/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <QDir>
#include <QFileInfo>
#include <QPalette>

#include "hesiod/gui/widgets/properties_panel_style.hpp"
#include "hesiod/logger.hpp"

namespace hesiod
{

namespace
{

// Industrial neutral-gray token set (all grays have R == G == B), orange
// chrome accent. The palette is what the custom-painted Meta widgets (sliders,
// range bars, canvases) read; the stylesheet below covers the standard Qt
// widgets of the panel.
QPalette make_panel_palette()
{
  QPalette pal;

  pal.setColor(QPalette::Window, QColor("#2b2b2b"));
  pal.setColor(QPalette::WindowText, QColor("#e0e0e0"));
  pal.setColor(QPalette::Base, QColor("#1c1c1c"));          // rail wells / tracks
  pal.setColor(QPalette::AlternateBase, QColor("#262626"));
  pal.setColor(QPalette::Text, QColor("#9a9a9a"));          // default param state
  pal.setColor(QPalette::PlaceholderText, QColor("#8a8a8a"));
  pal.setColor(QPalette::Button, QColor("#383838"));
  pal.setColor(QPalette::ButtonText, QColor("#e0e0e0"));
  pal.setColor(QPalette::BrightText, QColor("#ffffff"));    // modified param state
  pal.setColor(QPalette::Highlight, QColor("#e08a2e"));     // chrome accent (fills)
  pal.setColor(QPalette::HighlightedText, QColor("#ffffff"));
  pal.setColor(QPalette::Mid, QColor("#4a4a4a"));           // field borders
  pal.setColor(QPalette::Dark, QColor("#161616"));          // rail hairlines
  pal.setColor(QPalette::Light, QColor("#444444"));
  pal.setColor(QPalette::ToolTipBase, QColor("#262626"));
  pal.setColor(QPalette::ToolTipText, QColor("#e0e0e0"));
  pal.setColor(QPalette::Link, QColor("#7d9cc0"));          // range-bar spans

  for (const auto role : {QPalette::WindowText,
                          QPalette::Text,
                          QPalette::ButtonText,
                          QPalette::HighlightedText})
    pal.setColor(QPalette::Disabled, role, QColor("#606060"));

  return pal;
}

// Scoped to the panel subtree: every selector is prefixed so nothing leaks
// outside the properties manager. The panel root carries the left hairline
// that separates it from the graph editor.
const char *panel_style_sheet = R"CSS(
QWidget#propertiesPanel {
    background-color: #2b2b2b;
    border-left: 1px solid #1a1a1a;
}

QWidget#propertiesPanel QWidget {
    background-color: transparent;
    color: #e0e0e0;
    border: 0px;
}

/* ── header strip ─────────────────────────────────────────────── */
QWidget#propertiesPanel QWidget#ppHeader {
    background-color: #262626;
    border-bottom: 1px solid #1a1a1a;
}

QWidget#propertiesPanel QFrame#ppHeaderLogo {
    background-color: #e08a2e;
    border: 1px solid #1a1a1a;
}

QWidget#propertiesPanel QLabel#ppHeaderTitle {
    font-size: 13px;
    font-weight: bold;
    color: #e0e0e0;
}

QWidget#propertiesPanel QLabel#ppHeaderSub {
    font-family: Consolas, Menlo, monospace;
    font-size: 11px;
    color: #8a8a8a;
}

/* ── node pin header (NodeHeader title row spec) ──────────────── */
QWidget#propertiesPanel QCheckBox#ppPinHeader {
    font-size: 13px;
    color: #e0e0e0;
    padding: 4px 0px;
}

/* ── node toolbar buttons (NodeHeader chip spec) ──────────────── */
QWidget#propertiesPanel QToolButton#ppToolButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3a3a3a, stop: 1 #313131);
    border: 1px solid #222222;
    border-radius: 2px;
}

QWidget#propertiesPanel QToolButton#ppToolButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #404040, stop: 1 #363636);
    border-color: #565656;
}

QWidget#propertiesPanel QToolButton#ppToolButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #333333, stop: 1 #282828);
    border-color: #222222;
}

/* ── collapsible section headers ──────────────────────────────── */
QWidget#propertiesPanel QToolButton#ppSectionHeader {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3d3d3d, stop: 0.04 #333333,
                                stop: 0.96 #333333, stop: 1 #232323);
    border: 0px;
    border-radius: 0px;
    text-align: left;
    padding: 9px 12px 9px 16px;
    font-size: 12px;
    font-weight: bold;
    color: #d0d0d0;
}

QWidget#propertiesPanel QToolButton#ppSectionHeader:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #424242, stop: 0.04 #383838,
                                stop: 0.96 #383838, stop: 1 #262626);
}

QWidget#propertiesPanel QToolButton#ppSectionHeader:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #363636, stop: 0.04 #303030,
                                stop: 0.96 #303030, stop: 1 #232323);
}

/* ── toggle / action buttons (bool attributes, binary buttons) ── */
QWidget#propertiesPanel QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3a3a3a, stop: 1 #313131);
    border: 1px solid #222222;
    border-radius: 2px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: bold;
    color: #b0b0b0;
}

QWidget#propertiesPanel QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #404040, stop: 1 #363636);
    border-color: #565656;
    color: #e0e0e0;
}

QWidget#propertiesPanel QPushButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #303030, stop: 1 #282828);
}

QWidget#propertiesPanel QPushButton:checked {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4a4a4a, stop: 1 #3a3a3a);
    border-color: #e08a2e;
    color: #e08a2e;
}

/* ── check boxes (CheckBoxRow spec: 18px square, accent tick) ─── */
QWidget#propertiesPanel QCheckBox {
    spacing: 12px;
    padding: 2px;
    color: #c9c9c9;
}

QWidget#propertiesPanel QCheckBox:hover {
    color: #e0e0e0;
}

QWidget#propertiesPanel QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 2px;
    background-color: #1f1f1f;
    border: 1px solid #4a4a4a;
}

QWidget#propertiesPanel QCheckBox::indicator:hover {
    background-color: #262626;
    border-color: #5a5a5a;
}

QWidget#propertiesPanel QCheckBox::indicator:checked {
    background-color: #1f1f1f;
    border-color: #e08a2e;
    image: url(%PP_CHECK%);
}

QWidget#propertiesPanel QCheckBox::indicator:checked:hover {
    background-color: #262626;
}

/* ── combo boxes (HCombo spec: h30 field + raised chevron) ────── */
QWidget#propertiesPanel QComboBox {
    background-color: #1f1f1f;
    border: 1px solid #4a4a4a;
    border-radius: 2px;
    padding: 3px 30px 3px 10px;
    min-height: 22px;
    color: #d0d0d0;
}

QWidget#propertiesPanel QComboBox:hover {
    background-color: #262626;
    border-color: #5a5a5a;
}

QWidget#propertiesPanel QComboBox:on {
    border-color: #e08a2e;
}

QWidget#propertiesPanel QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    height: 24px;
    margin-right: 2px;
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #454545, stop: 1 #383838);
    border: 1px solid #1f1f1f;
    border-radius: 2px;
}

QWidget#propertiesPanel QComboBox::drop-down:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #333333, stop: 1 #2a2a2a);
}

QWidget#propertiesPanel QComboBox::down-arrow {
    image: url(%PP_EXPAND%);
}

QWidget#propertiesPanel QComboBox QAbstractItemView {
    background-color: #262626;
    border: 1px solid #1a1a1a;
    border-radius: 2px;
    padding: 4px;
    color: #c9c9c9;
    selection-background-color: #333333;
    selection-color: #e08a2e;
    outline: 0px;
}

/* ── text fields / spin boxes ─────────────────────────────────── */
QWidget#propertiesPanel QLineEdit,
QWidget#propertiesPanel QAbstractSpinBox {
    background-color: #1f1f1f;
    border: 1px solid #4a4a4a;
    border-radius: 2px;
    padding: 3px 6px;
    color: #e0e0e0;
    selection-background-color: #e08a2e;
    selection-color: #1f1f1f;
    font-family: Consolas, Menlo, monospace;
}

QWidget#propertiesPanel QLineEdit:focus,
QWidget#propertiesPanel QAbstractSpinBox:focus {
    border-color: #e08a2e;
}

QWidget#propertiesPanel QAbstractSpinBox::up-button,
QWidget#propertiesPanel QAbstractSpinBox::down-button {
    background-color: #333333;
    border: 0px;
    width: 14px;
}

QWidget#propertiesPanel QAbstractSpinBox::up-button:hover,
QWidget#propertiesPanel QAbstractSpinBox::down-button:hover {
    background-color: #383838;
}

/* ── scrollbars (thin overlay, reference) ─────────────────────── */
QWidget#propertiesPanel QScrollBar:vertical {
    background-color: #2b2b2b;
    width: 6px;
    margin: 0px;
}

QWidget#propertiesPanel QScrollBar::handle:vertical {
    background-color: #383838;
    border-radius: 2px;
    min-height: 30px;
}

QWidget#propertiesPanel QScrollBar::handle:vertical:hover {
    background-color: #535353;
}

QWidget#propertiesPanel QScrollBar::add-line:vertical,
QWidget#propertiesPanel QScrollBar::sub-line:vertical {
    height: 0px;
}

QWidget#propertiesPanel QScrollBar::add-page:vertical,
QWidget#propertiesPanel QScrollBar::sub-page:vertical {
    background: none;
}

QWidget#propertiesPanel QScrollBar:horizontal {
    background-color: #2b2b2b;
    height: 6px;
    margin: 0px;
}

QWidget#propertiesPanel QScrollBar::handle:horizontal {
    background-color: #383838;
    border-radius: 2px;
    min-width: 30px;
}

QWidget#propertiesPanel QScrollBar::handle:horizontal:hover {
    background-color: #535353;
}

QWidget#propertiesPanel QScrollBar::add-line:horizontal,
QWidget#propertiesPanel QScrollBar::sub-line:horizontal {
    width: 0px;
}

QWidget#propertiesPanel QScrollBar::add-page:horizontal,
QWidget#propertiesPanel QScrollBar::sub-page:horizontal {
    background: none;
}

/* ── menus (context menus raised from panel widgets) ──────────── */
QMenu {
    background-color: #262626;
    border: 1px solid #1a1a1a;
    padding: 4px;
    color: #e0e0e0;
}

QMenu::item {
    background: transparent;
    padding: 4px 18px;
}

QMenu::item:selected {
    background-color: #333333;
    color: #e08a2e;
}
)CSS";

} // namespace

namespace
{

/// Qt resolves relative QSS url() against the process working directory, not
/// the binary - launch from anywhere unexpected and the indicators silently
/// vanish. Resolve to an absolute path once, here, so the sheet stops caring.
QString icon_url(const char *file_name)
{
  const QFileInfo info(QStringLiteral("data/icons/") + QLatin1String(file_name));

  if (!info.exists())
  {
    Logger::log()->warn(
        "apply_properties_panel_style: stylesheet icon not found: {} (cwd {})",
        info.filePath().toStdString(),
        QDir::currentPath().toStdString());
    return {};
  }

  return info.absoluteFilePath();
}

} // namespace

void apply_properties_panel_style(QWidget *panel)
{
  if (!panel)
    return;

  panel->setObjectName("propertiesPanel");

  // The custom-painted Meta widgets (SliderFloat/SliderInt, RangeBar, the
  // canvases, IconCheckBox text) resolve their colours from the palette, so
  // install the panel palette before the attribute widgets get built.
  panel->setPalette(make_panel_palette());
  panel->setAutoFillBackground(true);

  QString sheet = QString::fromLatin1(panel_style_sheet);
  sheet.replace(QLatin1String("%PP_CHECK%"), icon_url("pp_check.svg"));
  sheet.replace(QLatin1String("%PP_EXPAND%"), icon_url("pp_expand_more.svg"));

  panel->setStyleSheet(sheet);
}

} // namespace hesiod
