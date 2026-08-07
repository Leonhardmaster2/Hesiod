/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */

/**
 * @file properties_tokens.hpp
 * @brief Design tokens for the Hesiod industrial properties panel.
 *
 * Single source of truth for every colour and metric the custom-painted
 * property widgets use. Values are measured from the reference implementation
 * in `.claude/skills/hesiod-ui/` - see reference-tokens.md there before
 * changing anything.
 *
 * The two rules these tokens exist to enforce:
 *   - the rail fill is ALWAYS the group accent, it never encodes state;
 *   - only TEXT encodes state (white modified / grey default / dim locked).
 */
#pragma once
#include <QColor>
#include <QString>

namespace hesiod::pp
{

// ---------------------------------------------------------------- surfaces
inline const QColor kPage{"#2b2b2b"};        ///< panel background
inline const QColor kBar{"#262626"};         ///< top / bottom chrome strips
inline const QColor kApplyBar{"#2e2e2e"};    ///< apply strip
inline const QColor kSectionHeader{"#333333"};
inline const QColor kSectionHeaderHover{"#383838"};
inline const QColor kSectionHeaderPress{"#303030"};
inline const QColor kRailWell{"#1c1c1c"};    ///< slider track groove
inline const QColor kField{"#1f1f1f"};       ///< value box, switch track (off)
inline const QColor kFieldHover{"#262626"};
inline const QColor kFieldEditing{"#161616"};
inline const QColor kPadSurface{"#242424"};
inline const QColor kPopup{"#262626"};

// ------------------------------------------------------- bevels, hairlines
inline const QColor kBevelTop{"#3d3d3d"};    ///< section header top edge
inline const QColor kBevelBottom{"#232323"}; ///< section header bottom edge
inline const QColor kHairline{"#1a1a1a"};
inline const QColor kRailBorder{"#161616"};
inline const QColor kFieldBorder{"#4a4a4a"};
inline const QColor kFieldBorderHover{"#5a5a5a"};
inline const QColor kButtonBorder{"#1f1f1f"};
inline const QColor kChipBorder{"#222222"};
inline const QColor kChipBorderHover{"#565656"};

// ------------------------------------------------------- raised gradients
inline const QColor kButtonTop{"#454545"};
inline const QColor kButtonBottom{"#383838"};
inline const QColor kButtonTopPress{"#333333"};
inline const QColor kButtonBottomPress{"#2a2a2a"};
inline const QColor kChipTop{"#3a3a3a"};
inline const QColor kChipBottom{"#313131"};
inline const QColor kChipTopHover{"#404040"};
inline const QColor kChipBottomHover{"#363636"};
inline const QColor kChipTopPress{"#303030"};
inline const QColor kChipBottomPress{"#282828"};
inline const QColor kChipTopActive{"#4a4a4a"};
inline const QColor kChipBottomActive{"#3a3a3a"};

// ------------------------------------------------------------------- ink
inline const QColor kInkPrimary{"#e0e0e0"};
inline const QColor kInkTitle{"#d0d0d0"};
inline const QColor kInkDefault{"#9a9a9a"};  ///< parameter at its default
inline const QColor kInkDim{"#8a8a8a"};
inline const QColor kInkLocked{"#606060"};   ///< parameter locked
inline const QColor kInkModified{"#ffffff"}; ///< parameter changed
inline const QColor kInkIcon{"#c9c9c9"};

// ----------------------------------------------------------------- metal
inline const QColor kThumbTop{"#d6d6d6"};
inline const QColor kThumbBottom{"#a8a8a8"};
inline const QColor kThumbBorder{"#1a1a1a"};
inline const QColor kThumbNotch{"#5f5f5f"};
inline const QColor kKnobOnTop{"#e8e8e8"};
inline const QColor kKnobOnBottom{"#b8b8b8"};
inline const QColor kKnobOffTop{"#8a8a8a"};
inline const QColor kKnobOffBottom{"#6a6a6a"};

// --------------------------------------------------------------- accents
inline const QColor kAccent{"#e08a2e"};      ///< chrome accent, selection
inline const QColor kAccentErosion{"#cfa143"};
inline const QColor kAccentDowncutting{"#3aa899"};
inline const QColor kAccentScale{"#7d9cc0"};
inline const QColor kAccentFlow{"#c06478"};
inline const QColor kAccentSelective{"#a08bb8"};
inline const QColor kAccentOther{"#9a9a9a"};

/// Group accents in presentation order; sections cycle through these so
/// consecutive groups stay visually distinct.
inline const QColor *group_accent(int index)
{
  static const QColor palette[] = {kAccentErosion,
                                   kAccentDowncutting,
                                   kAccentScale,
                                   kAccentFlow,
                                   kAccentSelective,
                                   kAccentOther};
  static const int    count = sizeof(palette) / sizeof(palette[0]);
  return &palette[((index % count) + count) % count];
}

// ---------------------------------------------------------- editor extras
inline const QColor kPadBorder{"#4a4a4a"};
inline const QColor kPadGrid{"#3a3a3a"};
inline const QColor kPadCrosshair{"#7d9cc0"};
inline const QColor kPadHandle{"#e0e0e0"};
inline const QColor kRangeTrack{"#3a3a3a"};
inline const QColor kRangeSpan{"#7d9cc0"};
inline const QColor kPathPoint{"#d03030"};

// -------------------------------------------------------------- opacities
inline constexpr double kFillOpacity = 0.9;        ///< rail fill, normal
inline constexpr double kFillOpacityLocked = 0.3;  ///< rail fill, locked
inline constexpr double kThumbOpacityLocked = 0.4;
inline constexpr double kRangeSpanIdle = 0.5;

// -------------------------------------------------------------- geometry
inline constexpr int kRowHeight = 36;
inline constexpr int kRowSpacing = 10;
inline constexpr int kRailHeight = 6;
inline constexpr int kRailRadius = 1;
inline constexpr int kThumbWidth = 10;
inline constexpr int kThumbHeight = 18;
inline constexpr int kThumbRadius = 2;
inline constexpr int kNotchWidth = 2;
inline constexpr int kNotchHeight = 8;
inline constexpr int kGap = 12;
inline constexpr int kFieldHeight = 24;
inline constexpr int kFieldRadius = 2;
inline constexpr int kSectionHeaderHeight = 38;
inline constexpr int kBodyPadX = 20;
inline constexpr int kBodyPadXNarrow = 12;
inline constexpr int kBodyPadY = 12;
inline constexpr int kStackedRowHeight = 56;
inline constexpr int kCheckRowHeight = 28;
inline constexpr int kCheckBoxRowHeight = 26;
inline constexpr int kSwitchWidth = 36;
inline constexpr int kSwitchHeight = 18;
inline constexpr int kKnobSize = 12;
inline constexpr int kBoxSize = 18;
inline constexpr int kChipHeight = 26;

// ------------------------------------------------------- gradient editor
/// Taller than the reference's 34: the bar is the thing being judged, and the
/// old height read as a divider rather than a preview.
inline constexpr int kGradientBarHeight = 48;
inline constexpr int kGradientGutter = 18;    ///< stop-marker strip below the bar
inline constexpr int kGradientStopW = 11;
inline constexpr int kGradientStopH = 14;
inline constexpr int kGradientCheckSize = 6;  ///< alpha checkerboard cell
/// Presets wrap into a grid rather than scrolling sideways, so the swatch is
/// sized to fit several per row in a narrow pane.
inline constexpr int kSwatchHeight = 22;
inline constexpr int kSwatchMinWidth = 56;
inline constexpr int kSwatchGap = 6;

/// Below this width a row switches to its compact metrics. NOTE: this is
/// compared against the ROW's own width, not the window's - inside a section
/// body a row is roughly 40px narrower than the panel.
inline constexpr int kNarrowThreshold = 430;

inline int label_width(int row_width)
{
  return qBound(90, static_cast<int>(row_width * 0.3), 168);
}

inline int field_width(int row_width)
{
  return row_width < kNarrowThreshold ? 64 : 74;
}

// ------------------------------------------------------------- animation
inline constexpr int kGlideMs = 260;      ///< every value change glides
inline constexpr int kHoverMs = 120;
inline constexpr int kCollapseMs = 220;
inline constexpr int kSwitchMs = 150;
inline constexpr int kDoubleClickMs = 450;
inline constexpr int kDragThresholdPx = 6; ///< below this, it stays a click

// ------------------------------------------------------------ typography
inline constexpr int kLabelPx = 12;
inline constexpr int kValuePx = 13;
inline constexpr int kTitlePx = 12;
inline constexpr int kIndexPx = 11;
inline constexpr int kChipPx = 11;

/// macOS ships Menlo; Windows and Linux do not, and an unknown family
/// silently degrades the numerics to a proportional face.
QString mono_family();

} // namespace hesiod::pp
