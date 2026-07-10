//! headshot design tokens + shared chrome, matching the design reference
//! (`headshot_ui_ref.html`): feathers' dark palette for structure, a small
//! set of app accent colors, Fira Sans/Mono type. Feathers widgets keep
//! reading `UiTheme` tokens; app-specific chrome uses these constants
//! directly (one fixed dark theme — no token indirection needed).

use bevy::feathers::constants::fonts;
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::palette;
use bevy::feathers::theme::ThemedText;
use bevy::prelude::*;
use bevy::text::FontSize;

use super::Screen;

// ---- app palette (on top of bevy::feathers::palette) --------------------

/// Crop/model-frame accent (teal).
pub const CROP: Color = Color::srgb_u8(0x33, 0xd6, 0xc6);
/// Warnings that need a user decision (D-Log, destructive reselect).
pub const HAZARD: Color = Color::srgb_u8(0xe3, 0x9b, 0x3a);
/// `HAZARD` at panel-background strength.
pub const HAZARD_SOFT: Color = Color::srgba(0.89, 0.608, 0.227, 0.12);
/// GPS/telemetry badge blue.
pub const GPS: Color = Color::srgb_u8(0x5e, 0xa1, 0xe6);
/// Hairline between panes (darker than the feathers control border).
pub const BORDER_SUBTLE: Color = Color::srgb_u8(0x34, 0x34, 0x3a);
/// Chip background over imagery.
pub const SCRIM: Color = Color::srgba(0.0, 0.0, 0.0, 0.45);
/// Floating panel background over the 3D view.
pub const PANEL: Color = Color::srgba(0.122, 0.122, 0.141, 0.92);

// ---- fonts ---------------------------------------------------------------

/// Fira handles for dynamically spawned rows (static `bsn!` trees load the
/// same assets by path via [`InheritableFont`]; `commands.spawn` cannot).
#[derive(Resource)]
pub struct UiFonts {
    pub regular: Handle<Font>,
    pub bold: Handle<Font>,
    pub mono: Handle<Font>,
}

pub fn load_fonts(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(UiFonts {
        regular: assets.load(fonts::REGULAR),
        bold: assets.load(fonts::BOLD),
        mono: assets.load(fonts::MONO),
    });
}

// ---- text bundles for dynamic spawns -------------------------------------

fn text(font: Handle<Font>, size: f32, color: Color, s: String) -> impl Bundle {
    (
        Text::new(s),
        TextFont { font: font.into(), font_size: FontSize::Px(size), ..default() },
        TextColor(color),
    )
}

/// Fira Sans label.
pub fn sans(fonts: &UiFonts, size: f32, color: Color, s: impl Into<String>) -> impl Bundle {
    text(fonts.regular.clone(), size, color, s.into())
}

/// Fira Sans bold label.
pub fn sans_bold(fonts: &UiFonts, size: f32, color: Color, s: impl Into<String>) -> impl Bundle {
    text(fonts.bold.clone(), size, color, s.into())
}

/// Fira Mono value/readout text.
pub fn mono(fonts: &UiFonts, size: f32, color: Color, s: impl Into<String>) -> impl Bundle {
    text(fonts.mono.clone(), size, color, s.into())
}

// ---- text spans for static `bsn!` trees -----------------------------------
//
// `TextFont::font` is a `FontSource`, which scene templates can't fill from
// an asset path — so these wrap the span in feathers' `InheritableFont`,
// whose `Handle<Font>` field can.

fn t_span(font: &'static str, size: f32, color: Color, s: String) -> impl SceneList {
    bsn_list! {(
        Node
        InheritableFont { font: {font}, font_size: {FontSize::Px(size)} }
        Children [ (Text({s}) ThemedText TextColor({color})) ]
    )}
}

/// Fira Sans span for `bsn!` children lists.
pub fn t_sans(size: f32, color: Color, s: impl Into<String>) -> impl SceneList {
    t_span(fonts::REGULAR, size, color, s.into())
}

/// Fira Sans bold span for `bsn!` children lists.
pub fn t_bold(size: f32, color: Color, s: impl Into<String>) -> impl SceneList {
    t_span(fonts::BOLD, size, color, s.into())
}

/// Fira Mono span for `bsn!` children lists.
pub fn t_mono(size: f32, color: Color, s: impl Into<String>) -> impl SceneList {
    t_span(fonts::MONO, size, color, s.into())
}

// ---- small chrome bundles --------------------------------------------------

/// Tag chip (`D-LOG`, `GPS`, `REF`, `169 RAW`, …).
pub fn badge(fonts: &UiFonts, fg: Color, bg: Color, label: impl Into<String>) -> impl Bundle {
    (
        Node {
            padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(3.0)),
            align_items: AlignItems::Center,
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(bg),
        BorderColor::all(fg.with_alpha(0.4)),
        children![text(fonts.bold.clone(), 10.0, fg, label.into())],
    )
}

/// 15px checkbox visual for tree rows (row click toggles; not a widget).
pub fn check_square(on: bool) -> impl Bundle {
    (
        Node {
            width: Val::Px(15.0),
            height: Val::Px(15.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(2.0)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(if on { palette::ACCENT } else { palette::GRAY_2 }),
        BorderColor::all(if on { palette::ACCENT } else { BORDER_SUBTLE }),
        children![(
            Node {
                width: Val::Px(7.0),
                height: Val::Px(7.0),
                border_radius: BorderRadius::all(Val::Px(1.0)),
                ..default()
            },
            BackgroundColor(if on { palette::WHITE } else { Color::NONE }),
        )],
    )
}

/// Flexible gap.
pub fn spacer() -> impl SceneList {
    bsn_list! {( Node { flex_grow: 1.0 } )}
}

/// 1×22 vertical separator for the header bar.
fn vdiv() -> impl SceneList {
    bsn_list! {(
        Node { width: px(1), height: px(22), flex_shrink: 0.0 }
        BackgroundColor(BORDER_SUBTLE)
    )}
}

// ---- header bar -------------------------------------------------------------

fn crumb(label: &'static str, active: bool) -> impl SceneList {
    let (color, underline) = if active {
        (palette::WHITE, palette::ACCENT)
    } else {
        (palette::LIGHT_GRAY_2, Color::NONE)
    };
    bsn_list! {(
        Node {
            padding: UiRect::bottom(px(4)),
            border: UiRect::bottom(px(2)),
        }
        BorderColor::all(underline)
        InheritableFont { font: fonts::BOLD, font_size: FontSize::Px(11.0) }
        Children [ (Text({label.to_string()}) ThemedText TextColor({color})) ]
    )}
}

/// Setup › Review › Reconstruct trail with the active step underlined.
pub fn breadcrumb(active: Screen) -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(10),
        }
        Children [
            {crumb("Setup", active == Screen::Setup)},
            {t_sans(11.0, palette::LIGHT_GRAY_2, "›")},
            {crumb("Review", active == Screen::Review)},
            {t_sans(11.0, palette::LIGHT_GRAY_2, "›")},
            {crumb("Reconstruct", active == Screen::Session)},
        ]
    )}
}

/// The 52px top bar: `headshot / <sub>` + screen-specific middle content
/// + breadcrumb. `extra` is inserted between the divider and the trail.
pub fn header_bar(sub: &'static str, active: Screen, extra: impl SceneList) -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(14),
            height: px(52),
            padding: UiRect::horizontal(px(16)),
            border: UiRect::bottom(px(1)),
            flex_shrink: 0.0,
        }
        BackgroundColor(palette::GRAY_1)
        BorderColor::all(BORDER_SUBTLE)
        Children [
            {t_bold(15.0, palette::WHITE, "headshot")},
            {t_mono(12.0, palette::LIGHT_GRAY_2, sub)},
            {vdiv()},
            {extra},
            {spacer()},
            {breadcrumb(active)},
        ]
    )}
}
