//! Reconstruct screen (doc/05 §4, design: headshot_ui_ref.html §03):
//! floating panels over the progressive point-cloud view — live filters
//! (confidence percentile, frame groups, frusta), the server log, and a
//! running point count. Panels mirror the same `Scene` state the keyboard
//! shortcuts mutate, so both stay usable.

use bevy::ecs::error::Result;
use bevy::feathers::constants::fonts;
use bevy::feathers::controls::{FeathersSlider, FeathersToggleSwitch};
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::palette;
use bevy::feathers::theme::ThemedText;
use bevy::picking::events::{Click, Pointer};
use bevy::prelude::*;
use bevy::text::FontSize;
use bevy::ui::Checked;
use bevy::ui_widgets::{
    SliderPrecision, SliderStep, SliderValue, ValueChange, checkbox_self_update,
};

use crate::{FrameGroup, Scene as CloudScene};

use super::log::StatusLog;
use super::theme::{self, UiFonts};
use super::{Screen, SessionSettings};

#[derive(Component, Default, Clone)]
struct ReconstructRoot;

#[derive(Component, Default, Clone)]
struct HeaderStatus;

#[derive(Component, Default, Clone)]
struct ConfValue;

#[derive(Component, Default, Clone)]
struct ConfSlider;

#[derive(Component, Default, Clone)]
struct GroupList;

#[derive(Component)]
struct GroupRow(FrameGroup);

#[derive(Component, Default, Clone)]
struct FrustaToggle;

#[derive(Component, Default, Clone)]
struct LogText;

#[derive(Component, Default, Clone)]
struct PointsText;

#[derive(Component, Default, Clone)]
struct ChunksText;

pub struct ReconstructScreenPlugin;

impl Plugin for ReconstructScreenPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Session), spawn_screen)
            .add_systems(OnExit(Screen::Session), despawn_screen)
            .add_systems(
                Update,
                (refresh_status, refresh_filters, refresh_log, refresh_stats)
                    .run_if(in_state(Screen::Session)),
            );
    }
}

fn spawn_screen(world: &mut World) -> Result {
    let server = world.resource::<SessionSettings>().server.clone();
    world.spawn_scene(screen(server))?;
    Ok(())
}

fn screen(server: String) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
        }
        ReconstructRoot
        Children [
            {theme::header_bar("/ reconstruct", Screen::Session, header_extra())},
            (
                // transparent canvas over the 3D view; panels are absolute
                Node {
                    position_type: PositionType::Relative,
                    flex_grow: 1.0,
                    min_height: px(0),
                }
                Children [
                    {filters_panel()},
                    {log_panel(server)},
                    {stats_panel()},
                    {help_bar()},
                ]
            ),
        ]
    }
}

fn header_extra() -> impl SceneList {
    bsn_list! {
        (
            Node {
                width: px(7),
                height: px(7),
                border_radius: BorderRadius::all(px(4)),
            }
            BackgroundColor(palette::ACCENT)
        ),
        (
            Node
            InheritableFont { font: fonts::MONO, font_size: FontSize::Px(12.0) }
            Children [ (Text("connecting…") ThemedText TextColor(palette::ACCENT) HeaderStatus) ]
        ),
    }
}

fn filters_panel() -> impl SceneList {
    bsn_list! {(
        Node {
            position_type: PositionType::Absolute,
            right: px(16),
            top: px(16),
            width: px(256),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(14),
            padding: px(14),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
        }
        BackgroundColor(theme::PANEL)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                }
                Children [
                    {theme::t_bold(12.0, palette::WHITE, "Filters")},
                    {theme::spacer()},
                    {theme::t_mono(10.0, palette::ACCENT, "live")},
                ]
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(8),
                }
                Children [
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Baseline,
                        }
                        Children [
                            {theme::t_sans(11.0, palette::LIGHT_GRAY_2, "Confidence percentile")},
                            {theme::spacer()},
                            (
                                Node
                                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                                Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_1) ConfValue) ]
                            ),
                        ]
                    ),
                    (
                        @FeathersSlider {
                            @max: 90.0,
                            @value: 30.0,
                        }
                        ConfSlider
                        SliderStep(10.)
                        // feathers' fill/text sync needs this
                        SliderPrecision(0)
                        on(|change: On<ValueChange<f32>>,
                           mut scene: ResMut<CloudScene>,
                           mut commands: Commands| {
                            scene.conf_quantile = (change.value / 100.0).clamp(0.0, 0.9);
                            scene.dirty = true;
                            commands.entity(change.source).insert(SliderValue(change.value));
                        })
                    ),
                ]
            ),
            (
                Node { height: px(1) }
                BackgroundColor(theme::BORDER_SUBTLE)
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(10),
                }
                Children [
                    {theme::t_sans(11.0, palette::LIGHT_GRAY_2, "Frame groups")},
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            row_gap: px(8),
                        }
                        GroupList
                    ),
                ]
            ),
            (
                Node { height: px(1) }
                BackgroundColor(theme::BORDER_SUBTLE)
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                }
                Children [
                    (
                        @FeathersToggleSwitch
                        FrustaToggle
                        Checked
                        on(checkbox_self_update)
                        on(|change: On<ValueChange<bool>>, mut scene: ResMut<CloudScene>| {
                            scene.show_frusta = change.value;
                        })
                    ),
                    {theme::t_sans(12.0, palette::LIGHT_GRAY_1, "Show camera frusta")},
                ]
            ),
        ]
    )}
}

fn log_panel(server: String) -> impl SceneList {
    bsn_list! {(
        Node {
            position_type: PositionType::Absolute,
            left: px(16),
            bottom: px(16),
            width: px(360),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(9),
            padding: UiRect::axes(px(14), px(12)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
        }
        BackgroundColor(theme::PANEL)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                }
                Children [
                    {theme::t_bold(11.0, palette::LIGHT_GRAY_1, "server log")},
                    {theme::spacer()},
                    {theme::t_mono(10.0, palette::LIGHT_GRAY_2, server)},
                ]
            ),
            (
                Node
                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) LogText) ]
            ),
        ]
    )}
}

fn stats_panel() -> impl SceneList {
    bsn_list! {(
        Node {
            position_type: PositionType::Absolute,
            right: px(16),
            bottom: px(16),
            width: px(256),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            padding: px(14),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
        }
        BackgroundColor(theme::PANEL)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Baseline,
                    column_gap: px(8),
                }
                Children [
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(26.0) }
                        Children [ (Text("0") ThemedText TextColor(palette::WHITE) PointsText) ]
                    ),
                    {theme::t_sans(12.0, palette::LIGHT_GRAY_2, "points")},
                ]
            ),
            (
                Node
                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(10.0) }
                Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) ChunksText) ]
            ),
        ]
    )}
}

fn help_bar() -> impl SceneList {
    bsn_list! {(
        // full-width strip that centers the pill (no translate in UI)
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            right: px(0),
            bottom: px(16),
            display: Display::Flex,
            justify_content: JustifyContent::Center,
        }
        Children [(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(14),
            padding: UiRect::axes(px(14), px(8)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
        }
        BackgroundColor(theme::PANEL)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            {theme::t_mono(11.0, palette::LIGHT_GRAY_2,
                "drag to orbit · scroll to zoom · [ ] confidence · G group · F frusta")},
            (
                Node { width: px(1), height: px(14) }
                BackgroundColor(theme::BORDER_SUBTLE)
            ),
            {theme::t_bold(11.0, palette::X_AXIS, "X")},
            {theme::t_bold(11.0, palette::Y_AXIS, "Y")},
            {theme::t_bold(11.0, palette::Z_AXIS, "Z")},
        ]
        )]
    )}
}

fn despawn_screen(mut commands: Commands, roots: Query<Entity, With<ReconstructRoot>>) {
    for e in &roots {
        commands.entity(e).despawn();
    }
}

// ---- observers ----------------------------------------------------------

fn group_clicked(
    click: On<Pointer<Click>>,
    rows: Query<&GroupRow>,
    mut scene: ResMut<CloudScene>,
) {
    if let Ok(row) = rows.get(click.entity)
        && scene.frame_group != row.0
    {
        scene.frame_group = row.0;
        scene.dirty = true;
    }
}

// ---- systems --------------------------------------------------------------

fn refresh_status(
    scene: Res<CloudScene>,
    mut status: Query<(&mut Text, &mut TextColor), With<HeaderStatus>>,
) {
    if !scene.is_changed() {
        return;
    }
    let Ok((mut t, mut c)) = status.single_mut() else { return };
    t.0 = scene.status.clone();
    c.0 = if scene.status.starts_with("FAILED") { theme::HAZARD } else { palette::ACCENT };
}

/// Mirror `Scene` into the filter widgets (keyboard shortcuts mutate the
/// same state, so the panel follows `[`/`]`, `G` and `F` too).
fn refresh_filters(
    scene: Res<CloudScene>,
    fonts: Option<Res<UiFonts>>,
    mut commands: Commands,
    mut conf_value: Query<&mut Text, With<ConfValue>>,
    conf_slider: Query<(Entity, &SliderValue), With<ConfSlider>>,
    group_list: Query<Entity, With<GroupList>>,
    frusta: Query<(Entity, Has<Checked>), With<FrustaToggle>>,
) {
    if !scene.is_changed() {
        return;
    }
    let Some(fonts) = fonts else { return };
    if let Ok(mut t) = conf_value.single_mut() {
        t.0 = format!("≥ {:.0} %", scene.conf_quantile * 100.0);
    }
    if let Ok((e, v)) = conf_slider.single() {
        let want = scene.conf_quantile * 100.0;
        if (v.0 - want).abs() > 0.5 {
            commands.entity(e).insert(SliderValue(want));
        }
    }
    if let Ok((e, checked)) = frusta.single() {
        match (scene.show_frusta, checked) {
            (true, false) => {
                commands.entity(e).insert(Checked);
            }
            (false, true) => {
                commands.entity(e).remove::<Checked>();
            }
            _ => {}
        }
    }

    let Ok(list) = group_list.single() else { return };
    let cameras = scene.cameras.len();
    commands.entity(list).despawn_related::<Children>();
    commands.entity(list).with_children(|parent| {
        for (group, label) in [
            (FrameGroup::All, "All frames"),
            (FrameGroup::Even, "Even frames"),
            (FrameGroup::Odd, "Odd frames"),
        ] {
            let selected = scene.frame_group == group;
            let count = (0..cameras).filter(|&i| group.keeps(i as u16)).count();
            let mut row = parent.spawn((
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                },
                GroupRow(group),
            ));
            row.observe(group_clicked).with_children(|row| {
                row.spawn(radio_dot(selected));
                row.spawn(theme::sans(
                    &fonts,
                    12.0,
                    if selected { palette::WHITE } else { palette::LIGHT_GRAY_1 },
                    label,
                ));
                row.spawn((Node { flex_grow: 1.0, ..default() },));
                row.spawn(theme::mono(&fonts, 11.0, palette::LIGHT_GRAY_2, count.to_string()));
            });
        }
    });
}

fn radio_dot(selected: bool) -> impl Bundle {
    (
        Node {
            width: Val::Px(14.0),
            height: Val::Px(14.0),
            border: UiRect::all(Val::Px(2.0)),
            border_radius: BorderRadius::all(Val::Px(7.0)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_shrink: 0.0,
            ..default()
        },
        BorderColor::all(if selected { palette::ACCENT } else { palette::WARM_GRAY_1 }),
        children![(
            Node {
                width: Val::Px(6.0),
                height: Val::Px(6.0),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(if selected { palette::ACCENT } else { Color::NONE }),
        )],
    )
}

fn refresh_log(log: Res<StatusLog>, mut text: Query<&mut Text, With<LogText>>) {
    if !log.is_changed() {
        return;
    }
    if let Ok(mut t) = text.single_mut() {
        t.0 = log.tail(8);
    }
}

#[allow(clippy::type_complexity)]
fn refresh_stats(
    scene: Res<CloudScene>,
    mut texts: ParamSet<(
        Query<&mut Text, With<PointsText>>,
        Query<&mut Text, With<ChunksText>>,
    )>,
) {
    if !scene.is_changed() {
        return;
    }
    let total: usize = scene.chunks.iter().map(|c| c.positions.len()).sum();
    if let Ok(mut t) = texts.p0().single_mut() {
        t.0 = fmt_points(total);
    }
    if let Ok(mut t) = texts.p1().single_mut() {
        t.0 = format!(
            "{} chunks · {} cameras · conf ≥ {:.2}",
            scene.chunks.len(),
            scene.cameras.len(),
            scene.conf_threshold,
        );
    }
}

fn fmt_points(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}
