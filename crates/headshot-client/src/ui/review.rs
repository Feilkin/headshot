//! Review screen (doc/05 §2, design: headshot_ui_ref.html §02): the
//! keyframe editor. Scrub any source's candidates, see the exact centered
//! crop the model will get (session aspect × per-frame zoom), include /
//! exclude frames, pick the frame shape, then reconstruct. All edits are
//! pure `SessionPlan` mutations; nothing here touches the media.

use bevy::asset::RenderAssetUsages;
use bevy::ecs::error::Result;
use bevy::feathers::constants::fonts;
use bevy::feathers::controls::{ButtonVariant, FeathersButton, FeathersSlider, FeathersToggleSwitch};
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::palette;
use bevy::feathers::theme::ThemedText;
use bevy::picking::events::{Click, Pointer};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::text::FontSize;
use bevy::ui::Checked;
use bevy::ui_widgets::{
    Activate, SliderPrecision, SliderValue, ValueChange, checkbox_self_update,
};
use headshot_capture::keyframe::RgbFrame;
use headshot_capture::plan::{AspectChoice, PlanUnit, SessionPlan};
use headshot_capture::preprocess::centered_crop;
use headshot_shared::sizing;

use super::theme::{self, UiFonts};
use super::{PlanRes, Screen, SessionSettings, WorkerTx, worker::Command, worker::RealizeRequest};

/// Photos group pseudo-tab.
const PHOTOS_TAB: usize = usize::MAX;

#[derive(Resource, Default)]
pub struct ReviewState {
    /// Active tab: video index or [`PHOTOS_TAB`].
    pub tab: usize,
    /// Focused candidate (video tab) / photo index (photos tab).
    pub focus: usize,
    /// What the preview texture currently shows.
    uploaded: Option<PlanUnit>,
    /// Aspect of the uploaded texture, for letterbox fitting.
    aspect: Option<f32>,
}

#[derive(Component, Default, Clone)]
struct ReviewRoot;

// header
#[derive(Component, Default, Clone)]
struct TabsRow;
#[derive(Component)]
struct TabButton(usize);
#[derive(Component, Default, Clone)]
struct MeterText;
#[derive(Component, Default, Clone)]
struct MeterFill;

// preview
#[derive(Component, Default, Clone)]
struct PreviewPane;
#[derive(Component, Default, Clone)]
struct PreviewBox;
#[derive(Component, Default, Clone)]
struct PreviewImage;
#[derive(Component, Default, Clone)]
struct CropOverlay;
#[derive(Component, Default, Clone)]
struct CropLabel;
#[derive(Component, Default, Clone)]
struct FileChip;
#[derive(Component, Default, Clone)]
struct SharpDot;
#[derive(Component, Default, Clone)]
struct SharpText;
#[derive(Component, Default, Clone)]
struct PipelineRow;

// transport
#[derive(Component, Default, Clone)]
struct TimeCur;
#[derive(Component, Default, Clone)]
struct TimeTotal;
#[derive(Component, Default, Clone)]
struct CandStats;
#[derive(Component, Default, Clone)]
struct MarkerStrip;
#[derive(Component, Default, Clone)]
struct Scrubber;
#[derive(Component, Default, Clone)]
struct StepPrev;
#[derive(Component, Default, Clone)]
struct StepNext;

// controls row
#[derive(Component, Default, Clone)]
struct IncludeToggle;
#[derive(Component, Default, Clone)]
struct ZoomValue;
#[derive(Component, Default, Clone)]
struct CropSlider;
#[derive(Component, Default, Clone)]
struct FrameInfo;

// sidebar
#[derive(Component, Default, Clone)]
struct AspectList;
#[derive(Component)]
struct AspectCard(AspectChoice);
#[derive(Component, Default, Clone)]
struct BudgetText;
#[derive(Component, Default, Clone)]
struct BudgetFill;
#[derive(Component, Default, Clone)]
struct BudgetHint;
#[derive(Component, Default, Clone)]
struct ReselectLabel;
#[derive(Component, Default, Clone)]
struct SummaryCount;
#[derive(Component, Default, Clone)]
struct SumShape;
#[derive(Component, Default, Clone)]
struct SumModel;
#[derive(Component, Default, Clone)]
struct SumEst;
#[derive(Component, Default, Clone)]
struct SumServer;

// strip
#[derive(Component, Default, Clone)]
struct StripCount;
#[derive(Component, Default, Clone)]
struct StripRow;
#[derive(Component)]
struct StripItem(PlanUnit);

pub struct ReviewScreenPlugin;

impl Plugin for ReviewScreenPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReviewState>()
            .add_systems(OnEnter(Screen::Review), (reset_state, spawn_screen).chain())
            .add_systems(OnExit(Screen::Review), despawn_screen)
            .add_systems(
                Update,
                (
                    normalize_tab,
                    (
                        rebuild_tabs,
                        rebuild_aspect_cards,
                        rebuild_strip,
                        sync_preview,
                        fit_preview,
                        sync_transport,
                        sync_controls,
                        refresh_budget,
                        refresh_summary,
                    ),
                )
                    .chain()
                    .run_if(in_state(Screen::Review)),
            );
    }
}

fn reset_state(mut state: ResMut<ReviewState>) {
    *state = ReviewState::default();
}

/// Keep `tab`/`focus` pointing at something that exists — the default tab
/// is video 0, which photos-only sessions don't have, and a re-scan can
/// shrink the source lists.
fn normalize_tab(plan: Res<PlanRes>, mut state: ResMut<ReviewState>) {
    let Some(p) = plan.0.as_deref() else { return };
    if state.tab != PHOTOS_TAB && state.tab >= p.videos.len() {
        let tab = if p.videos.is_empty() { PHOTOS_TAB } else { 0 };
        state.tab = tab;
        state.focus = 0;
    }
    if state.tab == PHOTOS_TAB && p.photos.is_empty() && !p.videos.is_empty() {
        state.tab = 0;
        state.focus = 0;
    }
    let n = tab_len(p, state.tab);
    if state.focus >= n && n > 0 {
        state.focus = n - 1;
    }
}

fn spawn_screen(world: &mut World) -> Result {
    world.spawn_scene(screen())?;
    Ok(())
}

fn screen() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
        }
        ReviewRoot
        BackgroundColor(palette::GRAY_0)
        Children [
            {theme::header_bar("/ review", Screen::Review, header_extra())},
            (
                // body: preview column + sidebar
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    flex_grow: 1.0,
                    min_height: px(0),
                }
                Children [
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            flex_grow: 1.0,
                            min_width: px(0),
                        }
                        Children [
                            {preview_pane()},
                            {transport_bar()},
                            {controls_row()},
                        ]
                    ),
                    {sidebar()},
                ]
            ),
            {strip_pane()},
        ]
    }
}

fn header_extra() -> impl SceneList {
    bsn_list! {
        (
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: px(5),
                flex_grow: 1.0,
            }
            TabsRow
        ),
        (
            // keyframe budget meter
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: px(10),
                padding: UiRect::axes(px(12), px(7)),
                border_radius: BorderRadius::all(px(4)),
                flex_shrink: 0.0,
            }
            BackgroundColor(palette::GRAY_2)
            Children [
                (
                    Node
                    InheritableFont { font: fonts::MONO, font_size: FontSize::Px(13.0) }
                    Children [ (Text("") ThemedText TextColor(palette::WHITE) MeterText) ]
                ),
                {theme::t_sans(11.0, palette::LIGHT_GRAY_2, "keyframes")},
                (
                    Node {
                        width: px(90),
                        height: px(5),
                        border_radius: BorderRadius::all(px(3)),
                        overflow: Overflow::clip(),
                    }
                    BackgroundColor(palette::BLACK)
                    Children [
                        (
                            Node { width: percent(0), height: percent(100) }
                            BackgroundColor(palette::ACCENT)
                            MeterFill
                        )
                    ]
                ),
            ]
        ),
    }
}

fn preview_pane() -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_grow: 1.0,
            min_height: px(0),
            overflow: Overflow::clip(),
        }
        PreviewPane
        BackgroundColor(palette::BLACK)
        Children [
            (
                // image + crop overlay; fit_preview letterboxes this to
                // the pane at the source aspect
                Node {
                    position_type: PositionType::Relative,
                }
                PreviewBox
                Children [
                    (
                        Node { width: percent(100), height: percent(100) }
                        ImageNode::default()
                        PreviewImage
                    ),
                    (
                        Node {
                            position_type: PositionType::Absolute,
                            border: UiRect::all(px(2)),
                        }
                        BorderColor::all(theme::CROP)
                        CropOverlay
                        Children [
                            {corner_tick(true, true)},
                            {corner_tick(false, true)},
                            {corner_tick(true, false)},
                            {corner_tick(false, false)},
                            (
                                // "model frame W × H" tag above the crop
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: px(-2),
                                    top: px(-26),
                                    padding: UiRect::axes(px(8), px(4)),
                                    border: UiRect::all(px(1)),
                                    border_radius: BorderRadius::all(px(3)),
                                }
                                BackgroundColor(palette::BLACK)
                                BorderColor::all(theme::CROP)
                                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                                Children [ (Text("") ThemedText TextColor(theme::CROP) CropLabel) ]
                            ),
                        ]
                    ),
                ]
            ),
            (
                // top-left: source file + frame position
                Node {
                    position_type: PositionType::Absolute,
                    left: px(14),
                    top: px(12),
                    padding: UiRect::axes(px(9), px(6)),
                    border_radius: BorderRadius::all(px(4)),
                }
                BackgroundColor(theme::SCRIM)
                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_1) FileChip) ]
            ),
            (
                // top-right: sharpness
                Node {
                    position_type: PositionType::Absolute,
                    right: px(14),
                    top: px(12),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(7),
                    padding: UiRect::axes(px(10), px(7)),
                    border_radius: BorderRadius::all(px(4)),
                }
                BackgroundColor(theme::SCRIM)
                Children [
                    (
                        Node {
                            width: px(7),
                            height: px(7),
                            border_radius: BorderRadius::all(px(4)),
                        }
                        BackgroundColor(palette::Y_AXIS)
                        SharpDot
                    ),
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                        Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_1) SharpText) ]
                    ),
                ]
            ),
            (
                // bottom-left: source → crop → model readout
                Node {
                    position_type: PositionType::Absolute,
                    left: px(14),
                    bottom: px(12),
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(4),
                    padding: UiRect::axes(px(10), px(8)),
                    border_radius: BorderRadius::all(px(4)),
                }
                BackgroundColor(theme::SCRIM)
                Children [
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: px(6),
                        }
                        PipelineRow
                    ),
                    {theme::t_mono(10.0, palette::LIGHT_GRAY_2,
                        "centered readout — not a crop tool · zoom trims edges, shape is session-wide")},
                ]
            ),
        ]
    )}
}

/// One L-shaped crop corner (3px strokes, 16px arms).
fn corner_tick(left: bool, top: bool) -> impl SceneList {
    let x = if left { UiRect::left(Val::Px(3.0)) } else { UiRect::right(Val::Px(3.0)) };
    let border = if top { x.with_top(Val::Px(3.0)) } else { x.with_bottom(Val::Px(3.0)) };
    let (l, r) = if left { (Val::Px(-2.0), Val::Auto) } else { (Val::Auto, Val::Px(-2.0)) };
    let (t, b) = if top { (Val::Px(-2.0), Val::Auto) } else { (Val::Auto, Val::Px(-2.0)) };
    bsn_list! {(
        Node {
            position_type: PositionType::Absolute,
            left: {l},
            right: {r},
            top: {t},
            bottom: {b},
            width: px(16),
            height: px(16),
            border: {border},
        }
        BorderColor::all(theme::CROP)
    )}
}

fn transport_bar() -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(9),
            padding: UiRect::axes(px(16), px(12)),
            border: UiRect::top(px(1)),
            flex_shrink: 0.0,
        }
        BackgroundColor(palette::GRAY_1)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                }
                Children [
                    {step_button("‹")},
                    {step_button_next("›")},
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(12.0) }
                        Children [ (Text("") ThemedText TextColor(palette::WHITE) TimeCur) ]
                    ),
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(12.0) }
                        Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) TimeTotal) ]
                    ),
                    {theme::spacer()},
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                        Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) CandStats) ]
                    ),
                ]
            ),
            (
                // selected-candidate markers + playhead (readout only)
                Node {
                    height: px(12),
                    position_type: PositionType::Relative,
                    border_radius: BorderRadius::all(px(3)),
                    overflow: Overflow::clip(),
                }
                BackgroundColor(palette::BLACK)
                MarkerStrip
            ),
            (
                @FeathersSlider {
                    @max: 1000.0,
                    @value: 0.0,
                }
                Scrubber
                // feathers' fill/text sync needs this
                SliderPrecision(0)
                on(scrub)
            ),
        ]
    )}
}

fn step_button(label: &'static str) -> impl SceneList {
    bsn_list! {(
        Node {
            width: px(28),
            height: px(26),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border_radius: BorderRadius::all(px(3)),
        }
        BackgroundColor(palette::GRAY_3)
        StepPrev
        on(step_focus)
        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(13.0) }
        Children [ (Text({label.to_string()}) ThemedText TextColor(palette::LIGHT_GRAY_1)) ]
    )}
}

fn step_button_next(label: &'static str) -> impl SceneList {
    bsn_list! {(
        Node {
            width: px(28),
            height: px(26),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border_radius: BorderRadius::all(px(3)),
        }
        BackgroundColor(palette::GRAY_3)
        StepNext
        on(step_focus)
        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(13.0) }
        Children [ (Text({label.to_string()}) ThemedText TextColor(palette::LIGHT_GRAY_1)) ]
    )}
}

fn controls_row() -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(18),
            padding: UiRect::axes(px(16), px(12)),
            border: UiRect::top(px(1)),
            flex_shrink: 0.0,
        }
        BackgroundColor(palette::GRAY_0)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(9),
                }
                Children [
                    (
                        @FeathersToggleSwitch
                        IncludeToggle
                        on(checkbox_self_update)
                        on(toggle_include)
                    ),
                    {theme::t_sans(12.0, palette::LIGHT_GRAY_1, "Include frame")},
                ]
            ),
            (
                Node { width: px(1), height: px(22) }
                BackgroundColor(theme::BORDER_SUBTLE)
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(12),
                    width: px(380),
                }
                Children [
                    {theme::t_sans(12.0, palette::LIGHT_GRAY_2, "Zoom")},
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(12.0) }
                        Children [ (Text("100 %") ThemedText TextColor(palette::LIGHT_GRAY_1) ZoomValue) ]
                    ),
                    (
                        Node { flex_grow: 1.0 }
                        Children [
                            (
                                @FeathersSlider {
                                    @min: 30.0,
                                    @max: 100.0,
                                    @value: 100.0,
                                }
                                Node { width: percent(100) }
                                CropSlider
                                // feathers' fill/text sync needs this
                                SliderPrecision(0)
                                on(change_crop_scale)
                            )
                        ]
                    ),
                    (
                        @FeathersButton
                        on(reset_crop)
                        Children [ (Text("reset") ThemedText) ]
                    ),
                ]
            ),
            {theme::spacer()},
            (
                Node
                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) FrameInfo) ]
            ),
        ]
    )}
}

fn sidebar() -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            width: px(340),
            flex_shrink: 0.0,
            padding: px(16),
            row_gap: px(16),
            border: UiRect::left(px(1)),
            overflow: Overflow::scroll_y(),
        }
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
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
                            {theme::t_bold(12.0, palette::LIGHT_GRAY_1, "Frame shape")},
                            {theme::spacer()},
                            {theme::t_mono(10.0, palette::LIGHT_GRAY_2, "session-wide · free to change")},
                        ]
                    ),
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            row_gap: px(6),
                        }
                        AspectList
                    ),
                    {theme::t_mono(10.0, palette::LIGHT_GRAY_2,
                        "The defining source stays uncropped; the other is centre-cropped to match.")},
                ]
            ),
            (
                // budget pane
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(9),
                    padding: px(14),
                    border_radius: BorderRadius::all(px(4)),
                }
                BackgroundColor(palette::GRAY_1)
                Children [
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Baseline,
                        }
                        Children [
                            {theme::t_bold(12.0, palette::LIGHT_GRAY_1, "Keyframe budget")},
                            {theme::spacer()},
                            (
                                Node
                                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(13.0) }
                                Children [ (Text("") ThemedText TextColor(palette::WHITE) BudgetText) ]
                            ),
                        ]
                    ),
                    (
                        Node {
                            height: px(8),
                            border_radius: BorderRadius::all(px(4)),
                            overflow: Overflow::clip(),
                        }
                        BackgroundColor(palette::BLACK)
                        Children [
                            (
                                Node { width: percent(0), height: percent(100) }
                                BackgroundColor(palette::ACCENT)
                                BudgetFill
                            )
                        ]
                    ),
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(10.0) }
                        Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) BudgetHint) ]
                    ),
                ]
            ),
            (
                // auto reselect (destructive) card
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(9),
                    padding: px(13),
                    border: UiRect::all(px(1)),
                    border_radius: BorderRadius::all(px(4)),
                }
                BackgroundColor(theme::HAZARD_SOFT)
                BorderColor::all(theme::HAZARD)
                Children [
                    {theme::t_bold(12.0, theme::HAZARD, "Auto reselect")},
                    {theme::t_mono(11.0, palette::LIGHT_GRAY_2,
                        "Re-runs the automatic picker and discards all manual include / zoom edits.")},
                    (
                        Node {
                            align_self: AlignSelf::FlexStart,
                            padding: UiRect::axes(px(12), px(7)),
                            border: UiRect::all(px(1)),
                            border_radius: BorderRadius::all(px(4)),
                        }
                        BorderColor::all(theme::HAZARD)
                        on(|_: On<Pointer<Click>>, mut plan: ResMut<PlanRes>| {
                            if let Some(p) = plan.0.as_deref_mut() {
                                p.reselect();
                            }
                        })
                        InheritableFont { font: fonts::BOLD, font_size: FontSize::Px(12.0) }
                        Children [ (Text("Reselect keyframes") ThemedText TextColor(theme::HAZARD) ReselectLabel) ]
                    ),
                ]
            ),
            {theme::spacer()},
            (
                // session summary
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: px(11),
                    padding: px(16),
                    border_radius: BorderRadius::all(px(4)),
                }
                BackgroundColor(palette::GRAY_1)
                Children [
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(20.0) }
                        Children [ (Text("") ThemedText TextColor(palette::WHITE) SummaryCount) ]
                    ),
                    (
                        Node { height: px(1) }
                        BackgroundColor(theme::BORDER_SUBTLE)
                    ),
                    {summary_row("frame shape", sum_shape())},
                    {summary_row("model frame", sum_model())},
                    {summary_row("est. time", sum_est())},
                    {summary_row("server", sum_server())},
                ]
            ),
            (
                @FeathersButton {
                    @variant: ButtonVariant::Primary,
                }
                on(reconstruct)
                Children [ (Text("Reconstruct") ThemedText) ]
            ),
        ]
    )}
}

fn summary_row(label: &'static str, value: impl SceneList) -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Baseline,
        }
        Children [
            {theme::t_mono(12.0, palette::LIGHT_GRAY_2, label)},
            {theme::spacer()},
            {value},
        ]
    )}
}

macro_rules! sum_value {
    ($name:ident, $marker:ident) => {
        fn $name() -> impl SceneList {
            bsn_list! {(
                Node
                InheritableFont { font: fonts::MONO, font_size: FontSize::Px(12.0) }
                Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_1) $marker) ]
            )}
        }
    };
}

sum_value!(sum_shape, SumShape);
sum_value!(sum_model, SumModel);
sum_value!(sum_est, SumEst);
sum_value!(sum_server, SumServer);

fn strip_pane() -> impl SceneList {
    bsn_list! {(
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            height: px(134),
            padding: UiRect::axes(px(14), px(10)),
            row_gap: px(8),
            border: UiRect::top(px(1)),
            flex_shrink: 0.0,
        }
        BackgroundColor(palette::GRAY_1)
        BorderColor::all(theme::BORDER_SUBTLE)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(12),
                }
                Children [
                    {theme::t_bold(12.0, palette::WHITE, "Selected keyframes")},
                    (
                        Node
                        InheritableFont { font: fonts::MONO, font_size: FontSize::Px(11.0) }
                        Children [ (Text("") ThemedText TextColor(palette::LIGHT_GRAY_2) StripCount) ]
                    ),
                    {theme::spacer()},
                    (
                        Node {
                            width: px(10),
                            height: px(10),
                            border: UiRect::all(px(2)),
                            border_radius: BorderRadius::all(px(2)),
                        }
                        BorderColor::all(theme::HAZARD)
                    ),
                    {theme::t_mono(10.0, palette::LIGHT_GRAY_2, "reference frame")},
                ]
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    column_gap: px(8),
                    flex_grow: 1.0,
                    min_height: px(0),
                    overflow: Overflow::scroll_x(),
                }
                StripRow
            ),
        ]
    )}
}

fn despawn_screen(mut commands: Commands, roots: Query<Entity, With<ReviewRoot>>) {
    for e in &roots {
        commands.entity(e).despawn();
    }
}

// ---- observers ----------------------------------------------------------

fn scrub(
    change: On<ValueChange<f32>>,
    mut state: ResMut<ReviewState>,
    plan: Res<PlanRes>,
    mut commands: Commands,
) {
    commands.entity(change.source).insert(SliderValue(change.value));
    let Some(p) = plan.0.as_deref() else { return };
    let n = tab_len(p, state.tab);
    if n > 0 {
        state.focus = ((change.value / 1000.0) * (n - 1) as f32).round() as usize;
    }
}

fn step_focus(
    click: On<Pointer<Click>>,
    prev: Query<(), With<StepPrev>>,
    next: Query<(), With<StepNext>>,
    mut state: ResMut<ReviewState>,
    plan: Res<PlanRes>,
) {
    let Some(p) = plan.0.as_deref() else { return };
    let n = tab_len(p, state.tab);
    if n == 0 {
        return;
    }
    if prev.contains(click.entity) {
        state.focus = state.focus.saturating_sub(1);
    } else if next.contains(click.entity) {
        state.focus = (state.focus + 1).min(n - 1);
    }
}

fn toggle_include(
    change: On<ValueChange<bool>>,
    mut plan: ResMut<PlanRes>,
    state: Res<ReviewState>,
) {
    let Some(p) = plan.0.as_deref_mut() else { return };
    if let Some(unit) = focused_unit(p, &state) {
        p.set_included(unit, change.value);
    }
}

fn change_crop_scale(
    change: On<ValueChange<f32>>,
    mut plan: ResMut<PlanRes>,
    state: Res<ReviewState>,
    mut commands: Commands,
) {
    commands.entity(change.source).insert(SliderValue(change.value));
    let Some(p) = plan.0.as_deref_mut() else { return };
    if let Some(unit) = focused_unit(p, &state) {
        p.set_crop_scale(unit, change.value / 100.0);
    }
}

fn reset_crop(_: On<Activate>, mut plan: ResMut<PlanRes>, state: Res<ReviewState>) {
    let Some(p) = plan.0.as_deref_mut() else { return };
    if let Some(unit) = focused_unit(p, &state) {
        p.set_crop_scale(unit, 1.0);
    }
}

fn choose_aspect(
    click: On<Pointer<Click>>,
    cards: Query<&AspectCard>,
    mut plan: ResMut<PlanRes>,
) {
    let Ok(card) = cards.get(click.entity) else { return };
    if let Some(p) = plan.0.as_deref_mut() {
        p.aspect = card.0;
    }
}

fn reconstruct(
    _: On<Activate>,
    mut plan: ResMut<PlanRes>,
    settings: Res<SessionSettings>,
    tx: Res<WorkerTx>,
    mut next: ResMut<NextState<Screen>>,
) {
    let Some(p) = plan.0.take() else { return };
    if let Err(e) = p.validate() {
        warn!("plan invalid: {e}");
        plan.0 = Some(p);
        return;
    }
    let _ = tx.0.send(Command::Realize(Box::new(RealizeRequest {
        plan: p,
        server: settings.server.clone(),
        edge_threshold: settings.edge_threshold,
        dump_keyframes: settings.dump_keyframes.clone(),
    })));
    next.set(Screen::Session);
}

fn strip_clicked(
    click: On<Pointer<Click>>,
    items: Query<&StripItem>,
    plan: Res<PlanRes>,
    mut state: ResMut<ReviewState>,
) {
    let Ok(item) = items.get(click.entity) else { return };
    let Some(p) = plan.0.as_deref() else { return };
    match item.0 {
        PlanUnit::Video { vi, ci } => {
            state.tab = vi;
            state.focus = ci;
        }
        PlanUnit::Photo { pi } => {
            state.tab = PHOTOS_TAB;
            state.focus = pi.min(p.photos.len().saturating_sub(1));
        }
    }
}

fn tab_clicked(click: On<Pointer<Click>>, tabs: Query<&TabButton>, mut state: ResMut<ReviewState>) {
    if let Ok(tab) = tabs.get(click.entity) {
        state.tab = tab.0;
        state.focus = 0;
    }
}

// ---- shared lookups --------------------------------------------------------

fn tab_len(p: &SessionPlan, tab: usize) -> usize {
    if tab == PHOTOS_TAB {
        p.photos.len()
    } else {
        p.videos.get(tab).map_or(0, |v| v.cands.len())
    }
}

fn focused_unit(p: &SessionPlan, state: &ReviewState) -> Option<PlanUnit> {
    if state.tab == PHOTOS_TAB {
        (state.focus < p.photos.len()).then_some(PlanUnit::Photo { pi: state.focus })
    } else {
        let v = p.videos.get(state.tab)?;
        (state.focus < v.cands.len())
            .then_some(PlanUnit::Video { vi: state.tab, ci: state.focus })
    }
}

fn thumb_for(p: &SessionPlan, unit: PlanUnit) -> &RgbFrame {
    match unit {
        PlanUnit::Video { vi, ci } => &p.videos[vi].thumbs[ci],
        PlanUnit::Photo { pi } => &p.photos[pi].thumb,
    }
}

fn source_dims(p: &SessionPlan, unit: PlanUnit) -> (u32, u32) {
    match unit {
        PlanUnit::Video { vi, .. } => (p.videos[vi].meta.width, p.videos[vi].meta.height),
        PlanUnit::Photo { pi } => p.photos[pi].dims,
    }
}

fn file_name(path: &std::path::Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn make_image(thumb: &RgbFrame) -> Image {
    let rgba: Vec<u8> =
        thumb.data.chunks_exact(3).flat_map(|px| [px[0], px[1], px[2], 255]).collect();
    Image::new(
        Extent3d { width: thumb.width, height: thumb.height, depth_or_array_layers: 1 },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// `secs` → `hh:mm:ss.ff`.
fn timecode(secs: f64) -> String {
    let h = (secs / 3600.0) as u64;
    let m = ((secs / 60.0) as u64) % 60;
    format!("{h:02}:{m:02}:{:05.2}", secs % 60.0)
}

/// `secs` → `mm:ss` for thumbnail chips.
fn mmss(secs: f64) -> String {
    format!("{:02}:{:02}", (secs / 60.0) as u64, (secs % 60.0) as u64)
}

/// Human aspect label: a well-known ratio when close, else raw dims.
fn ratio_label(w: u32, h: u32) -> String {
    const KNOWN: &[(u32, u32)] =
        &[(16, 9), (9, 16), (3, 2), (2, 3), (4, 3), (3, 4), (1, 1), (21, 9)];
    let r = w as f64 / h as f64;
    for &(a, b) in KNOWN {
        if (r - a as f64 / b as f64).abs() < 0.02 {
            return format!("{a}:{b}");
        }
    }
    format!("{w}×{h}")
}

/// Fraction of `(sw, sh)` lost when centre-cropped to the `(tw, th)` shape.
fn crop_loss(sw: u32, sh: u32, tw: u32, th: u32) -> f64 {
    let [_, _, cw, ch] = centered_crop(sw, sh, tw, th, 1.0);
    1.0 - (cw as f64 * ch as f64) / (sw as f64 * sh as f64)
}

// ---- systems -----------------------------------------------------------------

fn rebuild_tabs(
    plan: Res<PlanRes>,
    state: Res<ReviewState>,
    fonts: Option<Res<UiFonts>>,
    mut commands: Commands,
    row: Query<Entity, With<TabsRow>>,
) {
    if !plan.is_changed() && !state.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let Some(fonts) = fonts else { return };
    let Ok(row) = row.single() else { return };
    commands.entity(row).despawn_related::<Children>();
    let spawn_tab = |parent: &mut ChildSpawnerCommands,
                         idx: usize,
                         name: String,
                         sub: String| {
        let active = state.tab == idx;
        parent
            .spawn((
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(if active { palette::GRAY_3 } else { palette::GRAY_2 }),
                BorderColor::all(if active { palette::ACCENT } else { Color::NONE }),
                TabButton(idx),
                children![
                    theme::sans_bold(
                        &fonts,
                        12.0,
                        if active { palette::WHITE } else { palette::LIGHT_GRAY_1 },
                        name,
                    ),
                    theme::mono(
                        &fonts,
                        10.0,
                        if active { theme::GPS } else { palette::LIGHT_GRAY_2 },
                        sub,
                    ),
                ],
            ))
            .observe(tab_clicked);
    };
    commands.entity(row).with_children(|parent| {
        for (vi, v) in p.videos.iter().enumerate() {
            spawn_tab(parent, vi, file_name(&v.path), format!("candidates · {}", v.cands.len()));
        }
        if !p.photos.is_empty() {
            let dir = p.photos[0]
                .path
                .parent()
                .and_then(|d| d.file_name())
                .map(|n| format!("{}/", n.to_string_lossy()))
                .unwrap_or_else(|| "photos".into());
            spawn_tab(parent, PHOTOS_TAB, dir, format!("photos · {}", p.photos.len()));
        }
    });
}

/// Frame-shape option cards: Auto + one per source shape.
fn rebuild_aspect_cards(
    plan: Res<PlanRes>,
    fonts: Option<Res<UiFonts>>,
    mut commands: Commands,
    list: Query<Entity, With<AspectList>>,
) {
    if !plan.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let Some(fonts) = fonts else { return };
    let Ok(list) = list.single() else { return };
    commands.entity(list).despawn_related::<Children>();

    // representative dims per class, for the "other side cropped −N%" hint
    let video_dims = p.videos.first().map(|v| (v.meta.width, v.meta.height));
    let photo_dims = p.photos.iter().find(|ph| ph.kept).map(|ph| ph.dims);

    let mut options: Vec<(AspectChoice, String, (u32, u32))> = Vec::new();
    let (_, _, atw, ath) = {
        let (tw, th) = p.target_size();
        (0, 0, tw, th)
    };
    options.push((AspectChoice::Auto, "Auto — dominant source".into(), (atw, ath)));
    for (vi, v) in p.videos.iter().enumerate() {
        let (w, h) = (v.meta.width, v.meta.height);
        let (_, _, tw, th) = sizing::target_size(w, h);
        options.push((
            AspectChoice::Video(vi),
            format!("{} — {} uncropped", ratio_label(w, h), file_name(&v.path)),
            (tw, th),
        ));
    }
    if let Some(pi) = p.photos.iter().position(|ph| ph.kept) {
        let (w, h) = p.photos[pi].dims;
        let (_, _, tw, th) = sizing::target_size(w, h);
        options.push((
            AspectChoice::Photo(pi),
            format!("{} — photos uncropped", ratio_label(w, h)),
            (tw, th),
        ));
    }

    commands.entity(list).with_children(|parent| {
        for (choice, title, (tw, th)) in options {
            let selected = p.aspect == choice;
            let mut sub = format!("frames {tw}×{th}");
            let other_loss = match choice {
                AspectChoice::Video(_) => photo_dims.map(|(w, h)| ("photos", crop_loss(w, h, tw, th))),
                AspectChoice::Photo(_) => video_dims.map(|(w, h)| ("clips", crop_loss(w, h, tw, th))),
                AspectChoice::Auto => None,
            };
            if let Some((what, loss)) = other_loss
                && loss > 0.005
            {
                sub += &format!(" · {what} cropped −{:.0}%", loss * 100.0);
            }
            parent
                .spawn((
                    Node {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::FlexStart,
                        column_gap: Val::Px(10.0),
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(9.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(4.0)),
                        ..default()
                    },
                    BackgroundColor(if selected { palette::GRAY_3 } else { palette::GRAY_1 }),
                    BorderColor::all(if selected { palette::ACCENT } else { theme::BORDER_SUBTLE }),
                    AspectCard(choice),
                    children![
                        radio_dot(selected),
                        (
                            Node {
                                display: Display::Flex,
                                flex_direction: FlexDirection::Column,
                                row_gap: Val::Px(3.0),
                                min_width: Val::Px(0.0),
                                ..default()
                            },
                            children![
                                theme::sans_bold(
                                    &fonts,
                                    12.0,
                                    if selected { palette::WHITE } else { palette::LIGHT_GRAY_1 },
                                    title,
                                ),
                                theme::mono(&fonts, 10.0, palette::LIGHT_GRAY_2, sub),
                            ],
                        ),
                    ],
                ))
                .observe(choose_aspect);
        }
    });
}

fn radio_dot(selected: bool) -> impl Bundle {
    (
        Node {
            width: Val::Px(14.0),
            height: Val::Px(14.0),
            margin: UiRect::top(Val::Px(1.0)),
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

/// Upload the focused thumbnail, size the preview to the source aspect,
/// place the crop overlay, and refresh the overlay chips.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sync_preview(
    plan: Res<PlanRes>,
    mut state: ResMut<ReviewState>,
    fonts: Option<Res<UiFonts>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut preview: Query<&mut ImageNode, With<PreviewImage>>,
    mut overlay: Query<&mut Node, (With<CropOverlay>, Without<PreviewBox>)>,
    mut texts: ParamSet<(
        Query<&mut Text, With<CropLabel>>,
        Query<&mut Text, With<FileChip>>,
        Query<&mut Text, With<SharpText>>,
    )>,
    pipeline: Query<Entity, With<PipelineRow>>,
) {
    let Some(p) = plan.0.as_deref() else { return };
    let Some(unit) = focused_unit(p, &state) else { return };
    let Some(fonts) = fonts else { return };
    if !plan.is_changed() && !state.is_changed() && state.uploaded == Some(unit) {
        return;
    }

    // texture (fit_preview picks up the aspect)
    if state.uploaded != Some(unit)
        && let Ok(mut node) = preview.single_mut()
    {
        let thumb = thumb_for(p, unit);
        node.image = images.add(make_image(thumb));
        state.uploaded = Some(unit);
        state.aspect = Some(thumb.width as f32 / thumb.height as f32);
    }

    // crop overlay from the exact rect realize will use
    let sel = p.selection_index(unit);
    let crop_scale = sel.map_or(1.0, |i| p.selected[i].crop_scale);
    let (tw, th) = p.target_size();
    let (sw, sh) = source_dims(p, unit);
    let [cx, cy, cw, ch] = centered_crop(sw, sh, tw, th, crop_scale);
    if let Ok(mut o) = overlay.single_mut() {
        let pct = |v: u32, total: u32| Val::Percent(v as f32 / total as f32 * 100.0);
        o.left = pct(cx, sw);
        o.top = pct(cy, sh);
        o.width = pct(cw, sw);
        o.height = pct(ch, sh);
    }
    if let Ok(mut t) = texts.p0().single_mut() {
        t.0 = format!("model frame {tw} × {th}");
    }

    // info chips
    let sharpness = match unit {
        PlanUnit::Video { vi, ci } => p.videos[vi].cands[ci].sharpness,
        PlanUnit::Photo { pi } => p.photos[pi].sharpness,
    };
    if let Ok(mut t) = texts.p1().single_mut() {
        t.0 = match unit {
            PlanUnit::Video { vi, ci } => {
                let v = &p.videos[vi];
                let c = &v.cands[ci];
                let gps = c
                    .gps
                    .map(|g| format!("\ngps {:.5}, {:.5}", g.lat, g.lon))
                    .unwrap_or_default();
                format!(
                    "{}\nframe {} / {} · {}{gps}",
                    file_name(&v.path),
                    ci + 1,
                    v.cands.len(),
                    timecode(c.time_s),
                )
            }
            PlanUnit::Photo { pi } => {
                let ph = &p.photos[pi];
                format!(
                    "{}\nphoto {} / {}{}",
                    file_name(&ph.path),
                    pi + 1,
                    p.photos.len(),
                    if ph.kept { "" } else { " · burst-rejected" },
                )
            }
        };
        if p.reference == unit {
            t.0 += "\nREFERENCE FRAME";
        }
    }
    if let Ok(mut t) = texts.p2().single_mut() {
        t.0 = format!("sharpness {sharpness:.0}");
    }

    // source → crop → model readout
    if let Ok(row) = pipeline.single() {
        commands.entity(row).despawn_related::<Children>();
        commands.entity(row).with_children(|parent| {
            parent.spawn(theme::mono(&fonts, 11.0, palette::LIGHT_GRAY_2, format!("source {sw}×{sh}")));
            parent.spawn(theme::mono(&fonts, 11.0, palette::WARM_GRAY_1, "→"));
            parent.spawn(theme::mono(&fonts, 11.0, theme::CROP, format!("crop {cw}×{ch}")));
            if crop_scale < 0.995 {
                parent.spawn(theme::mono(
                    &fonts,
                    11.0,
                    theme::HAZARD,
                    format!("(zoom {:.0}%)", crop_scale * 100.0),
                ));
            }
            parent.spawn(theme::mono(&fonts, 11.0, palette::WARM_GRAY_1, "→"));
            parent.spawn(theme::mono(&fonts, 11.0, palette::LIGHT_GRAY_1, format!("model {tw}×{th}")));
        });
    }
}

/// Letterbox the preview box into the pane at the uploaded texture's
/// aspect (runs every frame — pane size follows window resizes).
fn fit_preview(
    state: Res<ReviewState>,
    pane: Query<&ComputedNode, With<PreviewPane>>,
    mut boxes: Query<&mut Node, With<PreviewBox>>,
) {
    let Some(aspect) = state.aspect else { return };
    let (Ok(pane), Ok(mut b)) = (pane.single(), boxes.single_mut()) else { return };
    let avail = pane.size() * pane.inverse_scale_factor();
    // margins keep the crop tag and info chips readable at full crop
    let (avail_w, avail_h) = (avail.x - 48.0, avail.y - 72.0);
    if avail_w <= 0.0 || avail_h <= 0.0 {
        return;
    }
    let w = avail_w.min(avail_h * aspect);
    let h = w / aspect;
    if b.width != Val::Px(w.round()) || b.height != Val::Px(h.round()) {
        b.width = Val::Px(w.round());
        b.height = Val::Px(h.round());
    }
}

/// Timecode, candidate stats, tick markers and the scrubber thumb.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sync_transport(
    plan: Res<PlanRes>,
    state: Res<ReviewState>,
    mut commands: Commands,
    mut texts: ParamSet<(
        Query<&mut Text, With<TimeCur>>,
        Query<&mut Text, With<TimeTotal>>,
        Query<&mut Text, With<CandStats>>,
    )>,
    strip: Query<Entity, With<MarkerStrip>>,
    scrubber: Query<(Entity, &SliderValue), With<Scrubber>>,
) {
    if !plan.is_changed() && !state.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let n = tab_len(p, state.tab);

    if state.tab == PHOTOS_TAB {
        if let Ok(mut t) = texts.p0().single_mut() {
            t.0 = format!("photo {}", (state.focus + 1).min(n));
        }
        if let Ok(mut t) = texts.p1().single_mut() {
            t.0 = format!("/ {n}");
        }
        if let Ok(mut t) = texts.p2().single_mut() {
            let selected =
                p.selected.iter().filter(|s| matches!(s.unit, PlanUnit::Photo { .. })).count();
            t.0 = format!("photos · {n} total · {selected} selected");
        }
    } else if let Some(v) = p.videos.get(state.tab) {
        let cur = v.cands.get(state.focus).map_or(0.0, |c| c.time_s);
        if let Ok(mut t) = texts.p0().single_mut() {
            t.0 = timecode(cur);
        }
        if let Ok(mut t) = texts.p1().single_mut() {
            t.0 = v.meta.duration_s.map(|d| format!("/ {}", timecode(d))).unwrap_or_default();
        }
        if let Ok(mut t) = texts.p2().single_mut() {
            let selected = p
                .selected
                .iter()
                .filter(|s| matches!(s.unit, PlanUnit::Video { vi, .. } if vi == state.tab))
                .count();
            let rate = v
                .meta
                .duration_s
                .filter(|d| *d > 0.0)
                .map(|d| format!("candidates · {:.1} / sec · ", v.cands.len() as f64 / d))
                .unwrap_or_else(|| "candidates · ".into());
            t.0 = format!("{rate}{} total · {selected} selected", v.cands.len());
        }
    }

    // tick markers: one per selected frame of this tab, plus the playhead
    if let Ok(strip) = strip.single() {
        commands.entity(strip).despawn_related::<Children>();
        if n > 1 {
            let x = |i: usize| Val::Percent(i as f32 / (n - 1) as f32 * 100.0);
            commands.entity(strip).with_children(|parent| {
                for s in &p.selected {
                    let i = match (state.tab, s.unit) {
                        (PHOTOS_TAB, PlanUnit::Photo { pi }) => pi,
                        (tab, PlanUnit::Video { vi, ci }) if vi == tab => ci,
                        _ => continue,
                    };
                    parent.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: x(i),
                            top: Val::Px(0.0),
                            bottom: Val::Px(0.0),
                            width: Val::Px(2.0),
                            ..default()
                        },
                        BackgroundColor(palette::ACCENT.with_alpha(0.55)),
                    ));
                }
                parent.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: x(state.focus.min(n - 1)),
                        top: Val::Px(0.0),
                        bottom: Val::Px(0.0),
                        width: Val::Px(2.0),
                        ..default()
                    },
                    BackgroundColor(theme::CROP),
                ));
            });
        }
    }

    // keep the scrubber thumb on the focused frame (tab switches, steps,
    // strip clicks) — the widget only updates itself while dragged
    if let Ok((e, v)) = scrubber.single() {
        let want = if n > 1 { state.focus as f32 / (n - 1) as f32 * 1000.0 } else { 0.0 };
        if (v.0 - want).abs() > 0.5 {
            commands.entity(e).insert(SliderValue(want));
        }
    }
}

/// Mirror the focused frame into the include toggle, zoom slider, and info.
#[allow(clippy::type_complexity)]
fn sync_controls(
    plan: Res<PlanRes>,
    state: Res<ReviewState>,
    mut commands: Commands,
    toggle: Query<(Entity, Has<Checked>), With<IncludeToggle>>,
    crop_slider: Query<(Entity, &SliderValue), With<CropSlider>>,
    mut texts: ParamSet<(
        Query<&mut Text, With<ZoomValue>>,
        Query<&mut Text, With<FrameInfo>>,
    )>,
) {
    if !plan.is_changed() && !state.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let Some(unit) = focused_unit(p, &state) else { return };
    let sel = p.selection_index(unit);
    let crop_scale = sel.map_or(1.0, |i| p.selected[i].crop_scale);

    if let Ok((e, checked)) = toggle.single() {
        match (sel.is_some(), checked) {
            (true, false) => {
                commands.entity(e).insert(Checked);
            }
            (false, true) => {
                commands.entity(e).remove::<Checked>();
            }
            _ => {}
        }
    }
    if let Ok((e, v)) = crop_slider.single() {
        let want = (crop_scale * 100.0).clamp(30.0, 100.0);
        if (v.0 - want).abs() > 0.5 {
            commands.entity(e).insert(SliderValue(want));
        }
    }
    if let Ok(mut t) = texts.p0().single_mut() {
        t.0 = format!("{:.0} %", crop_scale * 100.0);
    }
    if let Ok(mut t) = texts.p1().single_mut() {
        t.0 = match unit {
            PlanUnit::Video { vi, ci } => {
                let c = &p.videos[vi].cands[ci];
                format!("frame {} · {}", c.source_frame, timecode(c.time_s))
            }
            PlanUnit::Photo { pi } => file_name(&p.photos[pi].path),
        };
    }
}

/// Budget meter/pane, reselect label, and strip header counts.
#[allow(clippy::type_complexity)]
fn refresh_budget(
    plan: Res<PlanRes>,
    mut texts: ParamSet<(
        Query<&mut Text, With<MeterText>>,
        Query<&mut Text, With<BudgetText>>,
        Query<&mut Text, With<BudgetHint>>,
        Query<&mut Text, With<ReselectLabel>>,
        Query<&mut Text, With<SummaryCount>>,
        Query<&mut Text, With<StripCount>>,
    )>,
    mut fills: Query<&mut Node, Or<(With<MeterFill>, With<BudgetFill>)>>,
) {
    if !plan.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let n = p.selected.len();
    let frac = (n as f32 / p.budget.max(1) as f32).min(1.0);

    if let Ok(mut t) = texts.p0().single_mut() {
        t.0 = format!("{n} / {}", p.budget);
    }
    if let Ok(mut t) = texts.p1().single_mut() {
        t.0 = format!("{n} / {}", p.budget);
    }
    if let Ok(mut t) = texts.p2().single_mut() {
        t.0 = format!(
            "{} remaining · each frame added raises est. cost",
            p.budget.saturating_sub(n)
        );
    }
    if let Ok(mut t) = texts.p3().single_mut() {
        t.0 = format!("Reselect {n} keyframes");
    }
    if let Ok(mut t) = texts.p4().single_mut() {
        t.0 = format!("{n} keyframes");
    }
    if let Ok(mut t) = texts.p5().single_mut() {
        t.0 = format!("{n} · in reconstruction order");
    }
    for mut f in &mut fills {
        f.width = Val::Percent(frac * 100.0);
    }
}

/// The session summary card values.
#[allow(clippy::type_complexity)]
fn refresh_summary(
    plan: Res<PlanRes>,
    settings: Res<SessionSettings>,
    mut sums: ParamSet<(
        Query<&mut Text, With<SumShape>>,
        Query<&mut Text, With<SumModel>>,
        Query<&mut Text, With<SumEst>>,
        Query<&mut Text, With<SumServer>>,
    )>,
) {
    if !plan.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let n = p.selected.len();
    let (tw, th) = p.target_size();

    if let Ok(mut t) = sums.p0().single_mut() {
        t.0 = match p.aspect {
            AspectChoice::Auto => format!("auto · {}", ratio_label(tw, th)),
            AspectChoice::Video(vi) => {
                format!("{} · {}", ratio_label(tw, th), file_name(&p.videos[vi].path))
            }
            AspectChoice::Photo(_) => format!("{} · photos", ratio_label(tw, th)),
        };
    }
    if let Ok(mut t) = sums.p1().single_mut() {
        t.0 = format!("{tw} × {th}");
    }
    if let Ok(mut t) = sums.p2().single_mut() {
        // s2 dominates and is quadratic: calibrated 227 s at 120 frames on
        // the Strix Halo box, plus roughly linear DINO+depth overhead
        let n_f = n as f64;
        let est = 227.0 * (n_f / 120.0).powi(2) + 100.0 * (n_f / 120.0);
        t.0 = format!("~{:.0} min", est / 60.0);
    }
    if let Ok(mut t) = sums.p3().single_mut() {
        t.0 = settings.server.clone();
    }
}

/// Rebuild the selected-frames strip whenever the plan or focus changes.
fn rebuild_strip(
    plan: Res<PlanRes>,
    state: Res<ReviewState>,
    fonts: Option<Res<UiFonts>>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    row: Query<Entity, With<StripRow>>,
) {
    if !plan.is_changed() && !state.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let Some(fonts) = fonts else { return };
    let Ok(row) = row.single() else { return };
    let focused = focused_unit(p, &state);
    commands.entity(row).despawn_related::<Children>();
    commands.entity(row).with_children(|parent| {
        for (order, s) in p.selected.iter().enumerate() {
            let thumb = thumb_for(p, s.unit);
            let handle = images.add(make_image(thumb));
            let is_ref = p.reference == s.unit;
            let is_focused = focused == Some(s.unit);
            let time_chip = match s.unit {
                PlanUnit::Video { vi, ci } => mmss(p.videos[vi].cands[ci].time_s),
                PlanUnit::Photo { pi } => file_name(&p.photos[pi].path),
            };
            let mut card = parent.spawn((
                Node {
                    width: Val::Px(148.0),
                    height: Val::Percent(100.0),
                    position_type: PositionType::Relative,
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    overflow: Overflow::clip(),
                    flex_shrink: 0.0,
                    ..default()
                },
                BorderColor::all(if is_ref {
                    theme::HAZARD
                } else if is_focused {
                    palette::ACCENT
                } else {
                    Color::NONE
                }),
                BackgroundColor(palette::GRAY_2),
                StripItem(s.unit),
            ));
            card.observe(strip_clicked).with_children(|card| {
                card.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: Val::Px(0.0),
                        right: Val::Px(0.0),
                        top: Val::Px(0.0),
                        bottom: Val::Px(0.0),
                        ..default()
                    },
                    ImageNode::new(handle),
                ));
                card.spawn((
                    chip_node(ChipCorner::TopLeft),
                    children![theme::mono(
                        &fonts,
                        10.0,
                        palette::LIGHT_GRAY_1,
                        format!("{}", order + 1),
                    )],
                ));
                card.spawn((
                    chip_node(ChipCorner::BottomLeft),
                    children![theme::mono(&fonts, 9.0, palette::LIGHT_GRAY_2, time_chip)],
                ));
                if is_ref {
                    card.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            right: Val::Px(5.0),
                            top: Val::Px(5.0),
                            padding: UiRect::axes(Val::Px(5.0), Val::Px(2.0)),
                            border_radius: BorderRadius::all(Val::Px(2.0)),
                            ..default()
                        },
                        BackgroundColor(theme::HAZARD),
                        children![theme::sans_bold(
                            &fonts,
                            9.0,
                            Color::srgb_u8(0x1a, 0x12, 0x07),
                            "REF",
                        )],
                    ));
                }
            });
        }
    });
}

enum ChipCorner {
    TopLeft,
    BottomLeft,
}

fn chip_node(corner: ChipCorner) -> impl Bundle {
    let mut node = Node {
        position_type: PositionType::Absolute,
        left: Val::Px(5.0),
        padding: UiRect::axes(Val::Px(5.0), Val::Px(2.0)),
        border_radius: BorderRadius::all(Val::Px(2.0)),
        ..default()
    };
    match corner {
        ChipCorner::TopLeft => node.top = Val::Px(5.0),
        ChipCorner::BottomLeft => node.bottom = Val::Px(5.0),
    }
    (node, BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)))
}
