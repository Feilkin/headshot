//! Review screen (doc/05 §2): the keyframe editor. Scrub any source's
//! candidates, see the exact centered crop the model will get (session
//! aspect × per-frame zoom), include/exclude frames, pick the aspect
//! bucket, then reconstruct. All edits are pure `SessionPlan` mutations;
//! nothing here touches the media.

use bevy::asset::RenderAssetUsages;
use bevy::feathers::controls::{FeathersButton, FeathersSlider, FeathersToggleSwitch};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::picking::events::{Click, Pointer};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::ui::Checked;
use bevy::ui_widgets::{Activate, SliderValue, ValueChange, checkbox_self_update};
use headshot_capture::keyframe::RgbFrame;
use headshot_capture::plan::{AspectChoice, PlanUnit, SessionPlan};
use headshot_capture::preprocess::centered_crop;

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
}

#[derive(Component, Default, Clone)]
struct ReviewRoot;

#[derive(Component, Default, Clone)]
struct TabsRow;

#[derive(Component)]
struct TabButton(usize);

#[derive(Component, Default, Clone)]
struct PreviewBox;

#[derive(Component, Default, Clone)]
struct PreviewImage;

#[derive(Component, Default, Clone)]
struct CropOverlay;

#[derive(Component, Default, Clone)]
struct InfoText;

#[derive(Component, Default, Clone)]
struct IncludeToggle;

#[derive(Component, Default, Clone)]
struct CropSlider;

#[derive(Component, Default, Clone)]
struct Scrubber;

#[derive(Component, Default, Clone)]
struct StripRow;

#[derive(Component)]
struct StripItem(PlanUnit);

#[derive(Component, Default, Clone)]
struct AspectLabel;

#[derive(Component, Default, Clone)]
struct SummaryText;

pub struct ReviewScreenPlugin;

impl Plugin for ReviewScreenPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReviewState>()
            .add_systems(OnEnter(Screen::Review), (reset_state, spawn_screen.spawn()).chain())
            .add_systems(OnExit(Screen::Review), despawn_screen)
            .add_systems(
                Update,
                (rebuild_tabs, rebuild_strip, sync_preview, refresh_summary)
                    .run_if(in_state(Screen::Review)),
            );
    }
}

fn reset_state(mut state: ResMut<ReviewState>) {
    *state = ReviewState::default();
}

fn spawn_screen() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
            row_gap: px(8),
            padding: px(12),
        }
        ReviewRoot
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    column_gap: px(6),
                }
                TabsRow
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    column_gap: px(12),
                    flex_grow: 1.0,
                }
                Children [
                    (
                        Node {
                            width: px(560),
                            height: px(315),
                            position_type: PositionType::Relative,
                        }
                        PreviewBox
                        Children [
                            (
                                Node {
                                    width: percent(100),
                                    height: percent(100),
                                }
                                ImageNode::default()
                                PreviewImage
                            ),
                            (
                                Node {
                                    position_type: PositionType::Absolute,
                                    border: UiRect::all(px(2)),
                                }
                                BorderColor::all(Color::srgb(1.0, 0.8, 0.2))
                                CropOverlay
                            ),
                        ]
                    ),
                    (
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            row_gap: px(8),
                            flex_grow: 1.0,
                        }
                        Children [
                            (Text("…") ThemedText InfoText),
                            (
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Row,
                                    align_items: AlignItems::Center,
                                    column_gap: px(8),
                                }
                                Children [
                                    (Text("include") ThemedText),
                                    (
                                        @FeathersToggleSwitch
                                        IncludeToggle
                                        on(checkbox_self_update)
                                        on(toggle_include)
                                    ),
                                ]
                            ),
                            (
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Row,
                                    align_items: AlignItems::Center,
                                    column_gap: px(8),
                                }
                                Children [
                                    (Text("crop") ThemedText),
                                    (
                                        @FeathersSlider {
                                            @max: 100.0,
                                            @value: 100.0,
                                        }
                                        Node { min_width: px(180) }
                                        CropSlider
                                        on(change_crop_scale)
                                    ),
                                ]
                            ),
                            (
                                @FeathersButton
                                on(cycle_aspect)
                                Children [ (Text("aspect: auto") ThemedText AspectLabel) ]
                            ),
                            (
                                @FeathersButton
                                on(|_: On<Activate>, mut plan: ResMut<PlanRes>| {
                                    if let Some(p) = plan.0.as_deref_mut() {
                                        p.reselect();
                                    }
                                })
                                Children [ (Text("auto reselect") ThemedText) ]
                            ),
                        ]
                    ),
                ]
            ),
            (
                @FeathersSlider {
                    @max: 1000.0,
                    @value: 0.0,
                }
                Scrubber
                on(scrub)
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    column_gap: px(4),
                    min_height: px(64),
                    overflow: Overflow::scroll_x(),
                }
                StripRow
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(12),
                }
                Children [
                    (Text("…") ThemedText SummaryText),
                    (
                        @FeathersButton
                        on(reconstruct)
                        Children [ (Text("Reconstruct") ThemedText) ]
                    ),
                ]
            ),
        ]
    }
}

fn despawn_screen(mut commands: Commands, roots: Query<Entity, With<ReviewRoot>>) {
    for e in &roots {
        commands.entity(e).despawn();
    }
}

// ---- observers ----------------------------------------------------------

fn scrub(change: On<ValueChange<f32>>, mut state: ResMut<ReviewState>, plan: Res<PlanRes>, mut commands: Commands) {
    commands.entity(change.source).insert(SliderValue(change.value));
    let Some(p) = plan.0.as_deref() else { return };
    let n = tab_len(p, state.tab);
    if n > 0 {
        state.focus = ((change.value / 1000.0) * (n - 1) as f32).round() as usize;
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
        p.set_crop_scale(unit, 0.3 + (change.value / 100.0) * 0.7);
    }
}

fn cycle_aspect(_: On<Activate>, mut plan: ResMut<PlanRes>) {
    let Some(p) = plan.0.as_deref_mut() else { return };
    p.aspect = match p.aspect {
        AspectChoice::Auto if !p.videos.is_empty() => AspectChoice::Video(0),
        AspectChoice::Auto => first_photo_aspect(p),
        AspectChoice::Video(vi) if vi + 1 < p.videos.len() => AspectChoice::Video(vi + 1),
        AspectChoice::Video(_) => first_photo_aspect(p),
        AspectChoice::Photo(_) => AspectChoice::Auto,
    };
}

fn first_photo_aspect(p: &SessionPlan) -> AspectChoice {
    p.photos
        .iter()
        .position(|ph| ph.kept)
        .map(AspectChoice::Photo)
        .unwrap_or(AspectChoice::Auto)
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

fn tab_clicked(
    click: On<Pointer<Click>>,
    tabs: Query<&TabButton>,
    mut state: ResMut<ReviewState>,
) {
    if let Ok(tab) = tabs.get(click.entity) {
        state.tab = tab.0;
        state.focus = 0;
    }
}

// ---- systems -------------------------------------------------------------

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

fn rebuild_tabs(
    plan: Res<PlanRes>,
    mut commands: Commands,
    row: Query<Entity, With<TabsRow>>,
) {
    if !plan.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let Ok(row) = row.single() else { return };
    commands.entity(row).despawn_related::<Children>();
    commands.entity(row).with_children(|parent| {
        for (vi, v) in p.videos.iter().enumerate() {
            let name = v
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("video {vi}"));
            parent
                .spawn((
                    Node {
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(Color::srgb(0.4, 0.4, 0.5)),
                    TabButton(vi),
                    children![(Text::new(name), ThemedText)],
                ))
                .observe(tab_clicked);
        }
        if !p.photos.is_empty() {
            parent
                .spawn((
                    Node {
                        padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        ..default()
                    },
                    BorderColor::all(Color::srgb(0.4, 0.4, 0.5)),
                    TabButton(PHOTOS_TAB),
                    children![(Text::new(format!("photos ({})", p.photos.len())), ThemedText)],
                ))
                .observe(tab_clicked);
        }
    });
}

/// Upload the focused thumbnail, place the crop overlay, refresh the info
/// text, and mirror the include/crop widgets to the focused frame.
#[allow(clippy::too_many_arguments)]
fn sync_preview(
    plan: Res<PlanRes>,
    mut state: ResMut<ReviewState>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    mut preview: Query<(&mut ImageNode, &ChildOf), With<PreviewImage>>,
    mut boxes: Query<&mut Node, With<PreviewBox>>,
    mut overlay: Query<&mut Node, (With<CropOverlay>, Without<PreviewBox>)>,
    mut info: Query<&mut Text, With<InfoText>>,
    toggle: Query<(Entity, Has<Checked>), With<IncludeToggle>>,
    crop_slider: Query<(Entity, &SliderValue), With<CropSlider>>,
) {
    let Some(p) = plan.0.as_deref() else { return };
    let Some(unit) = focused_unit(p, &state) else { return };
    let dirty = plan.is_changed() || state.is_changed() || state.uploaded != Some(unit);
    if !dirty {
        return;
    }

    // texture
    if state.uploaded != Some(unit)
        && let Ok((mut node, parent)) = preview.single_mut()
    {
        let thumb = thumb_for(p, unit);
        node.image = images.add(make_image(thumb));
        state.uploaded = Some(unit);
        // keep the preview box at the source aspect
        if let Ok(mut b) = boxes.get_mut(parent.parent()) {
            let (sw, sh) = (thumb.width as f32, thumb.height as f32);
            b.height = Val::Px(560.0 * sh / sw);
        }
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

    // info line
    if let Ok(mut t) = info.single_mut() {
        t.0 = match unit {
            PlanUnit::Video { vi, ci } => {
                let c = &p.videos[vi].cands[ci];
                format!(
                    "frame {} @ {:.1}s · sharpness {:.0}{}{}",
                    c.source_frame,
                    c.time_s,
                    c.sharpness,
                    c.gps
                        .map(|g| format!(" · gps {:.5},{:.5}", g.lat, g.lon))
                        .unwrap_or_default(),
                    if p.reference == unit { " · REFERENCE" } else { "" },
                )
            }
            PlanUnit::Photo { pi } => {
                let ph = &p.photos[pi];
                format!(
                    "{} · sharpness {:.0}{}{}",
                    ph.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                    ph.sharpness,
                    if ph.kept { "" } else { " · burst-rejected" },
                    if p.reference == unit { " · REFERENCE" } else { "" },
                )
            }
        };
    }

    // include toggle mirrors selection state
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
    // crop slider mirrors the focused frame's scale (immutable component:
    // changed by re-insert, matching the widget's own update path)
    if let Ok((e, v)) = crop_slider.single() {
        let want = ((crop_scale - 0.3) / 0.7 * 100.0).clamp(0.0, 100.0);
        if (v.0 - want).abs() > 0.5 {
            commands.entity(e).insert(SliderValue(want));
        }
    }
}

/// Rebuild the selected-frames strip whenever the plan changes.
fn rebuild_strip(
    plan: Res<PlanRes>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    row: Query<Entity, With<StripRow>>,
) {
    if !plan.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    let Ok(row) = row.single() else { return };
    commands.entity(row).despawn_related::<Children>();
    commands.entity(row).with_children(|parent| {
        for s in &p.selected {
            let thumb = thumb_for(p, s.unit);
            let handle = images.add(make_image(thumb));
            let is_ref = p.reference == s.unit;
            parent
                .spawn((
                    Node {
                        width: Val::Px(96.0),
                        height: Val::Px(54.0),
                        border: UiRect::all(Val::Px(2.0)),
                        flex_shrink: 0.0,
                        ..default()
                    },
                    BorderColor::all(if is_ref {
                        Color::srgb(1.0, 0.3, 0.2)
                    } else {
                        Color::srgb(0.25, 0.25, 0.3)
                    }),
                    StripItem(s.unit),
                    children![(
                        Node { width: Val::Percent(100.0), height: Val::Percent(100.0), ..default() },
                        ImageNode::new(handle),
                    )],
                ))
                .observe(strip_clicked);
        }
    });
}

fn refresh_summary(
    plan: Res<PlanRes>,
    settings: Res<SessionSettings>,
    state: Res<ReviewState>,
    mut text: Query<&mut Text, With<SummaryText>>,
    mut aspect_label: Query<&mut Text, (With<AspectLabel>, Without<SummaryText>)>,
) {
    if !plan.is_changed() && !state.is_changed() {
        return;
    }
    let Some(p) = plan.0.as_deref() else { return };
    if let Ok(mut t) = text.single_mut() {
        let (tw, th) = p.target_size();
        let n = p.selected.len();
        // s2 dominates and is quadratic: calibrated 227 s at 120 frames on
        // the Strix Halo box, plus roughly linear DINO+depth overhead
        let n_f = n as f64;
        let est = 227.0 * (n_f / 120.0).powi(2) + 100.0 * (n_f / 120.0);
        t.0 = format!(
            "{n}/{} keyframes → {tw}x{th} · est ~{:.0} min on {}",
            p.budget,
            est / 60.0,
            settings.server,
        );
    }
    if let Ok(mut t) = aspect_label.single_mut() {
        t.0 = match p.aspect {
            AspectChoice::Auto => "aspect: auto".into(),
            AspectChoice::Video(vi) => format!(
                "aspect: {}",
                p.videos[vi].path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            AspectChoice::Photo(pi) => format!(
                "aspect: {}",
                p.photos[pi].path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            ),
        };
    }
}
