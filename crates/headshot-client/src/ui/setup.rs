//! Setup screen (doc/05 §1, design: headshot_ui_ref.html §01): add media
//! by path (drag & drop is a convenience only — unavailable on Wayland),
//! prune the discovered file tree, set the keyframe budget and tonemap,
//! then scan. Left: add-media field + discovered tree. Right: session
//! settings, summary, scan button, activity log.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use bevy::ecs::error::Result;
use bevy::feathers::constants::fonts;
use bevy::feathers::controls::{
    ButtonVariant, FeathersButton, FeathersSlider, FeathersTextInput, FeathersTextInputContainer,
    FeathersToggleSwitch,
};
use bevy::feathers::font_styles::InheritableFont;
use bevy::feathers::palette;
use bevy::feathers::theme::ThemedText;
use bevy::picking::events::{Click, Pointer};
use bevy::prelude::*;
use bevy::text::{EditableText, FontSize};
use bevy::ui::Checked;
use bevy::ui_widgets::{
    Activate, SliderPrecision, SliderStep, SliderValue, ValueChange, checkbox_self_update,
};
use bevy::window::FileDragAndDrop;

use super::log::StatusLog;
use super::theme::{self, UiFonts};
use super::{Screen, WorkerTx, worker::Command, worker::ScanRequest};

#[derive(Resource)]
pub struct SetupState {
    pub paths: Vec<PathBuf>,
    pub budget: usize,
    pub dlog_lut: Option<PathBuf>,
    pub dlog_parametric: bool,
    pub scanning: bool,
    /// Discovery result `(videos, photos)` for the excludable tree.
    pub discovered: Option<(Vec<PathBuf>, Vec<PathBuf>)>,
    pub excluded_dirs: HashSet<PathBuf>,
    pub excluded_videos: HashSet<PathBuf>,
    /// Roots the last `Discover` was sent for (change detector).
    pub(crate) last_discover: Vec<PathBuf>,
}

impl Default for SetupState {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            budget: 200,
            dlog_lut: None,
            dlog_parametric: false,
            scanning: false,
            discovered: None,
            excluded_dirs: HashSet::new(),
            excluded_videos: HashSet::new(),
            last_discover: Vec::new(),
        }
    }
}

impl SetupState {
    fn dir_excluded(&self, path: &Path) -> bool {
        path.parent().is_some_and(|d| self.excluded_dirs.contains(d))
    }

    /// The files a scan should ingest after tree pruning.
    pub fn effective_files(&self) -> Vec<PathBuf> {
        let Some((videos, photos)) = &self.discovered else { return self.paths.clone() };
        videos
            .iter()
            .filter(|p| !self.dir_excluded(p) && !self.excluded_videos.contains(*p))
            .chain(photos.iter().filter(|p| !self.dir_excluded(p)))
            .cloned()
            .collect()
    }
}

#[derive(Component, Default, Clone)]
struct SetupRoot;

#[derive(Component, Default, Clone)]
struct SourceList;

#[derive(Component, Default, Clone)]
struct DiscoveredCounts;

#[derive(Component, Default, Clone)]
struct BudgetValue;

#[derive(Component, Default, Clone)]
struct LutLabel;

#[derive(Component, Default, Clone)]
struct ParametricToggle;

#[derive(Component, Default, Clone)]
struct SummaryLabel;

#[derive(Component, Default, Clone)]
struct ActivityStatus;

#[derive(Component, Default, Clone)]
struct ActivityLines;

#[derive(Component, Default, Clone)]
struct PathInput;

#[derive(Component)]
struct TreeDirRow(PathBuf);

#[derive(Component)]
struct TreeVideoRow(PathBuf);

pub struct SetupScreenPlugin;

impl Plugin for SetupScreenPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Setup), spawn_screen)
            .add_systems(OnExit(Screen::Setup), despawn_screen)
            .add_systems(
                Update,
                (accept_drops, auto_discover, refresh_labels, refresh_activity)
                    .run_if(in_state(Screen::Setup)),
            );
    }
}

fn spawn_screen(world: &mut World) -> Result {
    let (budget, parametric) = {
        let state = world.resource::<SetupState>();
        (state.budget, state.dlog_parametric)
    };
    world.spawn_scene(screen(budget))?;
    if parametric {
        // mirror the CLI preset onto the freshly spawned switch
        let mut q = world.query_filtered::<Entity, With<ParametricToggle>>();
        let entities: Vec<Entity> = q.iter(world).collect();
        for e in entities {
            world.entity_mut(e).insert(Checked);
        }
    }
    Ok(())
}

fn screen(budget: usize) -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
        }
        SetupRoot
        BackgroundColor(palette::GRAY_0)
        Children [
            {theme::header_bar("/ setup", Screen::Setup,
                theme::t_mono(12.0, palette::LIGHT_GRAY_2, "new session"))},
            (
                // body: media column + settings sidebar
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
                            padding: px(20),
                            row_gap: px(16),
                        }
                        Children [
                            {theme::t_bold(13.0, palette::WHITE, "Add media")},
                            (
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Row,
                                    align_items: AlignItems::Center,
                                    column_gap: px(8),
                                }
                                Children [
                                    (
                                        @FeathersTextInputContainer
                                        Node { flex_grow: 1.0 }
                                        Children [
                                            (
                                                @FeathersTextInput {
                                                    @visible_width: 48f32,
                                                    @max_characters: 512usize,
                                                }
                                                PathInput
                                            )
                                        ]
                                    ),
                                    (
                                        @FeathersButton {
                                            @variant: ButtonVariant::Primary,
                                        }
                                        on(add_typed_path)
                                        Children [ (Text("Add") ThemedText) ]
                                    ),
                                ]
                            ),
                            {theme::t_mono(11.0, palette::LIGHT_GRAY_2,
                                "Paste or type a path — .cube files become the tonemap LUT. \
                                 Drag-and-drop is a convenience only — unavailable on Wayland.")},
                            (
                                // discovered media pane
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Column,
                                    flex_grow: 1.0,
                                    min_height: px(0),
                                    border_radius: BorderRadius::all(px(4)),
                                    overflow: Overflow::clip(),
                                }
                                BackgroundColor(palette::GRAY_1)
                                Children [
                                    (
                                        Node {
                                            display: Display::Flex,
                                            flex_direction: FlexDirection::Row,
                                            align_items: AlignItems::Center,
                                            height: px(34),
                                            padding: UiRect::horizontal(px(12)),
                                            border: UiRect::bottom(px(1)),
                                            flex_shrink: 0.0,
                                        }
                                        BorderColor::all(theme::BORDER_SUBTLE)
                                        Children [
                                            {theme::t_bold(12.0, palette::LIGHT_GRAY_1, "Discovered media")},
                                            {theme::spacer()},
                                            (
                                                Node
                                                InheritableFont {
                                                    font: fonts::MONO,
                                                    font_size: FontSize::Px(11.0),
                                                }
                                                Children [
                                                    (Text("") ThemedText
                                                     TextColor(palette::LIGHT_GRAY_2)
                                                     DiscoveredCounts)
                                                ]
                                            ),
                                        ]
                                    ),
                                    (
                                        Node {
                                            display: Display::Flex,
                                            flex_direction: FlexDirection::Column,
                                            flex_grow: 1.0,
                                            min_height: px(0),
                                            padding: px(6),
                                            row_gap: px(2),
                                            overflow: Overflow::scroll_y(),
                                        }
                                        SourceList
                                    ),
                                ]
                            ),
                        ]
                    ),
                    (
                        // session settings sidebar
                        Node {
                            display: Display::Flex,
                            flex_direction: FlexDirection::Column,
                            width: px(400),
                            flex_shrink: 0.0,
                            padding: px(20),
                            row_gap: px(16),
                            border: UiRect::left(px(1)),
                        }
                        BorderColor::all(theme::BORDER_SUBTLE)
                        Children [
                            {theme::t_bold(13.0, palette::WHITE, "Session settings")},
                            (
                                // keyframe budget pane
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Column,
                                    row_gap: px(10),
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
                                                InheritableFont {
                                                    font: fonts::MONO,
                                                    font_size: FontSize::Px(14.0),
                                                }
                                                Children [
                                                    (Text({budget.to_string()}) ThemedText
                                                     TextColor(palette::ACCENT)
                                                     BudgetValue)
                                                ]
                                            ),
                                        ]
                                    ),
                                    (
                                        @FeathersSlider {
                                            @min: 8.0,
                                            @max: 400.0,
                                            @value: {budget as f32},
                                        }
                                        SliderStep(8.)
                                        // feathers' fill/text sync needs this
                                        SliderPrecision(0)
                                        on(|change: On<ValueChange<f32>>,
                                           mut state: ResMut<SetupState>,
                                           mut commands: Commands| {
                                            state.budget = (change.value.round() as usize).max(8);
                                            commands.entity(change.source).insert(SliderValue(change.value));
                                        })
                                    ),
                                    (
                                        Node {
                                            display: Display::Flex,
                                            flex_direction: FlexDirection::Row,
                                        }
                                        Children [
                                            {theme::t_mono(10.0, palette::LIGHT_GRAY_2, "8")},
                                            {theme::spacer()},
                                            {theme::t_mono(10.0, palette::LIGHT_GRAY_2,
                                                "cost grows with the square of frame count")},
                                            {theme::spacer()},
                                            {theme::t_mono(10.0, palette::LIGHT_GRAY_2, "400")},
                                        ]
                                    ),
                                ]
                            ),
                            (
                                // tonemap pane
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Column,
                                    row_gap: px(12),
                                    padding: px(14),
                                    border_radius: BorderRadius::all(px(4)),
                                }
                                BackgroundColor(palette::GRAY_1)
                                Children [
                                    {theme::t_bold(12.0, palette::LIGHT_GRAY_1, "Tonemap")},
                                    (
                                        Node {
                                            display: Display::Flex,
                                            flex_direction: FlexDirection::Column,
                                            row_gap: px(6),
                                        }
                                        Children [
                                            {theme::t_sans(11.0, palette::LIGHT_GRAY_2, ".cube LUT")},
                                            (
                                                Node {
                                                    display: Display::Flex,
                                                    align_items: AlignItems::Center,
                                                    height: px(30),
                                                    padding: UiRect::horizontal(px(10)),
                                                    border: UiRect::all(px(1)),
                                                    border_radius: BorderRadius::all(px(4)),
                                                    overflow: Overflow::clip(),
                                                }
                                                BackgroundColor(palette::GRAY_1)
                                                BorderColor::all(theme::BORDER_SUBTLE)
                                                InheritableFont {
                                                    font: fonts::MONO,
                                                    font_size: FontSize::Px(12.0),
                                                }
                                                Children [
                                                    (Text("none — add a .cube via the path field")
                                                     ThemedText
                                                     TextColor(palette::LIGHT_GRAY_2)
                                                     LutLabel)
                                                ]
                                            ),
                                        ]
                                    ),
                                    (
                                        Node {
                                            display: Display::Flex,
                                            flex_direction: FlexDirection::Row,
                                            align_items: AlignItems::FlexStart,
                                            column_gap: px(10),
                                        }
                                        Children [
                                            (
                                                @FeathersToggleSwitch
                                                ParametricToggle
                                                on(checkbox_self_update)
                                                on(|change: On<ValueChange<bool>>,
                                                   mut state: ResMut<SetupState>| {
                                                    state.dlog_parametric = change.value;
                                                })
                                            ),
                                            (
                                                Node {
                                                    display: Display::Flex,
                                                    flex_direction: FlexDirection::Column,
                                                    row_gap: px(2),
                                                }
                                                Children [
                                                    {theme::t_sans(12.0, palette::LIGHT_GRAY_1,
                                                        "Parametric D-Log fallback")},
                                                    {theme::t_mono(10.0, palette::LIGHT_GRAY_2,
                                                        "used when no LUT is set")},
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            {theme::spacer()},
                            (
                                // summary pane
                                Node {
                                    padding: UiRect::axes(px(14), px(12)),
                                    border_radius: BorderRadius::all(px(4)),
                                }
                                BackgroundColor(palette::GRAY_1)
                                InheritableFont {
                                    font: fonts::MONO,
                                    font_size: FontSize::Px(12.0),
                                }
                                Children [
                                    (Text("add media to see the session summary")
                                     ThemedText
                                     TextColor(palette::LIGHT_GRAY_1)
                                     SummaryLabel)
                                ]
                            ),
                            (
                                @FeathersButton {
                                    @variant: ButtonVariant::Primary,
                                }
                                on(start_scan)
                                Children [ (Text("Scan & select keyframes") ThemedText) ]
                            ),
                            (
                                // activity / log pane
                                Node {
                                    display: Display::Flex,
                                    flex_direction: FlexDirection::Column,
                                    row_gap: px(8),
                                    padding: UiRect::axes(px(12), px(11)),
                                    border: UiRect::all(px(1)),
                                    border_radius: BorderRadius::all(px(4)),
                                }
                                BackgroundColor(palette::BLACK)
                                BorderColor::all(theme::BORDER_SUBTLE)
                                InheritableFont {
                                    font: fonts::MONO,
                                    font_size: FontSize::Px(11.0),
                                }
                                Children [
                                    (Text("idle") ThemedText
                                     TextColor(palette::LIGHT_GRAY_2)
                                     ActivityStatus),
                                    (Text("") ThemedText
                                     TextColor(palette::LIGHT_GRAY_2)
                                     ActivityLines),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ]
    }
}

fn despawn_screen(mut commands: Commands, roots: Query<Entity, With<SetupRoot>>) {
    for e in &roots {
        commands.entity(e).despawn();
    }
}

// ---- observers ------------------------------------------------------------

/// The reliable path when drag & drop can't reach us (Wayland): type or
/// paste a path and press Add. `.cube` files become the LUT here too.
fn add_typed_path(
    _: On<Activate>,
    input: Query<&EditableText, With<PathInput>>,
    mut state: ResMut<SetupState>,
    mut log: ResMut<StatusLog>,
) {
    let Ok(text) = input.single() else { return };
    let typed = text.value().to_string();
    let typed = typed.trim();
    if typed.is_empty() {
        return;
    }
    let path = PathBuf::from(shellexpand_home(typed));
    if !path.exists() {
        log.push(format!("not found: {}", path.display()));
        return;
    }
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("cube")) {
        state.dlog_lut = Some(path);
    } else if !state.paths.contains(&path) {
        state.paths.push(path);
    }
}

fn start_scan(_: On<Activate>, mut state: ResMut<SetupState>, tx: Res<WorkerTx>, mut log: ResMut<StatusLog>) {
    if state.scanning {
        return;
    }
    let files = state.effective_files();
    if files.is_empty() {
        log.push("nothing to scan — add media first".into());
        return;
    }
    state.scanning = true;
    log.push(format!("scanning {} files…", files.len()));
    let _ = tx.0.send(Command::Scan(ScanRequest {
        paths: files,
        budget: state.budget,
        dlog_lut: state.dlog_lut.clone(),
        dlog_parametric: state.dlog_parametric,
    }));
}

/// `~/…` → `$HOME/…` (no shell involved, just the common case).
fn shellexpand_home(s: &str) -> String {
    match (s.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => {
            let mut p = PathBuf::from(home);
            p.push(rest);
            p.to_string_lossy().into_owned()
        }
        _ => s.to_string(),
    }
}

fn toggle_dir(click: On<Pointer<Click>>, rows: Query<&TreeDirRow>, mut state: ResMut<SetupState>) {
    if let Ok(row) = rows.get(click.entity)
        && !state.excluded_dirs.remove(&row.0)
    {
        state.excluded_dirs.insert(row.0.clone());
    }
}

fn toggle_video(
    click: On<Pointer<Click>>,
    rows: Query<&TreeVideoRow>,
    mut state: ResMut<SetupState>,
) {
    if let Ok(row) = rows.get(click.entity)
        && !state.excluded_videos.remove(&row.0)
    {
        state.excluded_videos.insert(row.0.clone());
    }
}

// ---- systems ----------------------------------------------------------------

/// Dropped `.cube` files become the LUT; everything else joins the media
/// list (the scanner classifies and recurses).
fn accept_drops(mut drops: MessageReader<FileDragAndDrop>, mut state: ResMut<SetupState>) {
    for d in drops.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = d else { continue };
        if path_buf.extension().is_some_and(|e| e.eq_ignore_ascii_case("cube")) {
            state.dlog_lut = Some(path_buf.clone());
        } else if !state.paths.contains(path_buf) {
            state.paths.push(path_buf.clone());
        }
    }
}

/// Send a discovery walk whenever the media roots change.
fn auto_discover(mut state: ResMut<SetupState>, tx: Res<WorkerTx>) {
    if !state.is_changed() || state.paths == state.last_discover || state.paths.is_empty() {
        return;
    }
    state.last_discover = state.paths.clone();
    let _ = tx.0.send(Command::Discover(state.paths.clone()));
}

/// Scanning indicator + last log lines in the activity pane.
fn refresh_activity(
    state: Res<SetupState>,
    log: Res<StatusLog>,
    mut status: Query<(&mut Text, &mut TextColor), With<ActivityStatus>>,
    mut lines: Query<&mut Text, (With<ActivityLines>, Without<ActivityStatus>)>,
) {
    if !state.is_changed() && !log.is_changed() {
        return;
    }
    if let Ok((mut t, mut c)) = status.single_mut() {
        (t.0, c.0) = if state.scanning {
            ("scanning…".into(), palette::ACCENT)
        } else {
            ("idle".into(), palette::LIGHT_GRAY_2)
        };
    }
    if let Ok(mut t) = lines.single_mut() {
        let start = log.0.len().saturating_sub(4);
        t.0 = log.0.iter().skip(start).cloned().collect::<Vec<_>>().join("\n");
    }
}

/// Mirror `SetupState` into the labels and rebuild the discovered tree.
#[allow(clippy::type_complexity)]
fn refresh_labels(
    state: Res<SetupState>,
    fonts: Option<Res<UiFonts>>,
    mut commands: Commands,
    list: Query<Entity, With<SourceList>>,
    mut texts: ParamSet<(
        Query<&mut Text, With<BudgetValue>>,
        Query<&mut Text, With<LutLabel>>,
        Query<&mut Text, With<SummaryLabel>>,
        Query<&mut Text, With<DiscoveredCounts>>,
    )>,
    settings: Res<super::SessionSettings>,
) {
    if !state.is_changed() {
        return;
    }
    let Some(fonts) = fonts else { return };
    if let Ok(mut t) = texts.p0().single_mut() {
        t.0 = state.budget.to_string();
    }
    if let Ok(mut t) = texts.p1().single_mut() {
        t.0 = match (&state.dlog_lut, state.dlog_parametric) {
            (Some(p), _) => {
                p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
            }
            (None, true) => "none — parametric fallback active".into(),
            (None, false) => "none — add a .cube via the path field".into(),
        };
    }

    let counts = state.discovered.as_ref().map(|(videos, photos)| {
        let dirs: HashSet<_> =
            videos.iter().chain(photos).filter_map(|p| p.parent()).collect();
        (dirs.len(), videos.len(), photos.len())
    });
    if let Ok(mut t) = texts.p3().single_mut() {
        t.0 = match counts {
            Some((d, v, p)) => format!("{d} folders · {v} clips · {p} photos"),
            None => String::new(),
        };
    }
    if let Ok(mut t) = texts.p2().single_mut() {
        t.0 = match counts {
            Some((_, v, p)) => format!(
                "{p} photos + {v} clips → up to {} keyframes\nshared frame shape chosen in Review · server {}",
                state.budget, settings.server,
            ),
            None => "add media to see the session summary".into(),
        };
    }

    let Ok(list) = list.single() else { return };
    commands.entity(list).despawn_related::<Children>();

    let Some((videos, photos)) = &state.discovered else {
        let hint = if state.paths.is_empty() {
            "nothing added yet — type a path above".to_string()
        } else {
            format!("discovering {} roots…", state.paths.len())
        };
        commands.entity(list).with_children(|parent| {
            parent.spawn((
                Node { padding: UiRect::all(Val::Px(8.0)), ..default() },
                children![theme::mono(&fonts, 11.0, palette::LIGHT_GRAY_2, hint)],
            ));
        });
        return;
    };

    // group by parent directory: dir rows toggle everything below them,
    // video files get their own rows (clips matter individually)
    let mut dirs: BTreeMap<PathBuf, (Vec<PathBuf>, usize, usize)> = BTreeMap::new();
    for v in videos {
        let dir = v.parent().unwrap_or(Path::new("")).to_owned();
        dirs.entry(dir).or_default().0.push(v.clone());
    }
    for p in photos {
        let dir = p.parent().unwrap_or(Path::new("")).to_owned();
        let entry = dirs.entry(dir).or_default();
        entry.1 += 1;
        let is_raw = p
            .extension()
            .is_some_and(|e| headshot_capture::raw::is_raw_ext(&e.to_string_lossy()));
        if is_raw {
            entry.2 += 1;
        }
    }
    commands.entity(list).with_children(|parent| {
        for (dir, (vids, n_photos, n_raw)) in &dirs {
            let dir_on = !state.excluded_dirs.contains(dir);
            let dir_name = dir
                .file_name()
                .map(|n| format!("{}/", n.to_string_lossy()))
                .unwrap_or_else(|| dir.display().to_string());
            let mut row = parent.spawn((dir_row_node(32.0, 8.0), TreeDirRow(dir.clone())));
            row.observe(toggle_dir).with_children(|row| {
                let name_color = if dir_on { palette::WHITE } else { palette::LIGHT_GRAY_2 };
                row.spawn(theme::check_square(dir_on));
                row.spawn(theme::mono(&fonts, 13.0, name_color, dir_name));
                if *n_raw > 0 {
                    row.spawn(theme::badge(
                        &fonts,
                        palette::LIGHT_GRAY_1,
                        palette::GRAY_3,
                        format!("{n_raw} RAW"),
                    ));
                }
                if !dir_on {
                    row.spawn(theme::badge(
                        &fonts,
                        palette::LIGHT_GRAY_2,
                        palette::GRAY_2,
                        "excluded",
                    ));
                }
                row.spawn((Node { flex_grow: 1.0, ..default() },));
                row.spawn(theme::mono(
                    &fonts,
                    11.0,
                    palette::LIGHT_GRAY_2,
                    format!("{} clips · {} photos", vids.len(), n_photos),
                ));
            });
            for v in vids {
                let name =
                    v.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let v_on = dir_on && !state.excluded_videos.contains(v);
                let mut row = parent.spawn((dir_row_node(30.0, 34.0), TreeVideoRow(v.clone())));
                row.observe(toggle_video).with_children(|row| {
                    let name_color =
                        if v_on { palette::LIGHT_GRAY_1 } else { palette::LIGHT_GRAY_2 };
                    row.spawn(theme::check_square(v_on));
                    row.spawn(theme::mono(&fonts, 12.0, name_color, name));
                });
            }
        }
    });
}

fn dir_row_node(height: f32, indent: f32) -> Node {
    Node {
        display: Display::Flex,
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        column_gap: Val::Px(10.0),
        height: Val::Px(height),
        padding: UiRect::horizontal(Val::Px(8.0)).with_left(Val::Px(indent)),
        border_radius: BorderRadius::all(Val::Px(3.0)),
        flex_shrink: 0.0,
        ..default()
    }
}
