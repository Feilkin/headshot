//! Setup screen (doc/05 §1): drop media (videos, photos, directories, an
//! optional `.cube` LUT), prune the discovered file tree, set the keyframe
//! budget and tonemap, then scan.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use bevy::feathers::controls::{
    FeathersButton, FeathersSlider, FeathersTextInput, FeathersTextInputContainer,
    FeathersToggleSwitch,
};
use bevy::feathers::theme::{ThemeBackgroundColor, ThemedText};
use bevy::feathers::tokens;
use bevy::picking::events::{Click, Pointer};
use bevy::prelude::*;
use bevy::text::EditableText;
use bevy::ui_widgets::{Activate, SliderStep, SliderValue, ValueChange, checkbox_self_update};
use bevy::window::FileDragAndDrop;

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
struct BudgetLabel;

#[derive(Component, Default, Clone)]
struct TonemapLabel;

#[derive(Component, Default, Clone)]
struct PathInput;

#[derive(Component)]
struct TreeDirRow(PathBuf);

#[derive(Component)]
struct TreeVideoRow(PathBuf);

pub struct SetupScreenPlugin;

impl Plugin for SetupScreenPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Setup), spawn_screen.spawn())
            .add_systems(OnExit(Screen::Setup), despawn_screen)
            .add_systems(
                Update,
                (accept_drops, auto_discover, refresh_labels).run_if(in_state(Screen::Setup)),
            );
    }
}

fn spawn_screen() -> impl Scene {
    bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            row_gap: px(14),
            padding: px(24),
        }
        SetupRoot
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            (Text("headshot — new reconstruction") ThemedText),
            (Text("add media folders/files below, or drag & drop \
                   (drops need X11 — Wayland can't deliver them); \
                   click tree rows to include/exclude before scanning")
                ThemedText),
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
                        Node { min_width: px(420) }
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
                        @FeathersButton
                        on(add_typed_path)
                        Children [ (Text("Add") ThemedText) ]
                    ),
                ]
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Start,
                    row_gap: px(2),
                    min_height: px(60),
                    max_height: px(260),
                    overflow: Overflow::scroll_y(),
                }
                SourceList
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                }
                Children [
                    (Text("budget") ThemedText),
                    (
                        @FeathersSlider {
                            @max: 400.0,
                            @value: 200.0,
                        }
                        Node { min_width: px(240) }
                        SliderStep(8.)
                        on(|change: On<ValueChange<f32>>,
                           mut state: ResMut<SetupState>,
                           mut commands: Commands| {
                            state.budget = (change.value.round() as usize).max(8);
                            commands.entity(change.source).insert(SliderValue(change.value));
                        })
                    ),
                    (Text("200 keyframes") ThemedText BudgetLabel),
                ]
            ),
            (
                Node {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                }
                Children [
                    (Text("parametric D-Log fallback") ThemedText),
                    (
                        @FeathersToggleSwitch
                        on(checkbox_self_update)
                        on(|change: On<ValueChange<bool>>, mut state: ResMut<SetupState>| {
                            state.dlog_parametric = change.value;
                        })
                    ),
                    (Text("tonemap: none") ThemedText TonemapLabel),
                ]
            ),
            (
                @FeathersButton
                on(|_: On<Activate>,
                   mut state: ResMut<SetupState>,
                   tx: Res<WorkerTx>,
                   mut scene: ResMut<crate::Scene>| {
                    if state.scanning {
                        return;
                    }
                    let files = state.effective_files();
                    if files.is_empty() {
                        scene.status = "nothing to scan — add media first".into();
                        return;
                    }
                    state.scanning = true;
                    scene.status = format!("scanning {} files…", files.len());
                    let _ = tx.0.send(Command::Scan(ScanRequest {
                        paths: files,
                        budget: state.budget,
                        dlog_lut: state.dlog_lut.clone(),
                        dlog_parametric: state.dlog_parametric,
                    }));
                })
                Children [ (Text("Scan & select keyframes") ThemedText) ]
            ),
        ]
    }
}

fn despawn_screen(mut commands: Commands, roots: Query<Entity, With<SetupRoot>>) {
    for e in &roots {
        commands.entity(e).despawn();
    }
}

/// The reliable path when drag & drop can't reach us (Wayland): type or
/// paste a path and press Add. `.cube` files become the LUT here too.
fn add_typed_path(
    _: On<Activate>,
    input: Query<&EditableText, With<PathInput>>,
    mut state: ResMut<SetupState>,
    mut scene: ResMut<crate::Scene>,
) {
    let Ok(text) = input.single() else { return };
    let typed = text.value().to_string();
    let typed = typed.trim();
    if typed.is_empty() {
        return;
    }
    let path = PathBuf::from(shellexpand_home(typed));
    if !path.exists() {
        scene.status = format!("not found: {}", path.display());
        return;
    }
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("cube")) {
        state.dlog_lut = Some(path);
    } else if !state.paths.contains(&path) {
        state.paths.push(path);
    }
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

fn toggle_dir(
    click: On<Pointer<Click>>,
    rows: Query<&TreeDirRow>,
    mut state: ResMut<SetupState>,
) {
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

fn refresh_labels(
    state: Res<SetupState>,
    mut commands: Commands,
    list: Query<Entity, With<SourceList>>,
    mut budget: Query<&mut Text, (With<BudgetLabel>, Without<TonemapLabel>)>,
    mut tonemap: Query<&mut Text, (With<TonemapLabel>, Without<BudgetLabel>)>,
) {
    if !state.is_changed() {
        return;
    }
    if let Ok(mut t) = budget.single_mut() {
        t.0 = format!("{} keyframes", state.budget);
    }
    if let Ok(mut t) = tonemap.single_mut() {
        t.0 = match (&state.dlog_lut, state.dlog_parametric) {
            (Some(p), _) => format!("tonemap: {}", p.display()),
            (None, true) => "tonemap: parametric D-Log".into(),
            (None, false) => "tonemap: none".into(),
        };
    }
    let Ok(list) = list.single() else { return };
    commands.entity(list).despawn_related::<Children>();

    let Some((videos, photos)) = &state.discovered else {
        let hint = if state.paths.is_empty() {
            "(nothing added yet)".to_string()
        } else {
            format!("discovering {} roots…", state.paths.len())
        };
        commands.entity(list).with_children(|parent| {
            parent.spawn((Text::new(hint), ThemedText));
        });
        return;
    };

    // group by parent directory: dir rows toggle everything below them,
    // video files get their own rows (clips matter individually)
    let mut dirs: BTreeMap<PathBuf, (Vec<PathBuf>, usize)> = BTreeMap::new();
    for v in videos {
        let dir = v.parent().unwrap_or(Path::new("")).to_owned();
        dirs.entry(dir).or_default().0.push(v.clone());
    }
    for p in photos {
        let dir = p.parent().unwrap_or(Path::new("")).to_owned();
        dirs.entry(dir).or_default().1 += 1;
    }
    let mark = |on: bool| if on { "[x]" } else { "[  ]" };
    commands.entity(list).with_children(|parent| {
        for (dir, (vids, n_photos)) in &dirs {
            let dir_on = !state.excluded_dirs.contains(dir);
            parent
                .spawn((
                    Text::new(format!(
                        "{} {}/ — {} clips, {} photos",
                        mark(dir_on),
                        dir.display(),
                        vids.len(),
                        n_photos,
                    )),
                    ThemedText,
                    TreeDirRow(dir.clone()),
                ))
                .observe(toggle_dir);
            for v in vids {
                let name =
                    v.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let v_on = dir_on && !state.excluded_videos.contains(v);
                parent
                    .spawn((
                        Text::new(format!("      {} {name}", mark(v_on))),
                        ThemedText,
                        TreeVideoRow(v.clone()),
                        Node { margin: UiRect::left(Val::Px(18.0)), ..default() },
                    ))
                    .observe(toggle_video);
            }
        }
    });
}
