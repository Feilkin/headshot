//! headshot client (doc/05): capture GUI (drop media → review keyframes →
//! reconstruct) and progressive point-cloud viewer. With a media path on
//! the command line it runs the automatic CLI flow (M3-compatible);
//! without one it opens the Setup screen.
//!
//! Viewer controls: drag = orbit, wheel = zoom, `[`/`]` = confidence
//! percentile, `G` = cycle frame groups (all / even / odd), `F` = frusta.

mod session;
mod ui;

use bevy::asset::RenderAssetUsages;
use bevy::feathers::FeathersPlugins;
use bevy::feathers::dark_theme::create_dark_theme;
use bevy::feathers::theme::UiTheme;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use clap::Parser;
use headshot_shared::pose::Camera as PoseCamera;
use session::{ChunkPoints, SessionConfig, ViewerEvent};

/// Capture GUI + progressive point-cloud viewer for reconstruction
/// sessions.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Mixed media: a directory (searched recursively) of videos, photos
    /// and sidecar .srt telemetry, or a single file (doc/05 §1). Omit to
    /// open the interactive Setup screen.
    media: Option<std::path::PathBuf>,

    /// Server address.
    #[arg(long, default_value = "127.0.0.1:9276")]
    server: String,

    /// Total keyframe budget across all sources (doc/05 §2; server cost is
    /// quadratic in frame count).
    #[arg(long, default_value_t = 200)]
    budget: usize,

    /// Official DJI D-Log→Rec.709 .cube LUT, applied to video frames.
    #[arg(long)]
    dlog_lut: Option<std::path::PathBuf>,

    /// Tonemap video frames with the parametric D-Log approximation
    /// (no .cube available; doc/05 §1.1).
    #[arg(long)]
    dlog: bool,

    /// Write the preprocessed keyframes + manifest.json to this directory.
    #[arg(long)]
    dump_keyframes: Option<std::path::PathBuf>,

    /// 3×3 relative depth-jump threshold (doc/01 §5.3).
    #[arg(long, default_value_t = 0.03)]
    edge_threshold: f32,

    /// Run the session and print events without opening a window
    /// (debugging / display-less smoke test; requires a media path).
    #[arg(long)]
    headless: bool,

    /// With a media path: open the Setup/Review UI pre-populated instead
    /// of reconstructing immediately (also the workaround for Wayland,
    /// where drag & drop can't reach the window).
    #[arg(long)]
    review: bool,
}

#[derive(Resource)]
pub struct Scene {
    pub chunks: Vec<ChunkPoints>,
    pub chunk_entities: Vec<Entity>,
    pub cameras: Vec<PoseCamera>,
    /// Confidence quantile dropped (0.3 keeps the top 70 %).
    pub conf_quantile: f32,
    pub conf_threshold: f32,
    pub frame_group: FrameGroup,
    pub show_frusta: bool,
    pub dirty: bool,
    pub status: String,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            chunk_entities: Vec::new(),
            cameras: Vec::new(),
            conf_quantile: 0.3,
            conf_threshold: 0.0,
            frame_group: FrameGroup::All,
            show_frusta: true,
            dirty: false,
            status: String::new(),
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq)]
pub enum FrameGroup {
    #[default]
    All,
    Even,
    Odd,
}

impl FrameGroup {
    fn keeps(self, frame: u16) -> bool {
        match self {
            FrameGroup::All => true,
            FrameGroup::Even => frame.is_multiple_of(2),
            FrameGroup::Odd => frame % 2 == 1,
        }
    }
    fn label(self) -> &'static str {
        match self {
            FrameGroup::All => "all",
            FrameGroup::Even => "even",
            FrameGroup::Odd => "odd",
        }
    }
}

/// Route one session event into the point-cloud scene (shared by the CLI
/// forwarder and the GUI worker pump).
pub fn apply_viewer_event(scene: &mut Scene, event: ViewerEvent) {
    match event {
        ViewerEvent::Status(s) => scene.status = s,
        ViewerEvent::Cameras(cams) => scene.cameras = cams,
        ViewerEvent::Chunk(chunk) => {
            scene.chunks.push(chunk);
            scene.dirty = true;
        }
        ViewerEvent::Done => {}
        ViewerEvent::Failed(e) => scene.status = format!("FAILED: {e}"),
    }
}

#[derive(Component)]
struct ChunkMesh;

#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct Orbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
}

fn main() {
    let args = Args::parse();
    let settings = ui::SessionSettings {
        server: args.server.clone(),
        edge_threshold: args.edge_threshold,
        dump_keyframes: args.dump_keyframes.clone(),
    };

    // one unified event stream feeds the app in both modes
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let (evt_tx, evt_rx) = crossbeam_channel::unbounded();

    let mut preset = ui::setup::SetupState {
        budget: args.budget,
        dlog_lut: args.dlog_lut.clone(),
        dlog_parametric: args.dlog,
        ..Default::default()
    };

    let auto_media = if args.review { None } else { args.media.clone() };
    let initial = if let Some(media) = auto_media {
        // automatic flow: capture + session start immediately (M3 CLI)
        let config = SessionConfig {
            server: args.server,
            media,
            budget: args.budget,
            dlog_lut: args.dlog_lut,
            dlog: args.dlog,
            dump_keyframes: args.dump_keyframes,
            edge_threshold: args.edge_threshold,
        };
        let (vtx, vrx) = crossbeam_channel::unbounded();
        std::thread::spawn(move || session::run(config, vtx));

        if args.headless {
            for event in vrx.iter() {
                match event {
                    ViewerEvent::Status(s) => println!("status: {s}"),
                    ViewerEvent::Cameras(cams) => println!("cameras: {}", cams.len()),
                    ViewerEvent::Chunk(c) => println!("chunk: {} points", c.positions.len()),
                    ViewerEvent::Done => {
                        println!("done");
                        return;
                    }
                    ViewerEvent::Failed(e) => {
                        eprintln!("failed: {e}");
                        std::process::exit(1);
                    }
                }
            }
            return;
        }
        let fwd = evt_tx.clone();
        std::thread::spawn(move || {
            for e in vrx.iter() {
                let _ = fwd.send(ui::worker::Event::Viewer(e));
            }
        });
        ui::Screen::Session
    } else {
        if args.headless {
            eprintln!("--headless requires a media path (and is incompatible with --review)");
            std::process::exit(2);
        }
        if let Some(media) = args.media {
            preset.paths.push(media);
        }
        ui::Screen::Setup
    };

    // GUI worker (drives scans + reconstructions; idle in automatic mode)
    std::thread::spawn(move || ui::worker::run(cmd_rx, evt_tx));

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "headshot".into(), ..default() }),
            ..default()
        }))
        .add_plugins(FeathersPlugins)
        .insert_resource(UiTheme(create_dark_theme()))
        .insert_resource(ui::WorkerTx(cmd_tx))
        .insert_resource(ui::WorkerRx(evt_rx))
        .insert_resource(settings)
        .insert_resource(preset)
        .insert_resource(Scene::default())
        .insert_state(initial)
        .add_plugins(ui::CaptureUiPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (keyboard, rebuild_meshes, orbit_camera, draw_frusta, update_status),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, -2.0).looking_at(Vec3::new(0.0, 0.0, 1.0), -Vec3::Y),
        Orbit { yaw: 0.0, pitch: 0.0, distance: 2.0, target: Vec3::new(0.0, 0.0, 1.0) },
    ));
    commands.spawn((
        StatusText,
        Text::new("ready"),
        Node { position_type: PositionType::Absolute, left: Val::Px(8.0), top: Val::Px(8.0), ..default() },
        GlobalZIndex(10),
    ));
}

fn keyboard(mut scene: ResMut<Scene>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::BracketLeft) {
        scene.conf_quantile = (scene.conf_quantile - 0.1).max(0.0);
        scene.dirty = true;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        scene.conf_quantile = (scene.conf_quantile + 0.1).min(0.9);
        scene.dirty = true;
    }
    if keys.just_pressed(KeyCode::KeyG) {
        scene.frame_group = match scene.frame_group {
            FrameGroup::All => FrameGroup::Even,
            FrameGroup::Even => FrameGroup::Odd,
            FrameGroup::Odd => FrameGroup::All,
        };
        scene.dirty = true;
    }
    if keys.just_pressed(KeyCode::KeyF) {
        scene.show_frusta = !scene.show_frusta;
    }
}

/// Rebuild all chunk meshes from CPU-side data when a filter changes or a
/// new chunk arrives (no re-inference; doc/05 §4).
fn rebuild_meshes(
    mut commands: Commands,
    mut scene: ResMut<Scene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !scene.dirty {
        return;
    }
    scene.dirty = false;

    // percentile over the global confidence distribution (doc/04 §4)
    let mut all_conf: Vec<f32> = scene.chunks.iter().flat_map(|c| c.conf.iter().copied()).collect();
    scene.conf_threshold = if all_conf.is_empty() {
        f32::NEG_INFINITY
    } else {
        let idx = ((all_conf.len() - 1) as f32 * scene.conf_quantile) as usize;
        *all_conf.select_nth_unstable_by(idx, |a, b| a.total_cmp(b)).1
    };

    for e in scene.chunk_entities.drain(..) {
        commands.entity(e).despawn();
    }
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        ..default()
    });
    let mut entities = Vec::new();
    for chunk in &scene.chunks {
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        for i in 0..chunk.positions.len() {
            if chunk.conf[i] >= scene.conf_threshold && scene.frame_group.keeps(chunk.frame[i]) {
                positions.push(chunk.positions[i]);
                colors.push(chunk.colors[i]);
            }
        }
        if positions.is_empty() {
            continue;
        }
        let mut mesh = Mesh::new(PrimitiveTopology::PointList, RenderAssetUsages::RENDER_WORLD);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        entities.push(
            commands
                .spawn((ChunkMesh, Mesh3d(meshes.add(mesh)), MeshMaterial3d(material.clone())))
                .id(),
        );
    }
    scene.chunk_entities = entities;
}

fn orbit_camera(
    mut query: Query<(&mut Orbit, &mut Transform)>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
) {
    let Ok((mut orbit, mut transform)) = query.single_mut() else { return };
    if buttons.pressed(MouseButton::Left) {
        for ev in motion.read() {
            orbit.yaw -= ev.delta.x * 0.005;
            orbit.pitch = (orbit.pitch - ev.delta.y * 0.005).clamp(-1.5, 1.5);
        }
    } else {
        motion.clear();
    }
    for ev in wheel.read() {
        orbit.distance = (orbit.distance * (1.0 - ev.y * 0.1)).clamp(0.05, 100.0);
    }
    // OpenCV convention: +y down, +z forward — orbit in that frame
    let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, orbit.pitch, 0.0);
    let offset = rot * Vec3::new(0.0, 0.0, -orbit.distance);
    transform.translation = orbit.target + offset;
    transform.look_at(orbit.target, -Vec3::Y);
}

fn draw_frusta(scene: Res<Scene>, mut gizmos: Gizmos) {
    if !scene.show_frusta {
        return;
    }
    for (i, cam) in scene.cameras.iter().enumerate() {
        if !scene.frame_group.keeps(i as u16) {
            continue;
        }
        let center = Vec3::from_array(cam.center());
        let depth = 0.15;
        let corners = [
            (0.0, 0.0),
            (2.0 * cam.cx, 0.0),
            (2.0 * cam.cx, 2.0 * cam.cy),
            (0.0, 2.0 * cam.cy),
        ]
        .map(|(x, y)| Vec3::from_array(cam.unproject(x as u32, y as u32, depth)));
        let color = if i == 0 {
            Color::srgb(1.0, 0.3, 0.2) // reference frame stands out
        } else if i % 2 == 0 {
            Color::srgb(0.2, 0.8, 1.0)
        } else {
            Color::srgb(1.0, 0.9, 0.2)
        };
        for k in 0..4 {
            gizmos.line(center, corners[k], color);
            gizmos.line(corners[k], corners[(k + 1) % 4], color);
        }
    }
}

fn update_status(scene: Res<Scene>, mut query: Query<&mut Text, With<StatusText>>) {
    let Ok(mut text) = query.single_mut() else { return };
    let points: usize = scene.chunk_entities.len();
    let total: usize = scene.chunks.iter().map(|c| c.positions.len()).sum();
    text.0 = format!(
        "{}\nchunks {} | {} pts | conf q={:.1} (≥{:.2}) | frames: {} | [ ] conf, G group, F frusta",
        scene.status,
        points,
        total,
        scene.conf_quantile,
        scene.conf_threshold,
        scene.frame_group.label(),
    );
}
