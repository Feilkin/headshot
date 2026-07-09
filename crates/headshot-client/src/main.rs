//! headshot viewer (doc/05 §4): drives a reconstruction session against
//! the server and renders the point cloud progressively as depth chunks
//! stream in. Filters (confidence percentile, frame groups) re-filter from
//! CPU-side chunk data without re-inference.
//!
//! Controls: drag = orbit, wheel = zoom, `[`/`]` = confidence percentile,
//! `G` = cycle frame groups (all / even / odd), `F` = toggle frusta.

mod session;

use bevy::asset::RenderAssetUsages;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use clap::Parser;
use crossbeam_channel::Receiver;
use headshot_shared::pose::Camera as PoseCamera;
use session::{ChunkPoints, SessionConfig, ViewerEvent};

/// Progressive point-cloud viewer for a reconstruction session.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Directory of JPEG/PNG frames.
    frames_dir: std::path::PathBuf,

    /// Server address.
    #[arg(long, default_value = "127.0.0.1:9276")]
    server: String,

    /// Cap on frame count.
    #[arg(long, default_value_t = 64)]
    max_frames: usize,

    /// 3×3 relative depth-jump threshold (doc/01 §5.3).
    #[arg(long, default_value_t = 0.03)]
    edge_threshold: f32,

    /// Run the session and print events without opening a window
    /// (debugging / display-less smoke test).
    #[arg(long)]
    headless: bool,
}

#[derive(Resource)]
struct Events(Receiver<ViewerEvent>);

#[derive(Resource, Default)]
struct Scene {
    chunks: Vec<ChunkPoints>,
    chunk_entities: Vec<Entity>,
    cameras: Vec<PoseCamera>,
    /// Confidence quantile dropped (0.3 keeps the top 70 %).
    conf_quantile: f32,
    conf_threshold: f32,
    frame_group: FrameGroup,
    show_frusta: bool,
    dirty: bool,
    status: String,
}

#[derive(Default, Clone, Copy, PartialEq)]
enum FrameGroup {
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
    let (tx, rx) = crossbeam_channel::unbounded();
    let config = SessionConfig {
        server: args.server,
        frames_dir: args.frames_dir,
        max_frames: args.max_frames,
        edge_threshold: args.edge_threshold,
    };
    std::thread::spawn(move || session::run(config, tx));

    if args.headless {
        for event in rx.iter() {
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

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "headshot".into(), ..default() }),
            ..default()
        }))
        .insert_resource(Events(rx))
        .insert_resource(Scene {
            conf_quantile: 0.3,
            show_frusta: true,
            ..Default::default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (poll_events, keyboard, rebuild_meshes, orbit_camera, draw_frusta, update_status),
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
        Text::new("connecting…"),
        Node { position_type: PositionType::Absolute, left: Val::Px(8.0), top: Val::Px(8.0), ..default() },
    ));
}

fn poll_events(mut scene: ResMut<Scene>, events: Res<Events>) {
    for event in events.0.try_iter() {
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
