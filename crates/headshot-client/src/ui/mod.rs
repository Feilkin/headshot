//! Capture GUI (doc/05): Setup (drop media, budget, tonemap) → Review
//! (keyframe editor) → Session (progressive viewer). All heavy work runs
//! in [`worker`]; these modules are thin bevy state + feathers widgets.

pub mod log;
pub mod reconstruct;
pub mod review;
pub mod setup;
pub mod theme;
pub mod worker;

use std::path::PathBuf;

use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use headshot_capture::SessionPlan;

use crate::Scene as CloudScene;
use crate::session::ViewerEvent;
use worker::{Command, Event};

#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Screen {
    #[default]
    Setup,
    Review,
    Session,
}

#[derive(Resource)]
pub struct WorkerTx(pub Sender<Command>);

#[derive(Resource)]
pub struct WorkerRx(pub Receiver<Event>);

/// Sender side of the same event stream, for UI-spawned background work
/// (exports) to report progress through the normal pump.
#[derive(Resource)]
pub struct EventTx(pub Sender<Event>);

/// The editable plan between scan and realize.
#[derive(Resource, Default)]
pub struct PlanRes(pub Option<Box<SessionPlan>>);

/// Session parameters carried over from the CLI flags.
#[derive(Resource, Clone)]
pub struct SessionSettings {
    pub server: String,
    pub edge_threshold: f32,
    pub dump_keyframes: Option<PathBuf>,
    /// Auto-export the cloud here when a session finishes.
    pub export_ply: Option<PathBuf>,
}

/// Export on a background thread (clouds run to hundreds of MB); progress
/// and the result land in the status log via the event pump.
pub fn spawn_export(
    chunks: Vec<std::sync::Arc<crate::session::ChunkPoints>>,
    cameras: Vec<headshot_shared::pose::Camera>,
    path: PathBuf,
    tx: Sender<Event>,
) {
    std::thread::spawn(move || {
        let n: usize = chunks.iter().map(|c| c.positions.len()).sum();
        let _ = tx.send(Event::Progress(format!("exporting {n} points…")));
        let msg = match crate::export::export_ply(&chunks, &cameras, &path) {
            Ok(stats) => format!(
                "exported {} points → {} (+ {})",
                stats.points,
                stats.ply.display(),
                stats.cameras.display(),
            ),
            Err(e) => format!("export failed: {e:#}"),
        };
        let _ = tx.send(Event::Progress(msg));
    });
}

pub struct CaptureUiPlugin;

impl Plugin for CaptureUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<Screen>()
            .init_resource::<PlanRes>()
            .init_resource::<setup::SetupState>()
            .init_resource::<log::StatusLog>()
            .add_systems(Startup, theme::load_fonts)
            .add_systems(Update, pump_worker_events)
            .add_plugins((
                setup::SetupScreenPlugin,
                review::ReviewScreenPlugin,
                reconstruct::ReconstructScreenPlugin,
            ));
    }
}

/// Drain worker events every frame: progress → status line, scan results
/// → plan + screen change, session traffic → the point-cloud scene.
#[allow(clippy::too_many_arguments)]
fn pump_worker_events(
    rx: Option<Res<WorkerRx>>,
    tx: Res<EventTx>,
    settings: Res<SessionSettings>,
    mut scene: ResMut<CloudScene>,
    mut plan: ResMut<PlanRes>,
    mut setup_state: ResMut<setup::SetupState>,
    mut status_log: ResMut<log::StatusLog>,
    mut next: ResMut<NextState<Screen>>,
) {
    let Some(rx) = rx else { return };
    for event in rx.0.try_iter() {
        if let Event::Progress(m) = &event {
            status_log.push(m.clone());
        }
        if let Event::Viewer(ViewerEvent::Status(m)) = &event {
            status_log.push(m.clone());
        }
        if let Event::Viewer(ViewerEvent::Done) = &event
            && let Some(path) = &settings.export_ply
        {
            spawn_export(
                scene.chunks.clone(),
                scene.cameras.clone(),
                path.clone(),
                tx.0.clone(),
            );
        }
        match event {
            Event::Progress(m) => scene.status = m,
            Event::Discovered { videos, photos } => {
                status_log.push(format!(
                    "found {} videos, {} photos — untick anything to skip, then scan",
                    videos.len(),
                    photos.len()
                ));
                setup_state.discovered = Some((videos, photos));
            }
            Event::Scanned(p) => {
                setup_state.scanning = false;
                status_log.push(format!("{} frames selected", p.selected.len()));
                plan.0 = Some(p);
                next.set(Screen::Review);
            }
            Event::ScanFailed(m) => {
                setup_state.scanning = false;
                status_log.push(format!("scan failed: {m}"));
            }
            Event::PlanReturned(p) => {
                // keep it around so a future "back to review" can re-edit
                plan.0 = Some(p);
            }
            Event::Viewer(e) => crate::apply_viewer_event(&mut scene, e),
        }
    }
}
