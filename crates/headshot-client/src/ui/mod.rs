//! Capture GUI (doc/05): Setup (drop media, budget, tonemap) → Review
//! (keyframe editor) → Session (progressive viewer). All heavy work runs
//! in [`worker`]; these modules are thin bevy state + feathers widgets.

pub mod log;
pub mod review;
pub mod setup;
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

/// The editable plan between scan and realize.
#[derive(Resource, Default)]
pub struct PlanRes(pub Option<Box<SessionPlan>>);

/// Session parameters carried over from the CLI flags.
#[derive(Resource, Clone)]
pub struct SessionSettings {
    pub server: String,
    pub edge_threshold: f32,
    pub dump_keyframes: Option<PathBuf>,
}

pub struct CaptureUiPlugin;

impl Plugin for CaptureUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<Screen>()
            .init_resource::<PlanRes>()
            .init_resource::<setup::SetupState>()
            .add_systems(Update, pump_worker_events)
            .add_plugins((
                setup::SetupScreenPlugin,
                review::ReviewScreenPlugin,
                log::LogPanePlugin,
            ));
    }
}

/// Drain worker events every frame: progress → status line, scan results
/// → plan + screen change, session traffic → the point-cloud scene.
fn pump_worker_events(
    rx: Option<Res<WorkerRx>>,
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
        match event {
            Event::Progress(m) => scene.status = m,
            Event::Discovered { videos, photos } => {
                scene.status = format!(
                    "found {} videos, {} photos — untick anything to skip, then scan",
                    videos.len(),
                    photos.len()
                );
                setup_state.discovered = Some((videos, photos));
            }
            Event::Scanned(p) => {
                setup_state.scanning = false;
                scene.status = format!("{} frames selected", p.selected.len());
                plan.0 = Some(p);
                next.set(Screen::Review);
            }
            Event::ScanFailed(m) => {
                setup_state.scanning = false;
                scene.status = format!("scan failed: {m}");
            }
            Event::PlanReturned(p) => {
                // keep it around so a future "back to review" can re-edit
                plan.0 = Some(p);
            }
            Event::Viewer(e) => crate::apply_viewer_event(&mut scene, e),
        }
    }
}
