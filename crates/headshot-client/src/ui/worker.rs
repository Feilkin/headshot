//! Background capture/session worker for the GUI (doc/05): all decode,
//! scoring, full-res extraction and protocol traffic happens here; the
//! bevy side only sends commands and drains events. Same crossbeam
//! pattern as the CLI session thread.

use std::path::PathBuf;

use crossbeam_channel::{Receiver, Sender};
use headshot_capture::keyframe::SelectParams;
use headshot_capture::video::FfmpegCli;
use headshot_capture::{CaptureConfig, SessionPlan, plan_session, realize_session};

use crate::session::{ViewerEvent, run_protocol};

pub struct ScanRequest {
    pub paths: Vec<PathBuf>,
    pub budget: usize,
    pub dlog_lut: Option<PathBuf>,
    pub dlog_parametric: bool,
}

pub struct RealizeRequest {
    pub plan: Box<SessionPlan>,
    pub server: String,
    pub edge_threshold: f32,
    pub dump_keyframes: Option<PathBuf>,
}

pub enum Command {
    /// Cheap filesystem walk: what would a scan ingest?
    Discover(Vec<PathBuf>),
    /// Score the (possibly pruned) media into an editable plan.
    Scan(ScanRequest),
    /// Execute an edited plan and drive the reconstruction session.
    Realize(Box<RealizeRequest>),
}

pub enum Event {
    Progress(String),
    /// Discovery result for the Setup tree.
    Discovered { videos: Vec<PathBuf>, photos: Vec<PathBuf> },
    /// Scan finished; the plan is handed to the UI for editing.
    Scanned(Box<SessionPlan>),
    ScanFailed(String),
    /// Realize/session traffic (the plan rides back for a later re-edit).
    PlanReturned(Box<SessionPlan>),
    Viewer(ViewerEvent),
}

pub fn run(commands: Receiver<Command>, events: Sender<Event>) {
    let backend = FfmpegCli::default();
    for cmd in commands.iter() {
        match cmd {
            Command::Discover(roots) => {
                let mut progress = |m: String| {
                    let _ = events.send(Event::Progress(m));
                };
                match headshot_capture::discover_media(&roots, &mut progress) {
                    Ok((videos, photos)) => {
                        let _ = events.send(Event::Discovered { videos, photos });
                    }
                    Err(e) => {
                        let _ = events.send(Event::Progress(format!("discovery failed: {e:#}")));
                    }
                }
            }
            Command::Scan(req) => {
                let cfg = CaptureConfig {
                    media: req.paths,
                    budget: req.budget,
                    dlog_lut: req.dlog_lut,
                    dlog_parametric: req.dlog_parametric,
                    params: SelectParams { budget: req.budget, ..Default::default() },
                };
                let mut progress = |m: String| {
                    let _ = events.send(Event::Progress(m));
                };
                match plan_session(&cfg, &backend, &mut progress) {
                    Ok(plan) => {
                        let _ = events.send(Event::Scanned(Box::new(plan)));
                    }
                    Err(e) => {
                        let _ = events.send(Event::ScanFailed(format!("{e:#}")));
                    }
                }
            }
            Command::Realize(req) => {
                let mut progress = |m: String| {
                    let _ = events.send(Event::Progress(m));
                };
                let prepared = match realize_session(&req.plan, &backend, &mut progress) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = events
                            .send(Event::Viewer(ViewerEvent::Failed(format!("{e:#}"))));
                        let _ = events.send(Event::PlanReturned(req.plan));
                        continue;
                    }
                };
                if let Some(dir) = &req.dump_keyframes
                    && let Err(e) = prepared.dump(dir)
                {
                    let _ = events.send(Event::Progress(format!("keyframe dump failed: {e:#}")));
                }
                // session events flow through a small forwarder so
                // run_protocol keeps its plain ViewerEvent channel
                let (vtx, vrx) = crossbeam_channel::unbounded();
                let fwd_events = events.clone();
                let forwarder = std::thread::spawn(move || {
                    for e in vrx.iter() {
                        let _ = fwd_events.send(Event::Viewer(e));
                    }
                });
                if let Err(e) = run_protocol(prepared, &req.server, req.edge_threshold, &vtx) {
                    let _ = vtx.send(ViewerEvent::Failed(format!("{e:#}")));
                }
                drop(vtx);
                let _ = forwarder.join();
                let _ = events.send(Event::PlanReturned(req.plan));
            }
        }
    }
}
