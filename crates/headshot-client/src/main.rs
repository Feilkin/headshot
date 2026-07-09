//! headshot client (doc/05): capture preprocessing, keyframe selection,
//! progressive point-cloud visualization, export.
//!
//! M0 skeleton: an empty bevy window. Preprocessing (M4a) and the
//! reconstruction viewer (M3) hang off this app.

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "headshot".into(),
                ..default()
            }),
            ..default()
        }))
        .run();
}
