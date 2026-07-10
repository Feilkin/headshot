//! Session log pane: the last dozen progress lines, so multi-minute
//! stages (upload, trunk, depth) are visible at a glance instead of one
//! overwritten status line.

use std::collections::VecDeque;

use bevy::feathers::theme::ThemedText;
use bevy::prelude::*;

use super::Screen;

const VISIBLE_LINES: usize = 12;
const KEPT_LINES: usize = 200;

#[derive(Resource, Default)]
pub struct StatusLog(pub VecDeque<String>);

impl StatusLog {
    pub fn push(&mut self, line: String) {
        if self.0.back() == Some(&line) {
            return;
        }
        self.0.push_back(line);
        while self.0.len() > KEPT_LINES {
            self.0.pop_front();
        }
    }
}

#[derive(Component)]
struct LogPane;

#[derive(Component)]
struct LogText;

pub struct LogPanePlugin;

impl Plugin for LogPanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StatusLog>()
            .add_systems(OnEnter(Screen::Session), spawn_pane)
            .add_systems(OnExit(Screen::Session), despawn_pane)
            .add_systems(Update, refresh.run_if(in_state(Screen::Session)));
    }
}

fn spawn_pane(mut commands: Commands) {
    commands.spawn((
        LogPane,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(8.0),
            bottom: Val::Px(8.0),
            max_width: Val::Percent(60.0),
            padding: UiRect::all(Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        GlobalZIndex(9),
        children![(LogText, Text::new(""), ThemedText, TextFont::from_font_size(11.0))],
    ));
}

fn despawn_pane(mut commands: Commands, panes: Query<Entity, With<LogPane>>) {
    for e in &panes {
        commands.entity(e).despawn();
    }
}

fn refresh(log: Res<StatusLog>, mut text: Query<&mut Text, With<LogText>>) {
    if !log.is_changed() {
        return;
    }
    let Ok(mut t) = text.single_mut() else { return };
    let start = log.0.len().saturating_sub(VISIBLE_LINES);
    t.0 = log.0.iter().skip(start).cloned().collect::<Vec<_>>().join("\n");
}
