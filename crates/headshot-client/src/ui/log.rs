//! Rolling progress log: every worker/session progress line lands here so
//! multi-minute stages stay visible. Rendered by the Setup activity pane
//! and the Reconstruct server-log panel.

use std::collections::VecDeque;

use bevy::prelude::*;

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

    /// The last `n` lines, joined for a `Text` block.
    pub fn tail(&self, n: usize) -> String {
        let start = self.0.len().saturating_sub(n);
        self.0.iter().skip(start).cloned().collect::<Vec<_>>().join("\n")
    }
}
