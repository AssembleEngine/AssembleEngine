use std::collections::VecDeque;
use std::sync::Mutex;

/// An observer-eligible impact for the water surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterImpact {
    pub position: [f32; 2],
    pub radius: f32,
    pub seed: u32,
}

/// The client publishes impacts here for the renderer.
#[derive(Debug, Default)]
pub struct WaterFeed {
    queue: Mutex<Queue>,
}

#[derive(Debug, Default)]
struct Queue {
    generation: u64,
    impacts: VecDeque<WaterImpact>,
}

impl WaterFeed {
    pub fn emit(&self, impact: WaterImpact) {
        let mut guard = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        let queue = &mut guard.impacts;
        const MAX_PENDING: usize = 1024;
        if queue.len() == MAX_PENDING {
            queue.pop_front();
        }
        queue.push_back(impact);
    }

    pub fn take(&self) -> VecDeque<WaterImpact> {
        self.drain().1
    }

    pub(crate) fn drain(&self) -> (u64, VecDeque<WaterImpact>) {
        let mut queue = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        (queue.generation, std::mem::take(&mut queue.impacts))
    }

    pub fn clear(&self) {
        let mut queue = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        queue.impacts.clear();
        queue.generation = queue.generation.wrapping_add(1);
    }
}
