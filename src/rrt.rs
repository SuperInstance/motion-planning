//! Rapidly-exploring Random Trees (RRT) path planner.

use crate::obstacle::{Obstacle, Point2D};

/// RRT planner configuration.
#[derive(Debug, Clone)]
pub struct RrtConfig {
    /// Step size for tree expansion
    pub step_size: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Goal bias probability (0..1)
    pub goal_bias: f64,
    /// Goal tolerance
    pub goal_tolerance: f64,
}

impl Default for RrtConfig {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            max_iterations: 5000,
            goal_bias: 0.1,
            goal_tolerance: 0.5,
        }
    }
}

/// RRT tree node.
#[derive(Debug, Clone)]
pub struct RrtNode {
    pub point: Point2D,
    pub parent: Option<usize>,
}

/// RRT planner.
pub struct RrtPlanner {
    pub config: RrtConfig,
    nodes: Vec<RrtNode>,
}

impl RrtPlanner {
    pub fn new(config: RrtConfig) -> Self {
        Self { config, nodes: Vec::new() }
    }

    /// Plan a path from start to goal, avoiding obstacles.
    pub fn plan(&mut self, start: Point2D, goal: Point2D, obstacles: &[Obstacle], bounds: (f64, f64, f64, f64)) -> Option<Vec<Point2D>> {
        self.nodes.clear();
        self.nodes.push(RrtNode { point: start, parent: None });

        for _ in 0..self.config.max_iterations {
            // Sample random point (with goal bias)
            let sample = if random_f64() < self.config.goal_bias {
                goal
            } else {
                Point2D::new(
                    bounds.0 + random_f64() * (bounds.2 - bounds.0),
                    bounds.1 + random_f64() * (bounds.3 - bounds.1),
                )
            };

            // Find nearest node
            let (nearest_idx, _) = self.nearest(&sample);

            // Steer towards sample
            let new_point = self.steer(self.nodes[nearest_idx].point, sample);

            // Check collision
            if self.collision_free(self.nodes[nearest_idx].point, new_point, obstacles) {
                let idx = self.nodes.len();
                self.nodes.push(RrtNode { point: new_point, parent: Some(nearest_idx) });

                // Check if goal reached
                if new_point.distance_to(&goal) < self.config.goal_tolerance {
                    return Some(self.extract_path(idx, goal));
                }
            }
        }
        None
    }

    fn nearest(&self, p: &Point2D) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, node) in self.nodes.iter().enumerate() {
            let d = node.point.distance_to(p);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }

    fn steer(&self, from: Point2D, to: Point2D) -> Point2D {
        let dist = from.distance_to(&to);
        if dist <= self.config.step_size {
            to
        } else {
            let t = self.config.step_size / dist;
            from.lerp(&to, t)
        }
    }

    fn collision_free(&self, a: Point2D, b: Point2D, obstacles: &[Obstacle]) -> bool {
        let steps = 10;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = a.lerp(&b, t);
            for obs in obstacles {
                if obs.contains(&p) {
                    return false;
                }
            }
        }
        true
    }

    fn extract_path(&self, goal_idx: usize, goal: Point2D) -> Vec<Point2D> {
        let mut path = vec![goal];
        let mut idx = goal_idx;
        while let Some(parent) = self.nodes[idx].parent {
            path.push(self.nodes[idx].point);
            idx = parent;
        }
        path.push(self.nodes[0].point);
        path.reverse();
        path
    }

    /// Get the number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Simple deterministic PRNG (xorshift) for reproducibility in tests.
static mut SEED: u64 = 12345;

fn random_f64() -> f64 {
    unsafe {
        SEED ^= SEED << 13;
        SEED ^= SEED >> 7;
        SEED ^= SEED << 17;
        (SEED as f64) / (u64::MAX as f64)
    }
}

/// Reset the RNG seed (for testing).
pub fn reset_seed(seed: u64) {
    unsafe { SEED = seed; }
}

/// Smooth a path by shortcutting.
pub fn smooth_path(path: &[Point2D], obstacles: &[Obstacle], iterations: usize) -> Vec<Point2D> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut smoothed = path.to_vec();
    for _ in 0..iterations {
        if smoothed.len() <= 2 { break; }
        let i = (random_f64() * (smoothed.len() - 1) as f64) as usize;
        let j = (random_f64() * (smoothed.len() - 1) as f64) as usize;
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        if b - a <= 1 { continue; }
        // Check if direct connection is collision-free
        let collision = obstacles.iter().any(|obs| obs.intersects_segment(&smoothed[a], &smoothed[b]));
        if !collision {
            smoothed.drain(a + 1..b);
        }
    }
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrt_no_obstacles() {
        reset_seed(42);
        let mut rrt = RrtPlanner::new(RrtConfig { max_iterations: 1000, goal_tolerance: 0.5, ..Default::default() });
        let path = rrt.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 5.0),
            &[],
            (-1.0, -1.0, 6.0, 6.0),
        );
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.first().unwrap().distance_to(&Point2D::new(0.0, 0.0)) < 1e-9);
    }

    #[test]
    fn test_rrt_with_obstacle() {
        reset_seed(42);
        let mut rrt = RrtPlanner::new(RrtConfig { max_iterations: 5000, step_size: 0.3, goal_tolerance: 0.5, ..Default::default() });
        let obs = Obstacle::Circle { center: Point2D::new(2.5, 2.5), radius: 1.0 };
        let path = rrt.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 5.0),
            &[obs],
            (-1.0, -1.0, 6.0, 6.0),
        );
        assert!(path.is_some());
        if let Some(p) = path {
            assert!(crate::obstacle::path_collision_free(&p, &[
                Obstacle::Circle { center: Point2D::new(2.5, 2.5), radius: 1.0 }
            ]));
        }
    }

    #[test]
    fn test_rrt_blocked() {
        // Wall between start and goal
        reset_seed(42);
        let mut rrt = RrtPlanner::new(RrtConfig { max_iterations: 500, ..Default::default() });
        let obs = Obstacle::Rect {
            min: Point2D::new(-10.0, -10.0),
            max: Point2D::new(10.0, 10.0),
        };
        // Start is inside obstacle — should fail
        let result = rrt.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 5.0),
            &[obs],
            (-1.0, -1.0, 6.0, 6.0),
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_rrt_node_count() {
        reset_seed(42);
        let mut rrt = RrtPlanner::new(RrtConfig { max_iterations: 100, ..Default::default() });
        let _ = rrt.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 5.0),
            &[],
            (-1.0, -1.0, 6.0, 6.0),
        );
        assert!(rrt.node_count() > 1);
    }

    #[test]
    fn test_smooth_path() {
        reset_seed(42);
        let path = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(4.0, 0.0),
        ];
        let smoothed = smooth_path(&path, &[], 100);
        assert!(smoothed.len() <= path.len());
        assert!((smoothed.first().unwrap().x - 0.0).abs() < 1e-9);
        assert!((smoothed.last().unwrap().x - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_rrt_goal_bias() {
        reset_seed(42);
        let config = RrtConfig { goal_bias: 1.0, max_iterations: 100, ..Default::default() };
        let mut rrt = RrtPlanner::new(config);
        let path = rrt.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            &[],
            (-1.0, -1.0, 3.0, 3.0),
        );
        assert!(path.is_some());
    }
}
