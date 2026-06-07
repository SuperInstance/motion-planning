//! Artificial potential field path planning.

use crate::obstacle::{Obstacle, Point2D};

/// Potential field planner configuration.
#[derive(Debug, Clone)]
pub struct PotentialFieldConfig {
    /// Attractive gain
    pub k_att: f64,
    /// Repulsive gain
    pub k_rep: f64,
    /// Influence distance of obstacles
    pub influence_distance: f64,
    /// Step size for gradient descent
    pub step_size: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Goal tolerance
    pub goal_tolerance: f64,
}

impl Default for PotentialFieldConfig {
    fn default() -> Self {
        Self {
            k_att: 1.0,
            k_rep: 100.0,
            influence_distance: 2.0,
            step_size: 0.05,
            max_iterations: 5000,
            goal_tolerance: 0.1,
        }
    }
}

/// Potential field planner.
pub struct PotentialFieldPlanner {
    pub config: PotentialFieldConfig,
}

impl PotentialFieldPlanner {
    pub fn new(config: PotentialFieldConfig) -> Self {
        Self { config }
    }

    /// Plan a path using gradient descent on the potential field.
    pub fn plan(&self, start: Point2D, goal: Point2D, obstacles: &[Obstacle]) -> Option<Vec<Point2D>> {
        let mut path = vec![start];
        let mut current = start;

        for _ in 0..self.config.max_iterations {
            if current.distance_to(&goal) < self.config.goal_tolerance {
                path.push(goal);
                return Some(path);
            }

            let force = self.total_force(&current, &goal, obstacles);
            let norm = (force.0.powi(2) + force.1.powi(2)).sqrt();
            if norm < 1e-9 {
                // Stuck in local minimum
                return None;
            }

            current = Point2D::new(
                current.x + self.config.step_size * force.0 / norm,
                current.y + self.config.step_size * force.1 / norm,
            );
            path.push(current);
        }
        None
    }

    /// Compute the total force at a point.
    pub fn total_force(&self, point: &Point2D, goal: &Point2D, obstacles: &[Obstacle]) -> (f64, f64) {
        let (ax, ay) = self.attractive_force(point, goal);
        let (mut rx, mut ry) = (0.0, 0.0);
        for obs in obstacles {
            let (fx, fy) = self.repulsive_force(point, obs);
            rx += fx;
            ry += fy;
        }
        (ax + rx, ay + ry)
    }

    /// Attractive force towards goal.
    pub fn attractive_force(&self, point: &Point2D, goal: &Point2D) -> (f64, f64) {
        let dx = goal.x - point.x;
        let dy = goal.y - point.y;
        let dist = (dx.powi(2) + dy.powi(2)).sqrt();

        if dist < 1.0 {
            // Conic (linear) potential near goal
            (self.config.k_att * dx, self.config.k_att * dy)
        } else {
            // Quadratic potential far from goal
            (self.config.k_att * dx / dist, self.config.k_att * dy / dist)
        }
    }

    /// Repulsive force from an obstacle.
    pub fn repulsive_force(&self, point: &Point2D, obstacle: &Obstacle) -> (f64, f64) {
        let (nearest, dist) = nearest_point_on_obstacle(point, obstacle);

        if dist > self.config.influence_distance || dist < 1e-9 {
            return (0.0, 0.0);
        }

        let repulsive_mag = self.config.k_rep * (1.0 / dist - 1.0 / self.config.influence_distance) / (dist * dist);
        let dx = point.x - nearest.x;
        let dy = point.y - nearest.y;
        let norm = (dx.powi(2) + dy.powi(2)).sqrt();
        if norm < 1e-9 {
            return (0.0, 0.0);
        }
        (repulsive_mag * dx / norm, repulsive_mag * dy / norm)
    }

    /// Evaluate the potential at a point.
    pub fn potential(&self, point: &Point2D, goal: &Point2D, obstacles: &[Obstacle]) -> f64 {
        let dist_goal = point.distance_to(goal);
        let u_att = 0.5 * self.config.k_att * dist_goal.powi(2);

        let mut u_rep = 0.0;
        for obs in obstacles {
            let (_, dist) = nearest_point_on_obstacle(point, obs);
            if dist < self.config.influence_distance {
                u_rep += 0.5 * self.config.k_rep * (1.0 / dist - 1.0 / self.config.influence_distance).powi(2);
            }
        }
        u_att + u_rep
    }
}

fn nearest_point_on_obstacle(point: &Point2D, obstacle: &Obstacle) -> (Point2D, f64) {
    match obstacle {
        Obstacle::Circle { center, radius } => {
            let dx = point.x - center.x;
            let dy = point.y - center.y;
            let dist = (dx.powi(2) + dy.powi(2)).sqrt();
            if dist < 1e-9 {
                return (*center, *radius);
            }
            let nearest = Point2D::new(
                center.x + radius * dx / dist,
                center.y + radius * dy / dist,
            );
            (nearest, (dist - radius).max(0.0))
        }
        Obstacle::Rect { min, max } => {
            let nx = point.x.clamp(min.x, max.x);
            let ny = point.y.clamp(min.y, max.y);
            let nearest = Point2D::new(nx, ny);
            // Check if point is inside
            let inside = point.x > min.x && point.x < max.x && point.y > min.y && point.y < max.y;
            if inside {
                // Distance to nearest edge
                let d_left = point.x - min.x;
                let d_right = max.x - point.x;
                let d_bottom = point.y - min.y;
                let d_top = max.y - point.y;
                let _d = d_left.min(d_right).min(d_bottom).min(d_top);
                (nearest, 0.0)
            } else {
                (nearest, point.distance_to(&nearest))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_potential_no_obstacles() {
        let planner = PotentialFieldPlanner::new(PotentialFieldConfig::default());
        let path = planner.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            &[],
        );
        assert!(path.is_some());
        let p = path.unwrap();
        assert!((p.last().unwrap().x - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_potential_with_obstacle() {
        let planner = PotentialFieldPlanner::new(PotentialFieldConfig {
            k_rep: 50.0,
            influence_distance: 2.0,
            ..Default::default()
        });
        let obs = Obstacle::Circle { center: Point2D::new(1.5, 0.0), radius: 0.5 };
        let path = planner.plan(
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            &[obs],
        );
        if let Some(p) = path {
            // Path should not go through obstacle
            for pt in &p {
                assert!(pt.distance_to(&Point2D::new(1.5, 0.0)) > 0.4, "point inside obstacle");
            }
        }
    }

    #[test]
    fn test_attractive_force() {
        let planner = PotentialFieldPlanner::new(PotentialFieldConfig::default());
        let (fx, fy) = planner.attractive_force(&Point2D::new(0.0, 0.0), &Point2D::new(1.0, 0.0));
        assert!(fx > 0.0);
        assert!(fy.abs() < 1e-9);
    }

    #[test]
    fn test_repulsive_force() {
        let planner = PotentialFieldPlanner::new(PotentialFieldConfig::default());
        let obs = Obstacle::Circle { center: Point2D::new(0.5, 0.0), radius: 0.3 };
        let (fx, _fy) = planner.repulsive_force(&Point2D::new(0.0, 0.0), &obs);
        // Should push away from obstacle (negative x direction or positive)
        assert!(fx != 0.0);
    }

    #[test]
    fn test_potential_value() {
        let planner = PotentialFieldPlanner::new(PotentialFieldConfig::default());
        let p1 = planner.potential(&Point2D::new(0.0, 0.0), &Point2D::new(1.0, 0.0), &[]);
        let p2 = planner.potential(&Point2D::new(0.5, 0.0), &Point2D::new(1.0, 0.0), &[]);
        assert!(p2 < p1, "closer to goal should have lower potential");
    }

    #[test]
    fn test_potential_local_minimum() {
        // Concave obstacle can create local minimum
        let planner = PotentialFieldPlanner::new(PotentialFieldConfig {
            max_iterations: 100,
            ..Default::default()
        });
        // U-shaped obstacle trap
        let obs = vec![
            Obstacle::Rect { min: Point2D::new(-2.0, -0.5), max: Point2D::new(2.0, 0.0) },
        ];
        let result = planner.plan(
            Point2D::new(0.0, 1.0),
            Point2D::new(0.0, -1.0),
            &obs,
        );
        // May get stuck — potential field limitation
        // This test just verifies it doesn't panic
    }
}
