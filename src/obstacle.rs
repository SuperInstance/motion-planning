//! 2D point and obstacle types.

/// 2D point.
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn lerp(&self, other: &Point2D, t: f64) -> Point2D {
        Point2D {
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
        }
    }
}

/// Obstacle types for 2D planning.
#[derive(Debug, Clone)]
pub enum Obstacle {
    /// Circle obstacle (center, radius)
    Circle { center: Point2D, radius: f64 },
    /// Rectangle obstacle (min corner, max corner)
    Rect { min: Point2D, max: Point2D },
}

impl Obstacle {
    /// Check if a point is inside the obstacle.
    pub fn contains(&self, p: &Point2D) -> bool {
        match self {
            Obstacle::Circle { center, radius } => {
                center.distance_to(p) < *radius
            }
            Obstacle::Rect { min, max } => {
                p.x > min.x && p.x < max.x && p.y > min.y && p.y < max.y
            }
        }
    }

    /// Check if a line segment intersects the obstacle.
    pub fn intersects_segment(&self, a: &Point2D, b: &Point2D) -> bool {
        // Sample along segment
        let steps = 20;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let p = a.lerp(b, t);
            if self.contains(&p) {
                return true;
            }
        }
        false
    }
}

/// Check if a path (sequence of segments) is collision-free.
pub fn path_collision_free(path: &[Point2D], obstacles: &[Obstacle]) -> bool {
    for w in path.windows(2) {
        for obs in obstacles {
            if obs.intersects_segment(&w[0], &w[1]) {
                return false;
            }
        }
    }
    true
}
