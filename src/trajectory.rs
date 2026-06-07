//! Trajectory generation and smoothing.

use crate::obstacle::Point2D;

/// Trajectory point with position and time.
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    pub position: Point2D,
    pub time: f64,
}

/// A trajectory is a sequence of timed points.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    pub points: Vec<TrajectoryPoint>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a point to the trajectory.
    pub fn add_point(&mut self, position: Point2D, time: f64) {
        self.points.push(TrajectoryPoint { position, time });
    }

    /// Total duration.
    pub fn duration(&self) -> f64 {
        if self.points.len() < 2 { return 0.0; }
        self.points.last().unwrap().time - self.points.first().unwrap().time
    }

    /// Total path length.
    pub fn length(&self) -> f64 {
        let mut len = 0.0;
        for w in self.points.windows(2) {
            len += w[0].position.distance_to(&w[1].position);
        }
        len
    }

    /// Interpolate position at given time.
    pub fn interpolate(&self, t: f64) -> Option<Point2D> {
        if self.points.is_empty() { return None; }
        if t <= self.points[0].time { return Some(self.points[0].position); }
        if t >= self.points.last().unwrap().time { return Some(self.points.last().unwrap().position); }

        for w in self.points.windows(2) {
            if t >= w[0].time && t <= w[1].time {
                let dt = w[1].time - w[0].time;
                if dt < 1e-12 { return Some(w[0].position); }
                let alpha = (t - w[0].time) / dt;
                return Some(w[0].position.lerp(&w[1].position, alpha));
            }
        }
        None
    }

    /// Compute velocity at each segment (approximate).
    pub fn velocities(&self) -> Vec<f64> {
        let mut vels = Vec::new();
        for w in self.points.windows(2) {
            let dt = w[1].time - w[0].time;
            if dt > 1e-12 {
                vels.push(w[0].position.distance_to(&w[1].position) / dt);
            }
        }
        vels
    }

    /// Resample trajectory at uniform time intervals.
    pub fn resample(&self, dt: f64) -> Trajectory {
        if self.points.is_empty() { return Trajectory::new(); }
        let t_start = self.points[0].time;
        let t_end = self.points.last().unwrap().time;
        let mut resampled = Trajectory::new();
        let mut t = t_start;
        while t <= t_end {
            if let Some(p) = self.interpolate(t) {
                resampled.add_point(p, t);
            }
            t += dt;
        }
        resampled
    }
}

/// Generate a linear trajectory between two points.
pub fn linear_trajectory(start: Point2D, end: Point2D, duration: f64, steps: usize) -> Trajectory {
    let mut traj = Trajectory::new();
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        traj.add_point(start.lerp(&end, t), t * duration);
    }
    traj
}

/// Generate a trapezoidal velocity profile trajectory.
pub fn trapezoidal_trajectory(start: Point2D, end: Point2D, max_vel: f64, accel: f64) -> Trajectory {
    let dist = start.distance_to(&end);
    if dist < 1e-9 {
        let mut traj = Trajectory::new();
        traj.add_point(start, 0.0);
        return traj;
    }
    let direction = Point2D::new(
        (end.x - start.x) / dist,
        (end.y - start.y) / dist,
    );

    let t_accel = max_vel / accel;
    let d_accel = 0.5 * accel * t_accel * t_accel;

    let t_total;
    let t_cruise;

    if 2.0 * d_accel > dist {
        // Triangle profile
        let t_half = (dist / accel).sqrt();
        t_total = 2.0 * t_half;
        t_cruise = 0.0;
    } else {
        let d_cruise = dist - 2.0 * d_accel;
        t_cruise = d_cruise / max_vel;
        t_total = 2.0 * t_accel + t_cruise;
    }

    let mut traj = Trajectory::new();
    let steps = 100;
    for i in 0..=steps {
        let t = (i as f64 / steps as f64) * t_total;
        let d = compute_distance_trapezoid(t, t_accel, t_cruise, accel, max_vel, d_accel);
        let d_clamped = d.clamp(0.0, dist);
        let pos = Point2D::new(
            start.x + direction.x * d_clamped,
            start.y + direction.y * d_clamped,
        );
        traj.add_point(pos, t);
    }
    traj
}

fn compute_distance_trapezoid(t: f64, t_accel: f64, t_cruise: f64, accel: f64, max_vel: f64, _d_accel: f64) -> f64 {
    if t <= t_accel {
        0.5 * accel * t * t
    } else if t <= t_accel + t_cruise {
        0.5 * accel * t_accel * t_accel + max_vel * (t - t_accel)
    } else {
        let t_dec = t - t_accel - t_cruise;
        0.5 * accel * t_accel * t_accel + max_vel * t_cruise + max_vel * t_dec - 0.5 * accel * t_dec * t_dec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_trajectory() {
        let traj = linear_trajectory(Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0), 5.0, 10);
        assert_eq!(traj.points.len(), 11);
        assert!((traj.duration() - 5.0).abs() < 1e-9);
        assert!((traj.length() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolation() {
        let traj = linear_trajectory(Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0), 10.0, 2);
        let mid = traj.interpolate(5.0).unwrap();
        assert!((mid.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_velocities() {
        let traj = linear_trajectory(Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0), 5.0, 5);
        let vels = traj.velocities();
        // All velocities should be 2.0 (10/5)
        for v in &vels {
            assert!((v - 2.0).abs() < 0.1, "v={}", v);
        }
    }

    #[test]
    fn test_resample() {
        let traj = linear_trajectory(Point2D::new(0.0, 0.0), Point2D::new(10.0, 0.0), 10.0, 100);
        let resampled = traj.resample(1.0);
        assert!(resampled.points.len() >= 10);
    }

    #[test]
    fn test_trapezoidal_trajectory() {
        let traj = trapezoidal_trajectory(
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            5.0,
            10.0,
        );
        assert!(traj.points.len() > 2);
        // First and last points should be near start/end
        assert!(traj.points.first().unwrap().position.x.abs() < 1.0);
        assert!((traj.points.last().unwrap().position.x - 10.0).abs() < 2.0);
    }
}
