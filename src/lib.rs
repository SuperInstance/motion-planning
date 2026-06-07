//! # motion-planning
//!
//! Motion planning algorithms for robotics: RRT, potential fields, A* in 2D.
//! Pure Rust, no external dependencies.

pub mod rrt;
pub mod potential;
pub mod astar_path;
pub mod obstacle;
pub mod trajectory;

pub use obstacle::{Obstacle, Point2D};
