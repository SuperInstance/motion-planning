//! A* path planning on a 2D grid.

use crate::obstacle::{Obstacle, Point2D};

/// Grid cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    Free,
    Blocked,
}

/// 2D grid for A* planning.
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin: Point2D,
    cells: Vec<Cell>,
}

impl Grid {
    /// Create a new grid.
    pub fn new(width: usize, height: usize, resolution: f64, origin: Point2D) -> Self {
        Self {
            width,
            height,
            resolution,
            origin,
            cells: vec![Cell::Free; width * height],
        }
    }

    /// Get cell at grid coordinates.
    pub fn get(&self, x: usize, y: usize) -> Cell {
        self.cells[y * self.width + x]
    }

    /// Set cell at grid coordinates.
    pub fn set(&mut self, x: usize, y: usize, cell: Cell) {
        if x < self.width && y < self.height {
            self.cells[y * self.width + x] = cell;
        }
    }

    /// Convert world coordinates to grid coordinates.
    pub fn world_to_grid(&self, p: &Point2D) -> (usize, usize) {
        let gx = ((p.x - self.origin.x) / self.resolution) as usize;
        let gy = ((p.y - self.origin.y) / self.resolution) as usize;
        (gx.min(self.width - 1), gy.min(self.height - 1))
    }

    /// Convert grid coordinates to world coordinates.
    pub fn grid_to_world(&self, gx: usize, gy: usize) -> Point2D {
        Point2D::new(
            self.origin.x + (gx as f64 + 0.5) * self.resolution,
            self.origin.y + (gy as f64 + 0.5) * self.resolution,
        )
    }

    /// Add obstacles to the grid.
    pub fn add_obstacles(&mut self, obstacles: &[Obstacle]) {
        for y in 0..self.height {
            for x in 0..self.width {
                let world = self.grid_to_world(x, y);
                for obs in obstacles {
                    if obs.contains(&world) {
                        self.set(x, y, Cell::Blocked);
                    }
                }
            }
        }
    }

    /// Get 4-connected neighbors.
    pub fn neighbors4(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut n = Vec::new();
        if x > 0 && self.get(x - 1, y) == Cell::Free { n.push((x - 1, y)); }
        if x + 1 < self.width && self.get(x + 1, y) == Cell::Free { n.push((x + 1, y)); }
        if y > 0 && self.get(x, y - 1) == Cell::Free { n.push((x, y - 1)); }
        if y + 1 < self.height && self.get(x, y + 1) == Cell::Free { n.push((x, y + 1)); }
        n
    }

    /// Get 8-connected neighbors.
    pub fn neighbors8(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut n = self.neighbors4(x, y);
        if x > 0 && y > 0 && self.get(x - 1, y - 1) == Cell::Free { n.push((x - 1, y - 1)); }
        if x + 1 < self.width && y > 0 && self.get(x + 1, y - 1) == Cell::Free { n.push((x + 1, y - 1)); }
        if x > 0 && y + 1 < self.height && self.get(x - 1, y + 1) == Cell::Free { n.push((x - 1, y + 1)); }
        if x + 1 < self.width && y + 1 < self.height && self.get(x + 1, y + 1) == Cell::Free { n.push((x + 1, y + 1)); }
        n
    }
}

/// A* planner.
pub struct AStarPlanner {
    pub allow_diagonal: bool,
}

impl AStarPlanner {
    pub fn new(allow_diagonal: bool) -> Self {
        Self { allow_diagonal }
    }

    /// Find path on grid from start to goal (in grid coordinates).
    pub fn plan(&self, grid: &Grid, start: (usize, usize), goal: (usize, usize)) -> Option<Vec<(usize, usize)>> {
        if grid.get(start.0, start.1) == Cell::Blocked || grid.get(goal.0, goal.1) == Cell::Blocked {
            return None;
        }

        let total = grid.width * grid.height;
        let mut g_score = vec![f64::INFINITY; total];
        let mut f_score = vec![f64::INFINITY; total];
        let mut came_from = vec![None; total];
        let mut closed = vec![false; total];

        let idx = |x: usize, y: usize| y * grid.width + x;
        let si = idx(start.0, start.1);
        g_score[si] = 0.0;
        f_score[si] = heuristic(start, goal);

        // Open set as simple vector (not optimal but correct)
        let mut open = vec![si];

        while let Some(&current) = open.iter().min_by(|&&a, &&b| f_score[a].partial_cmp(&f_score[b]).unwrap()) {
            let cx = current % grid.width;
            let cy = current / grid.width;

            if cx == goal.0 && cy == goal.1 {
                return Some(reconstruct(&came_from, current, grid.width));
            }

            open.retain(|&x| x != current);
            closed[current] = true;

            let neighbors = if self.allow_diagonal {
                grid.neighbors8(cx, cy)
            } else {
                grid.neighbors4(cx, cy)
            };

            for (nx, ny) in neighbors {
                let ni = idx(nx, ny);
                if closed[ni] { continue; }

                let move_cost = if nx != cx && ny != cy { std::f64::consts::SQRT_2 } else { 1.0 };
                let tentative = g_score[current] + move_cost;

                if tentative < g_score[ni] {
                    came_from[ni] = Some(current);
                    g_score[ni] = tentative;
                    f_score[ni] = tentative + heuristic((nx, ny), goal);
                    if !open.contains(&ni) {
                        open.push(ni);
                    }
                }
            }
        }
        None
    }

    /// Plan and return world coordinates.
    pub fn plan_world(&self, grid: &Grid, start: &Point2D, goal: &Point2D) -> Option<Vec<Point2D>> {
        let sg = grid.world_to_grid(start);
        let gg = grid.world_to_grid(goal);
        self.plan(grid, sg, gg).map(|path| {
            path.iter().map(|&(x, y)| grid.grid_to_world(x, y)).collect()
        })
    }
}

fn heuristic(a: (usize, usize), b: (usize, usize)) -> f64 {
    let dx = (a.0 as f64 - b.0 as f64).abs();
    let dy = (a.1 as f64 - b.1 as f64).abs();
    dx.max(dy) + (std::f64::consts::SQRT_2 - 1.0) * dx.min(dy) // Octile distance
}

fn reconstruct(came_from: &[Option<usize>], current: usize, width: usize) -> Vec<(usize, usize)> {
    let mut path = vec![(current % width, current / width)];
    let mut c = current;
    while let Some(prev) = came_from[c] {
        path.push((prev % width, prev / width));
        c = prev;
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astar_empty_grid() {
        let grid = Grid::new(10, 10, 1.0, Point2D::new(0.0, 0.0));
        let planner = AStarPlanner::new(false);
        let path = planner.plan(&grid, (0, 0), (9, 9));
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p[0], (0, 0));
        assert_eq!(*p.last().unwrap(), (9, 9));
    }

    #[test]
    fn test_astar_with_wall() {
        let mut grid = Grid::new(10, 10, 1.0, Point2D::new(0.0, 0.0));
        // Horizontal wall
        for x in 0..9 {
            grid.set(x, 5, Cell::Blocked);
        }
        let planner = AStarPlanner::new(false);
        let path = planner.plan(&grid, (0, 0), (9, 9));
        assert!(path.is_some());
        // Path should go around wall
        let p = path.unwrap();
        for &(x, y) in &p {
            assert!(grid.get(x, y) == Cell::Free);
        }
    }

    #[test]
    fn test_astar_blocked_goal() {
        let mut grid = Grid::new(10, 10, 1.0, Point2D::new(0.0, 0.0));
        grid.set(9, 9, Cell::Blocked);
        let planner = AStarPlanner::new(false);
        let path = planner.plan(&grid, (0, 0), (9, 9));
        assert!(path.is_none());
    }

    #[test]
    fn test_astar_diagonal_shorter() {
        let grid = Grid::new(10, 10, 1.0, Point2D::new(0.0, 0.0));
        let planner4 = AStarPlanner::new(false);
        let planner8 = AStarPlanner::new(true);
        let p4 = planner4.plan(&grid, (0, 0), (5, 5)).unwrap();
        let p8 = planner8.plan(&grid, (0, 0), (5, 5)).unwrap();
        // Diagonal should be shorter
        assert!(p8.len() <= p4.len());
    }

    #[test]
    fn test_grid_obstacle_addition() {
        let mut grid = Grid::new(20, 20, 0.5, Point2D::new(0.0, 0.0));
        let obs = Obstacle::Circle { center: Point2D::new(5.0, 5.0), radius: 2.0 };
        grid.add_obstacles(&[obs]);
        // Center should be blocked
        let (cx, cy) = grid.world_to_grid(&Point2D::new(5.0, 5.0));
        assert_eq!(grid.get(cx, cy), Cell::Blocked);
    }

    #[test]
    fn test_astar_world_coordinates() {
        let mut grid = Grid::new(10, 10, 1.0, Point2D::new(0.0, 0.0));
        let planner = AStarPlanner::new(false);
        let path = planner.plan_world(&grid, &Point2D::new(0.5, 0.5), &Point2D::new(9.5, 9.5));
        assert!(path.is_some());
    }
}
