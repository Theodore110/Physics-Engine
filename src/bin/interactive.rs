//! Interactive graphical frontend for the physics2d engine.
//! Run with: cargo run --release --bin interactive

use macroquad::prelude::*;
use physics2d::body::RigidBody;
use physics2d::scene;
use physics2d::{Shape, Vec2, World, WorldConfig};

const BG: Color = Color::new(0.055, 0.067, 0.09, 1.0);
const PANEL: Color = Color::new(0.10, 0.12, 0.16, 0.97);
const ACCENT: Color = Color::new(0.30, 0.72, 0.98, 1.0);
const WORLD_SCALE: f32 = 42.0; // pixels per physics unit at zoom 1

struct App {
    world: World,
    paused: bool,
    show_contacts: bool,
    camera_target: Vec2,
    zoom: f32,
    selected: Option<usize>,
    drag_offset: Vec2,
    previous_mouse_world: Vec2,
    fps_smooth: f32,
    previous_mouse: Vec2,
}

impl App {
    fn new() -> Self {
        let mut app = Self {
            world: make_world(),
            paused: false,
            show_contacts: true,
            camera_target: Vec2::new(0.0, 3.5),
            zoom: 1.0,
            selected: None,
            drag_offset: Vec2::ZERO,
            previous_mouse: Vec2::new(0.0, 0.0),
            previous_mouse_world: Vec2::ZERO,
            fps_smooth: 60.0,
        };
        app.world.fit_cell_size();
        app
    }

    fn reset(&mut self) {
        self.world = make_world();
        self.world.fit_cell_size();
        self.selected = None;
        self.paused = false;
        self.camera_target = Vec2::new(0.0, 3.5);
        self.zoom = 1.0;
    }

    fn camera(&self) -> Camera2D {
        let s = WORLD_SCALE * self.zoom;
        Camera2D {
            target: mq_vec(self.camera_target),
            zoom: vec2(2.0 / (screen_width() / s), -2.0 / (screen_height() / s)),
            ..Default::default()
        }
    }

    fn mouse_world(&self) -> Vec2 {
        let p = self.camera().screen_to_world(mouse_position().into());
        Vec2::new(p.x, p.y)
    }

    fn update_input(&mut self) {
        let mouse = mouse_position();
        let world_mouse = self.mouse_world();

        // Toolbar clicks are handled in screen coordinates, outside the camera.
        if is_mouse_button_pressed(MouseButton::Left) && mouse.1 < 58.0 {
            if mouse.0 < 105.0 { self.paused = !self.paused; }
            else if mouse.0 < 205.0 { self.reset(); }
            else if mouse.0 < 320.0 { self.spawn_circle(); }
            else if mouse.0 < 430.0 { self.spawn_box(); }
            else if mouse.0 < 570.0 { self.show_contacts = !self.show_contacts; }
            return;
        }

        if is_key_pressed(KeyCode::Space) { self.paused = !self.paused; }
        if is_key_pressed(KeyCode::R) { self.reset(); }
        if is_key_pressed(KeyCode::C) { self.spawn_circle(); }
        if is_key_pressed(KeyCode::B) { self.spawn_box(); }
        if is_key_pressed(KeyCode::H) { self.show_contacts = !self.show_contacts; }
        if is_key_pressed(KeyCode::Escape) { self.selected = None; }

        let wheel = mouse_wheel().1;
        if wheel != 0.0 {
            self.zoom = (self.zoom * (1.0 + wheel * 0.08)).clamp(0.35, 2.8);
        }

        // Middle mouse pans the camera.
        if is_mouse_button_down(MouseButton::Middle) {
            let (mouse_x, mouse_y) = mouse_position();
            let current_mouse = Vec2::new(mouse_x, mouse_y);
            let delta = current_mouse - self.previous_mouse;
            self.previous_mouse = current_mouse;
        }

        if is_mouse_button_pressed(MouseButton::Left) {
            if let Some(i) = self.pick_body(world_mouse) {
                if self.world.bodies.inv_mass[i] > 0.0 {
                    self.selected = Some(i);
                    self.drag_offset = Vec2::new(
                        self.world.bodies.pos_x[i] - world_mouse.x,
                        self.world.bodies.pos_y[i] - world_mouse.y,
                    );
                    self.previous_mouse_world = world_mouse;
                    self.world.bodies.set_velocity(physics2d::BodyId(i as u32), Vec2::ZERO);
                }
            }
        }

        if let Some(i) = self.selected {
            if is_mouse_button_down(MouseButton::Left) {
                let target = world_mouse + self.drag_offset;
                let id = physics2d::BodyId(i as u32);
                self.world.bodies.set_position(id, target);
                self.world.bodies.set_velocity(id, Vec2::ZERO);
                self.previous_mouse_world = world_mouse;
            }
            if is_mouse_button_released(MouseButton::Left) {
                let velocity = (world_mouse - self.previous_mouse_world) * 12.0;
                self.world.bodies.set_velocity(physics2d::BodyId(i as u32), velocity);
                self.selected = None;
            }
        }
    }

    fn pick_body(&self, p: Vec2) -> Option<usize> {
        // Reverse order makes the visually topmost body win.
        for i in (0..self.world.bodies.len()).rev() {
            let b = &self.world.bodies;
            let dx = p.x - b.pos_x[i];
            let dy = p.y - b.pos_y[i];
            match b.shapes[i] {
                Shape::Circle { radius } => {
                    if dx * dx + dy * dy <= radius * radius { return Some(i); }
                }
                Shape::Box { half_extents, rotation, .. } => {
                    // Transform mouse point into the box's local coordinates.
                    let c = rotation.cos();
                    let s = rotation.sin();
                    let lx = dx * c + dy * s;
                    let ly = -dx * s + dy * c;
                    if lx.abs() <= half_extents.x && ly.abs() <= half_extents.y { return Some(i); }
                }
            }
        }
        None
    }

    fn spawn_circle(&mut self) {
        let p = Vec2::new(self.camera_target.x, self.camera_target.y + 4.0);
        self.world.add_body(RigidBody::dynamic(Shape::circle(0.38), 1.0, p));
        self.world.fit_cell_size();
    }

    fn spawn_box(&mut self) {
        let p = Vec2::new(self.camera_target.x, self.camera_target.y + 4.0);
        let rotation = macroquad::rand::gen_range(-0.35f32, 0.35f32);
        self.world.add_body(RigidBody::dynamic(Shape::oriented_box(Vec2::new(0.48, 0.34), rotation), 1.0, p));
        self.world.fit_cell_size();
    }

    fn draw_world(&self) {
        clear_background(BG);
        set_camera(&self.camera());

        // Subtle world grid.
        let left = self.camera_target.x - screen_width() / (2.0 * WORLD_SCALE * self.zoom);
        let right = self.camera_target.x + screen_width() / (2.0 * WORLD_SCALE * self.zoom);
        let bottom = self.camera_target.y - screen_height() / (2.0 * WORLD_SCALE * self.zoom);
        let top = self.camera_target.y + screen_height() / (2.0 * WORLD_SCALE * self.zoom);
        let grid_start_x = left.floor() as i32;
        let grid_end_x = right.ceil() as i32;
        let grid_start_y = bottom.floor() as i32;
        let grid_end_y = top.ceil() as i32;
        for x in grid_start_x..=grid_end_x {
            draw_line(x as f32, bottom, x as f32, top, 0.012, Color::new(0.20, 0.24, 0.30, 0.45));
        }
        for y in grid_start_y..=grid_end_y {
            draw_line(left, y as f32, right, y as f32, 0.012, Color::new(0.20, 0.24, 0.30, 0.45));
        }

        let b = &self.world.bodies;
        for i in 0..b.len() {
            let center = Vec2::new(b.pos_x[i], b.pos_y[i]);
            let is_static = b.inv_mass[i] == 0.0;
            let selected = self.selected == Some(i);
            let fill = if selected { Color::new(1.0, 0.72, 0.28, 1.0) }
                else if is_static { Color::new(0.34, 0.40, 0.49, 1.0) }
                else if i % 3 == 0 { Color::new(0.25, 0.69, 0.92, 1.0) }
                else if i % 3 == 1 { Color::new(0.42, 0.78, 0.62, 1.0) }
                else { Color::new(0.73, 0.54, 0.94, 1.0) };
            match b.shapes[i] {
                Shape::Circle { radius } => {
                    draw_circle(center.x, center.y, radius, fill);
                    draw_circle_lines(center.x, center.y, radius, 0.035, Color::new(0.88, 0.93, 0.98, 0.95));
                    draw_line(center.x, center.y, center.x + radius * 0.72, center.y, 0.025, Color::new(0.05, 0.08, 0.12, 0.8));
                }
                Shape::Box { half_extents, rotation, .. } => {
                    let corners = [
                        Vec2::new(-half_extents.x, -half_extents.y),
                        Vec2::new(half_extents.x, -half_extents.y),
                        Vec2::new(half_extents.x, half_extents.y),
                        Vec2::new(-half_extents.x, half_extents.y),
                    ];
                    let c = rotation.cos();
                    let s = rotation.sin();
                    let pts: Vec<Vec2> = corners.iter().map(|v| Vec2::new(center.x + v.x*c - v.y*s, center.y + v.x*s + v.y*c)).collect();
                    draw_triangle(mq_vec(pts[0]), mq_vec(pts[1]), mq_vec(pts[2]), fill);
                    draw_triangle(mq_vec(pts[0]), mq_vec(pts[2]), mq_vec(pts[3]), fill);
                    for k in 0..4 {
                        let a = pts[k]; let z = pts[(k + 1) % 4];
                        draw_line(a.x, a.y, z.x, z.y, 0.035, Color::new(0.90, 0.94, 0.99, 0.95));
                    }
                }
            }
        }

        if self.show_contacts {
            for contact in self.world.contacts() {
                draw_circle(
                    contact.contact_point.x,
                    contact.contact_point.y,
                    0.075,
                    Color::new(1.0, 0.30, 0.26, 1.0),
                );
            }
        }
        set_default_camera();
    }

    fn draw_ui(&self) {
        draw_rectangle(0.0, 0.0, screen_width(), 58.0, PANEL);
        draw_line(0.0, 58.0, screen_width(), 58.0, 1.0, Color::new(0.25, 0.30, 0.38, 1.0));
        button(12.0, 12.0, 88.0, if self.paused { "Resume" } else { "Pause" }, ACCENT);
        button(110.0, 12.0, 88.0, "Reset [R]", Color::new(0.30, 0.34, 0.42, 1.0));
        button(208.0, 12.0, 102.0, "Add circle [C]", Color::new(0.20, 0.48, 0.63, 1.0));
        button(320.0, 12.0, 102.0, "Add box [B]", Color::new(0.25, 0.52, 0.43, 1.0));
        button(432.0, 12.0, 130.0, if self.show_contacts { "Contacts: ON [H]" } else { "Contacts: OFF [H]" }, Color::new(0.38, 0.32, 0.50, 1.0));

        let b = &self.world.bodies;
        let dynamic = b.inv_mass.iter().filter(|&&m| m > 0.0).count();
        let status = format!("Bodies: {} ({} dynamic)   Pairs: {}   Contacts: {}   FPS: {:.0}   Zoom: {:.1}x",
            b.len(), dynamic, self.world.last_pair_count(), self.world.contacts().len(), self.fps_smooth, self.zoom);
        draw_text(&status, 18.0, screen_height() - 18.0, 19.0, Color::new(0.85, 0.89, 0.95, 1.0));
        draw_text("Drag bodies to move/throw  |  Wheel: zoom  |  Middle mouse: pan  |  Space: pause",
            18.0, screen_height() - 42.0, 17.0, Color::new(0.62, 0.69, 0.79, 1.0));
    }
}

fn button(x: f32, y: f32, w: f32, label: &str, color: Color) {
    draw_rectangle(x, y, w, 34.0, color);
    draw_rectangle_lines(x, y, w, 34.0, 1.0, Color::new(0.80, 0.88, 0.96, 0.4));
    let dims = measure_text(label, None, 16, 1.0);
    draw_text(label, x + (w - dims.width) * 0.5, y + 22.0, 16.0, WHITE);
}

fn make_world() -> World {
    let mut config = WorldConfig::default();
    config.collect_timings = false;
    let mut world = World::new(config);

    // Fixed container: floor and side walls. The engine uses fixed-orientation boxes.
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(10.5, 0.35)), Vec2::new(0.0, -0.35)));
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(0.35, 7.0)), Vec2::new(-10.15, 6.4)));
    world.add_body(RigidBody::fixed(Shape::aligned_box(Vec2::new(0.35, 7.0)), Vec2::new(10.15, 6.4)));

    // Use the project's established scene helper, then add a few larger boxes.
    scene::circle_pile(&mut world, 65, 2024);
    let mut rng = physics2d::util::Rng::new(7);
    for i in 0..5 {
        let x = rng.range(-2.5, 2.5);
        let angle = rng.range(-0.45, 0.45);
        let shape = Shape::oriented_box(Vec2::new(0.58, 0.38), angle);
        world.add_body(RigidBody::dynamic(shape, 2.0, Vec2::new(x, 7.5 + i as f32 * 1.0)).with_restitution(0.15));
    }
    world.fit_cell_size();
    world
}

fn mq_vec(v: Vec2) -> macroquad::prelude::Vec2 { vec2(v.x, v.y) }

#[macroquad::main("physics2d — Interactive Sandbox")]
async fn main() {
    request_new_screen_size(1280.0, 800.0);
    let mut app = App::new();
    loop {
        let dt = get_frame_time().min(0.05);
        app.fps_smooth = app.fps_smooth * 0.9 + get_fps() as f32 * 0.1;
        app.update_input();
        if !app.paused && app.selected.is_none() {
            app.world.advance(dt);
        }
        app.draw_world();
        app.draw_ui();
        next_frame().await;
    }
}
