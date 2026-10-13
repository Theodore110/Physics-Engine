physics2d interactive frontend setup

1. Copy interactive.rs into your project at:
   src/bin/interactive.rs

2. In Cargo.toml, change the dependencies section from:
   [dependencies]

   to:
   [dependencies]
   macroquad = "0.4"

   Keep your existing [features] and [profile.*] sections unchanged.

3. From the project root, run:
   cargo run --release --bin interactive

First build may take longer because Cargo must download and compile Macroquad. You need an internet connection for that initial dependency download.

Controls:
- Click Pause / press Space: pause or resume
- Click Reset / press R: reset scene
- Click Add circle / press C: spawn a circle
- Click Add box / press B: spawn a box
- Click Contacts / press H: show/hide contact points
- Left-click and drag a dynamic body to move it; release to give it a throw velocity
- Mouse wheel: zoom
- Middle mouse button: pan
- Escape: stop selecting a body

This adds a graphics dependency for the frontend only; the physics math and solver remain in your physics2d crate.
