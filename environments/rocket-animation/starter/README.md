# Rocket launch and failure

Build an 18-second rocket animation for a customer demo. The sequence should be
instantly readable: a staged liftoff, a brief shudder under aerodynamic load,
then a mid-air breakup whose fragments continue moving naturally. The supplied
viewer provides the artwork and playback controls; your job is the motion.

Ignition begins at **1 s**, liftoff at **2 s**, and max-Q occurs during
**5.5–6.5 s**. Let the flame build to full thrust by liftoff. Show a small angular
shake during max-Q that settles afterward. Keep the rocket close to upright
throughout powered flight.
At **8 s**, the vehicle breaks apart and the engine shuts off. Hide the intact
rocket at that point and continue the debris motion through the end of the clip.

The rocket starts upright with its center at **(0, 10) m**. Just before breakup,
it should be **60–100 m** above ground and moving upward at **20–35 m/s**.
Use **9.81 m/s²** for gravity. Include some sideways motion so the fragments'
inherited direction is easy to see.

Give the powered flight a visible exhaust plume and a smoke trail. The nozzle is
8 m below the rocket's center. Smoke should emerge there, drift, expand, and fade
smoothly. Stop creating smoke at breakup, but let existing puffs finish fading;
remove a puff only once it has fully faded. The remaining trail should be faded
out by the end of the clip.

Break the vehicle into at least four recognizable pieces, representing its
**nose, body, fins, and engine**. Start them where those parts belong on the
rocket. They should inherit its upward and sideways velocity, receive modest
individual kicks, spread apart, tumble, and fall under gravity. Keep their
identities and dimensions consistent. Pieces that reach the ground should make
sensible contact and settle there rather than vanish or teleport.

Keep the main action on screen. The rocket's useful flight area is approximately
x = −100…100 m and y = 2…130 m; the wider visible scene is approximately
x = −110…145 m and y = 0…140 m. Coordinates are meters, with x pointing right,
y pointing up, and ground at y = 0. Angles are clockwise radians from upright. The viewer
renders a fixed 16-meter rocket centered on the supplied position.

## Deliverable

Implement the following export in `/workspace/scene.js`:

```js
export function sceneAt(t) {
  return {
    rocket: {
      x: 0,
      y: 10,
      angle: 0,
      engine: 0,
      visible: true,
    },
    smoke: [],
    debris: [],
  };
}
```

`t` is elapsed time in seconds, from 0 to 18. Make this a pure function: the same
time must always return the same scene, including when playback seeks backward
or jumps between times. Use deterministic variation rather than wall-clock time
or unseeded randomness.

- `rocket` has numeric `x`, `y`, and `angle`, an `engine` intensity from 0 to 1,
  and boolean `visible`.
- Each smoke puff has `{id, x, y, radius, opacity}`. Radius is in meters and
  opacity is from 0 to 1.
- Each debris piece has `{id, x, y, angle, width, height, kind}`. Dimensions are
  in meters. Kind is `nose`, `body`, `fin`, or `engine`.
- IDs are nonempty strings, unique within each array, and stable for the lifetime
  of a puff or piece. Return no more than 400 smoke puffs and 60 debris pieces
  at once; fully faded smoke can be removed.

A debris piece's position is the center of its width-by-height bounds before
rotation. Body and engine pieces are rectangles. The nose is a triangle with its
tip at the top center and its base along the bottom. A fin is a triangle joining
the bottom-left, top-right, and bottom-right corners. Use those shapes when
placing pieces on the ground.

## Viewer constraints

Submit a regular, nonsymlink UTF-8 `scene.js` file of at most **128000 bytes**.
The module runs in a Web Worker without DOM access, external imports, or network
access. It must load within **4 seconds**, and each `sceneAt(t)` call must finish
within **2 seconds**. Return finite numbers: coordinates and angles must stay
within −1000000…1000000, and radii and dimensions within 0.000001…1000 m.

Use `preview` to inspect the animation at several points, especially around
liftoff and breakup. After your final edit, call `save_replay` and inspect its
returned frame sheet. This saves `rocket-animation.webp` for playback in the
trace Workspace, `rocket-preview.html` for interactive playback after download,
and `rocket-sequence.png` as a frame sheet. The animated WebP must fit the
**5 MiB (5242880-byte)** viewer attachment limit.
