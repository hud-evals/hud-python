# HUD Robot Spec v0 — authoring guide

How to **completely specify** a robot environment (an embodiment) and a robot model
(a policy) as JSON, so the two can be matched in `.initialize()`. This document is
written to let an AI agent **zero-shot generate a spec** for a new robot/model from
the web, papers, code, model cards, and URDF/MJCF — without seeing an example first.

The format is kept close in spirit to the LeRobot dataset schema (`info.json` /
`stats.json`): per-feature `dtype`, `shape`, `names`, `stats`, plus a `robot_type` and
a control rate. We extend it with the semantic layer needed for matching
(`state_type`, `state_representation`, `frame`, `order`, `units`, `limits`).

**v0 scope (this document).** A contract describes **one embodiment**, with **one
observation space and one action space** — no per-embodiment *decision variables* and
no multi-mode wrappers. A model that targets several embodiments (or exposes several
action/observation forms) is written as **separate contracts, one per form**. The
older multi-mode / decision-variable schema is preserved as documentation only under
`demos/contracts/experiments/spec_old.md`; the matcher does **not** load it.

**Rank ≥ 1 (law).** Every feature is at least 1-D: `shape` is a non-empty list. A
scalar feature uses `shape: [1]`, never `[]`. The `robot` wire codec promotes 0-D
arrays to 1-D and LeRobot dataset columns are always ≥ 1-D, so declaring `[1]` keeps
env, wire, and dataset consistent.

---

## 1. Two artifacts, one shape

There are two kinds of spec, and **they use the same feature schema** so they can be
compared field-for-field:

- **Environment / embodiment contract** (`envs/*.json`) — what the robot **emits**
(observations) and how it **expects to be acted on** (actions).
- **Model / policy contract** (`models/*.json`) — what the policy **consumes**
(observations) and what it **emits** (actions).

Matching reconciles the two: cameras by role, vectors by `state_type` + `order` +
`names`, geometry by `state_representation` + `frame`, scale by `normalization` +
`stats`, timing by control rate + `chunk_size`.

---

## 2. Top-level structure

### Environment contract


| Key            | Type     | Notes                                                  |
| -------------- | -------- | ------------------------------------------------------ |
| `robot_type`   | string   | Canonical embodiment id, e.g. `"franka_panda_libero"`. |
| `robot_class`  | string   | Coarse morphology class (see §3.9).                    |
| `control_rate` | int (Hz) | Rate the env consumes actions / emits observations.    |
| `features`     | object   | Observation + action features (see §4).                |
| `comment`      | string   | Concise notes; flag uncertainties with `OPEN:`.        |


### Model contract


| Key                    | Type          | Notes                                                                                                                                                                                                        |
| ---------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model`                | string        | Model id.                                                                                                                                                                                                    |
| `policy_class`         | string        | Implementation class, e.g. `"PI05Policy"`.                                                                                                                                                                   |
| `checkpoint`           | string        | Default weights id/link.                                                                                                                                                                                     |
| `robot_type`           | string        | The single embodiment this contract targets — the **sole** declaration of what the model supports. Matching gates on it. (Multi-embodiment checkpoints get one contract per embodiment.)                     |
| `robot_class`          | string        | Coarse morphology class (see §3.9).                                                                                                                                                                          |
| `chunk_size`           | int           | Action-horizon: how many steps the policy emits per inference.                                                                                                                                               |
| `control_rate`         | int (Hz)      | Rate the policy was trained/biased to.                                                                                                                                                                       |
| `features`             | object        | Observation features + the action.                                                                                                                                                                           |
| `comment`              | string        | Concise notes.                                                                                                                                                                                               |


---

## 3. Closed symbol sets

These are the controlled vocabularies. Prefer a value from the set; if nothing fits,
add a `comment` explaining and flag it `OPEN:`.

### 3.1 `role`

`observation` · `action`

### 3.2 Feature kinds (by key prefix)

- `observation.images.<name>` — visual stream
- `observation.text` — language / conditioning
- `observation.state.<name>` — proprioceptive vector
- `action.<name>` — action vector
- `observation.<other>` — audio, force/torque sensor, etc. (open-ended)

### 3.3 `dtype`

`uint8` (default camera), `uint16` (depth), `float16`, `float32`, `float64`,
`int32`, `int64`, `string` (text).

### 3.4 Image `type` (color space)

`rgb` · `bgr` · `gray` · `depth`

### 3.5 Image layout → `state_representation`

`HWC` · `CHW` · `THWC` (video) · `TCHW` (video).
**No batched layouts** — the batch dimension is implicit and always first; specs
describe a single sample.

### 3.6 `state_type` = `SPACE_REF_QUANTITY`

Uppercase, underscore-joined, three slots:


| Slot         | Set                                                                | Meaning                                                           |
| ------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------- |
| **SPACE**    | `JOINT`, `GRIPPER`, `EE`, `BASE`                                  | per-actuator DOFs · gripper aperture · end-effector/cartesian · mobile/floating base |
| **REF**      | `ABS`, `DEL`                                                       | absolute · delta                                                  |
| **QUANTITY** | `POS`, `POSE`, `ROT`, `VEL`, `ROTVEL`, `TWIST`, `EFF`, `PD`, `ACC` | see below                                                         |


Quantities pair 0th-order with 1st-order:


|              | Translation | Orientation | Combined (6-DoF) |
| ------------ | ----------- | ----------- | ---------------- |
| **position** | `POS`       | `ROT`       | `POSE`           |
| **velocity** | `VEL`       | `ROTVEL`    | `TWIST`          |


Plus `EFF` (force/torque/effort, unified), `PD` (PD/impedance target), `ACC`
(acceleration). Examples: `EE_ABS_POS`, `EE_DEL_ROT`, `JOINT_ABS_POS`,
`GRIPPER_ABS_POS`, `EE_ABS_TWIST`, `BASE_DEL_POSE`.

**`GRIPPER`** is the parallel-jaw end-effector aperture as a first-class space
(almost always `GRIPPER_ABS_POS`). Keep the gripper out of `JOINT` so its
`state_type` token never collides with an arm joint — a shared `JOINT_ABS_POS` token
pollutes the action signature used for matching/filtering (e.g. an EE-space arm with
a gripper would otherwise read as if it had a joint-space component). A raw
multi-joint `qpos` vector that already bundles finger joints with the arm stays one
`JOINT_*` feature; dexterous multi-DoF hands also stay `JOINT`. The gripper carries
no `frame`.

### 3.7 `state_representation`

How the numbers encode geometry. Pick by quantity:


| Quantity                         | Allowed representations                                                            |
| -------------------------------- | ---------------------------------------------------------------------------------- |
| `POS`                            | `XYZ` (cartesian) · `REAL` (joint scalars)                                         |
| `ROT`                            | `EULXYZ`, `EULZYX`, `QUATWXYZ`, `QUATXYZW`, `AXISANGLE`, `SO3`, `ROT6D`            |
| `POSE`                           | composite `<trans>_<rot>`: `XYZ_EULXYZ`, `XYZ_QUATWXYZ`, `XYZ_AXISANGLE`, …        |
| `VEL`                            | `XYZRATE` (cartesian) · `REAL` (joint)                                             |
| `ROTVEL`                         | `OMEGAXYZ`, `EULXYZRATE`, `EULZYXRATE`                                             |
| `TWIST`                          | composite `<linvel>_<angvel>`: `XYZRATE_OMEGAXYZ` (standard), `XYZRATE_EULXYZRATE` |
| `EFF` / `PD` / `ACC`             | `REAL` (joint) · `XYZ`-style (cartesian)                                           |
| gripper (under `GRIPPER`)        | `BINARY` (open/closed), `NORM01` ([0,1]), `NORM11` ([-1,1]), `REAL` (width m / finger rad) |
| any plain scalar / dimensionless | `REAL`                                                                             |


`REAL` replaces a "none" value: use it for joint scalars and any 1-D real number.

### 3.8 `frame`

`base` · `world` · `camera` · `eef` (tool). **Only on `EE`/cartesian features.**
May differ per sub-feature (e.g. OSC: translation in `base`, rotation delta vs
current `eef`).

### 3.9 `robot_class` (`armNgM` scheme)

Concise, structure-embedded names:
`arm6g1`, `arm7g1` (N-DoF arm + M gripper DoF), `bimanual6g1`, `bimanual7g1`,
`humanoid`, `quadruped`, `mobile_manip`, `unclassed`. Use `"multi"` for a
multi-embodiment model and list the embodiments in `robot_type`.

### 3.10 `units`

Combinations of `rad`, `deg`, `m`, `s`, `N`; `none` for dimensionless / normalized.

### 3.11 `normalization` (model side only)

`identity`, `min_max`, `mean_std`, `quantile`. May be a per-field object, e.g.
`{"default": "identity", "gripper.open_close": "min_max"}`. **Envs do not carry
`normalization`** — they declare raw `dtype` + `stats`.

### 3.12 Other per-feature keys

- `shape` — per-sample shape (no batch dim), e.g. `[3]`, `[256, 256, 3]`. **Rank ≥
1 (law):** always a non-empty list; a scalar is `[1]`, never `[]`.
- `order` — inclusive index range of this feature within the role-concatenated
vector, e.g. `"0-2"`, `"6"`. Lets split groups reassemble.
- `names` — element-level names (producer's own; see §6).
- `stats` — `mean`/`std`/`min`/`max` (distribution; for images nested per channel).
- `limits` — hard `[min, max]` per element (joint/clip bounds). **Distinct from
`stats`** (which is the observed distribution); add where known.
- `kp` / `kd` — impedance/PD gains (scalar or per-dim); on OSC cartesian or PD joint
actions. Recorded on **both** env and model (model is biased to its training gains).
- `padding` — `true` for synthetic pad slots (not a real input; ignored in matching).
- `chunk_size` — top-level model field (action horizon).

---

## 4. The feature object

Every entry in `features` shares a base shape; fields depend on the kind.

**Image** (`observation.images.`*):

```json
{ "role": "observation", "type": "rgb", "dtype": "uint8",
  "state_representation": "HWC", "shape": [256, 256, 3],
  "names": ["height", "width", "channel"],
  "stats": { "min": [[[0]], [[0]], [[0]]], "max": [[[255]], [[255]], [[255]]] },
  "comment": "..." }
```

**Text** (`observation.text`):

```json
{ "role": "observation", "type": "language", "dtype": "string",
  "comment": "Task instruction (language conditioning)." }
```

**Proprio / action vector** (`observation.state.`*, `action.*`):

```json
{ "role": "action", "state_type": "EE_DEL_POS", "state_representation": "XYZ",
  "frame": "base", "kp": 150.0, "kd": 24.49, "dtype": "float32", "units": "m",
  "shape": [3], "order": "0-2",
  "names": ["delta_eef_pos.dx", "delta_eef_pos.dy", "delta_eef_pos.dz"],
  "limits": { "min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0] },
  "normalization": "mean_std",
  "stats": { "mean": [...], "std": [...], "min": [...], "max": [...] },
  "comment": "..." }
```

**Split rule:** use one feature when a quantity is fully described by a consistent
`state_type` + `state_representation` + `frame` (e.g. `EE_ABS_POSE` + `XYZ_AXISANGLE`

- `base`); split only when sub-parts differ (e.g. translation in `base`, rotation
delta in `eef`, or gripper vs arm) and use `order` to reassemble the original vector.

---

## 5. One space per contract (multi-mode wrappers are retired)

The action always lives under `features` as `action.`* — there is **no**
`action_modes` / `observation_modes` wrapper and no `decision_variables` /
`robot_type_variables` schema in v0. A model/env that supports several action or
observation forms is expressed as **separate contracts**, one per form
(e.g. `xvla_libero.json`, `xvla_widowx.json`, `xvla_calvin.json` instead of a
single `xvla.json` with `action_modes`; `droid_joint_pos.json` and
`droid_joint_vel.json` instead of a `droid.json` with `action_modes`). The same
applies to env-side launch variants: an env that can serve several control modes
ships one complete contract per mode and selects the file at launch.

The original multi-mode specs are preserved **as documentation only** under
`demos/contracts/experiments/` (`spec_old.md` and the archived JSON corpus); the
matching code (`matching.py`) does not implement the wrappers.

---

## 6. Conventions & motivations

These come from explicit design decisions; follow them for consistency.

1. **Names follow the producer's own convention.** Env feature leaf-names use the
  simulator/robot's native keys (`agentview_image`, `robot0_eef_pos`, `left_arm`);
   model leaf-names use the checkpoint's keys (e.g. pi0.5's LeRobot keys `image`,
   `image2`). A `role` prefix (`observation.state.*` / `action.*`) keeps keys unique.
   *Why:* matching wires producer→consumer; each side should be self-describing in
   its own terms, and conversions are the matcher's job.
2. `**normalization` is model-side only.** Envs emit raw values → declare `dtype` +
  `stats` (and `limits`) only. *Why:* normalization is part of the model's identity
   (baked into its processors), not the environment.
3. **Encode the robot's *real* action.** When a simulator wrapper exposes a different
  action space than the physical robot (e.g. ALOHA real = absolute joint positions,
   some sims expose EE-delta), spec the real one and note the sim variant in a
   `comment`.
4. **Multi-limb side via key + `names` + `order`,** never a token. Bimanual ALOHA:
  `left_arm` (`order 0-5`), `left_gripper` (`6`), `right_arm` (`7-12`),
   `right_gripper` (`13`). *Why:* keeps `state_type` small and general.
5. **Image layout is explicit (`state_representation`), batch is implicit.** Specs
  describe a single sample; the batch dim is always first and never written.
6. **Image `dtype` = what the producer puts on the wire.** Sim bridges typically emit
  `uint8` [0,255]; a model contract declares what it ingests (often `float32`
   [0,1]). The matcher reconciles dtype + range. *Why:* faithful to each side's I/O.
7. `**frame` is per-feature and EE-only,** and may differ within one pose (OSC:
  base-frame translation, eef-frame rotation). *Why:* this is the #1 silent-failure
   source; making it explicit per sub-feature catches it.
8. **Gripper is its own space (`GRIPPER`)** — e.g. `GRIPPER_ABS_POS`, disambiguated by
  `state_representation` (`BINARY`/`NORM01`/`NORM11`/`REAL`). Keep it out of `JOINT`
   so a gripper never shares a `state_type` token with an arm joint (which otherwise
   pollutes the action signature used for matching/filtering). The gripper is usually
   **absolute even when the arm is delta** — splitting per-feature expresses this
   cleanly. *Exception:* a raw multi-joint `qpos` vector that already bundles finger
   joints with arm joints stays a single `JOINT_*` feature; use `GRIPPER` only for a
   standalone gripper feature. Dexterous multi-DoF hands remain `JOINT`.
9. `**kp`/`kd` on both sides;** `limits` distinct from `stats` (hard bound vs observed
  distribution); `chunk_size` top-level on the model.

---

## 7. Things to look out for / extra research

The hardest fields are semantic and rarely stated plainly — derive them from code,
configs, model cards, and papers, not assumptions. Flag anything uncertain `OPEN:`.

- `**state_representation` (rotation) — the #1 trap.**
  - Euler **order** (`EULXYZ` vs `EULZYX`) and intrinsic vs extrinsic.
  - Quaternion **order** (`QUATWXYZ` vs `QUATXYZW`) — robosuite uses xyzw; many
  libraries use wxyz.
  - `AXISANGLE` (rotvec) vs separate axis+angle; `ROT6D` ordering; `SO3` row/col major.
  - Composite `POSE`/`TWIST` ordering (translation first, then rotation).
- `**state_type` decomposition.**
  - `POS` (translation) vs `POSE` (full 6-DoF) vs `ROT` (orientation only).
  - `REF`: delta relative to *what* (previous step vs first state of an action chunk).
  - Gripper ref ≠ arm ref (absolute gripper, delta arm).
- `**frame`.** base vs world vs eef vs camera; absolute and delta can use different
frames; OSC splits translation/rotation frames. Verify against the controller.
- **Normalization stats.** Part of model identity; per-dataset; `quantile` (VLAs) vs
`mean_std`/`min_max` (imitation policies). Some base checkpoints ship **no** stats
(identity). Get them from the checkpoint's processor config.
- `**units`.** rad vs deg; **normalized/calibration-dependent** joint values (e.g.
SO-100/SO-101 servos report ~[-100,100] % of calibrated range; zero ≠ URDF zero).
Gripper in meters vs normalized vs joint angle.
- **Gripper sign/range.** open vs close sign, `[0,1]` vs `[-1,1]` vs binary.
- **Cameras.** Which physical view each slot is (ego/agent, wrist L/R, external).
Convention: order by importance — egocentric/agent first, then wrist, external last;
record the mapping in `comment`. On a view-count mismatch the model drops or
zero-pads (`padding: true`).
- **Control rate & chunking.** Native rate, `chunk_size`, how many steps execute
before re-inference; policy quality degrades off the native rate.
- **Special embodiments.** PD-target locomotion (Kp/Kd per joint, `action_scale`,
decimation, default joint pos); mobile base extra DOFs (`BASE_`*, SE(2)/SE(3));
discrete mode-switch / terminate flags (RT-X) — not yet first-class, note in
`comment`.
- `**robot_class` disambiguation.** Encode arm DoF + gripper DoF (`arm6g1` vs
`arm7g1`); use `bimanual`, `humanoid`, `quadruped`, `mobile_manip`, else
`unclassed`.

---

## 8. Worked examples (compact)

**Env — single 7-DoF arm, OSC delta (LIBERO Franka):**

```json
{ "robot_type": "franka_panda_libero", "robot_class": "arm7g1", "control_rate": 10,
  "features": {
    "observation.images.agentview_image": { "role": "observation", "type": "rgb",
      "dtype": "uint8", "state_representation": "HWC", "shape": [256,256,3],
      "names": ["height","width","channel"],
      "stats": { "min": [[[0]],[[0]],[[0]]], "max": [[[255]],[[255]],[[255]]] } },
    "observation.text": { "role": "observation", "type": "language", "dtype": "string" },
    "observation.state.robot0_eef_pos": { "role": "observation",
      "state_type": "EE_ABS_POS", "state_representation": "XYZ", "frame": "base",
      "dtype": "float32", "units": "m", "shape": [3], "order": "0-2",
      "names": ["robot0_eef_pos.x","robot0_eef_pos.y","robot0_eef_pos.z"],
      "stats": { "mean": [...], "std": [...], "min": [...], "max": [...] } },
    "action.delta_eef_pos": { "role": "action", "state_type": "EE_DEL_POS",
      "state_representation": "XYZ", "frame": "base", "kp": 150.0, "kd": 24.49,
      "dtype": "float32", "units": "m", "shape": [3], "order": "0-2",
      "names": ["delta_eef_pos.dx","delta_eef_pos.dy","delta_eef_pos.dz"],
      "limits": { "min": [-1.0,-1.0,-1.0], "max": [1.0,1.0,1.0] },
      "stats": { ... } }
  } }
```

**Model — single embodiment VLA (pi0.5):** same feature shape, plus top-level
`model`/`policy_class`/`checkpoint`/`chunk_size`/`control_rate`,
images `float32` with `normalization: "identity"`, and `normalization` on each vector.

---

## 9. Generation checklist (for the agent)

1. Identify the embodiment: `robot_type`, `robot_class` (arm DoF + gripper DoF),
  control rate, DoF layout (URDF/MJCF for joint names & limits).
2. Enumerate observations: cameras (count, resolution, color, layout, dtype), proprio
  vector (split per quantity), text/other modalities.
3. Enumerate the action: real action space; split per quantity; `order`; `frame`;
  `kp`/`kd`; `limits`.
4. For each vector feature set `state_type` + `state_representation` + `units` +
  `names` (producer's convention).
5. Model side only: `normalization` + `stats` (from the checkpoint processors),
  `chunk_size`. Several action/observation forms → one contract per form (§5).
6. Fill `stats`/`limits` where known; **flag every uncertain rotation/frame/unit with
  `OPEN:`** in a `comment`.

