"""Verify grading through scene sampling, including plausible but wrong physics."""

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("rocket_grader", Path(__file__).parents[1] / "grader.py")
assert _SPEC and _SPEC.loader
_GRADER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GRADER)
grade_scene = _GRADER.grade_scene


def _flight(t):
    elapsed = max(0, min(t, 8) - 2)
    return 0.2 * elapsed**2, 10 + 2 * elapsed**2


def _support(kind, angle):
    vertices = {
        "nose": ((-0.8, 1.5), (0, -1.5), (0.8, 1.5)),
        "fin": ((-0.8, 1.5), (0.8, -1.5), (0.8, 1.5)),
    }.get(kind, ((-0.8, 1.5), (-0.8, -1.5), (0.8, -1.5), (0.8, 1.5)))
    return max(x * math.sin(angle) + y * math.cos(angle) for x, y in vertices)


@lru_cache
def _impact(kind, initial_y, vy, spin, gravity):
    if gravity == 0:
        return float("inf")
    low, high = 0, 40
    for _ in range(60):
        mid = (low + high) / 2
        if initial_y + vy * mid - gravity * mid**2 / 2 > _support(kind, spin * mid):
            low = mid
        else:
            high = mid
    return high


def _scene(t, *, defect=None):
    x, y = _flight(t)
    engine = max(0, min(1, t - 1)) if t < 8 else 0
    angle = 0
    if 5.5 < t < 6.5:
        phase = t - 5.5
        angle = 0.06 * math.sin(2 * math.pi * 6 * phase) * math.sin(math.pi * phase)
    rocket = {"x": x, "y": y, "angle": angle, "engine": engine, "visible": t < 8}
    if defect == "static":
        return {
            "rocket": {**rocket, "x": 0, "y": 10, "engine": 0, "angle": 0, "visible": True},
            "smoke": [],
            "debris": [],
        }
    smoke = []
    for index in range(50):
        birth = 1 + index * 0.14
        age = t - birth
        if birth >= 8 or not 0 <= age < 3.2:
            continue
        birth_x, birth_y = _flight(birth)
        smoke.append(
            {
                "id": f"smoke-{index}",
                "x": 2000 if defect == "offscreen_smoke" else birth_x + math.sin(index) * 0.4 + age * 1.2,
                "y": 1000 if defect == "offscreen_smoke" else max(0.5 + age * 0.2, birth_y - 8 - age * 3),
                "radius": 0.0001 if defect == "tiny_smoke" else 0.4 + age * 0.8,
                "opacity": 0.8 * (1 - age / 3.2) if defect != "never_fades" else 0.8,
            }
        )
    debris = []
    if t >= 8:
        elapsed = t - 8
        for index in range(6):
            initial_x = 7.2 + (index - 2.5) * 0.6
            initial_y = 82 + (index - 2.5) * 2
            vx = 2.4 + (index - 2.5) * 0.8
            vy = 24 + (index - 2.5) * 0.7
            if defect == "no_inheritance":
                vx, vy = 0, 0
            elif defect == "lost_momentum":
                vx, vy = 2.4 * 0.65 + (index - 2.5) * 0.8, 24 * 0.65 + (index - 2.5) * 0.7
            acceleration = 0 if defect == "no_gravity" else 8 if defect == "wrong_gravity" else 9.81
            kind = ("nose", "body", "body", "fin", "fin", "engine")[index]
            spin = (index - 2.2) * 0.9
            impact = _impact(kind, initial_y, vy, spin, acceleration)
            motion_time = min(elapsed, impact)
            angle = spin * motion_time
            debris.append(
                {
                    "id": f"part-{index}",
                    "x": initial_x + vx * motion_time + (200 if defect == "teleport" else 0),
                    "y": _support(kind, angle)
                    if elapsed >= impact
                    else initial_y + vy * motion_time - acceleration * motion_time**2 / 2,
                    "angle": angle,
                    "width": 1.6,
                    "height": 3.0,
                    "kind": kind,
                }
            )
    return {"rocket": rocket, "smoke": smoke, "debris": debris}


async def _grade(defect=None):
    async def sample(t):
        return _scene(t, defect=defect)

    return await grade_scene(sample)


def _early_landings(t, *, teleport=False):
    scene = _scene(t)
    if t < 8:
        return scene
    elapsed = t - 8
    width, height, final_angle = 1.6, 3.0, 0.4
    ground_y = (width * math.sin(final_angle) + height * math.cos(final_angle)) / 2
    pieces = []
    for index in range(60):
        initial_y = 82 + (index % 5 - 2) * 0.5
        vy = -45 if index < 7 else 28
        vx = 2.4 + (index % 5 - 2) * 0.2
        impact = (vy + math.sqrt(vy**2 + 2 * 9.81 * (initial_y - ground_y))) / 9.81
        motion_time = min(elapsed, impact)
        y = initial_y + vy * motion_time - 9.81 * motion_time**2 / 2
        if elapsed >= impact or teleport and index < 7 and elapsed >= 0.2:
            y = ground_y
        pieces.append(
            {
                "id": f"part-{index}",
                "x": 7.2 + (index % 5 - 2) * 0.2 + vx * motion_time,
                "y": y,
                "angle": min(elapsed, 1) * final_angle,
                "width": width,
                "height": height,
                "kind": ("nose", "body", "fin", "engine")[index % 4],
            }
        )
    scene["debris"] = pieces
    return scene


@pytest.mark.asyncio
async def test_complete_physical_animation_receives_full_credit():
    result = await _grade()
    assert result.reward == pytest.approx(1)
    assert all(score.value == 1 for score in result.subscores)


@pytest.mark.asyncio
async def test_wrapped_orientations_preserve_the_same_physical_animation():
    async def sample(t):
        scene = _scene(t)
        if t >= 8:
            for piece in scene["debris"]:
                piece["angle"] = (piece["angle"] + math.pi) % (2 * math.pi) - math.pi
        return scene

    result = await grade_scene(sample)
    assert result.reward == pytest.approx(1)


@pytest.mark.asyncio
async def test_fewer_large_puffs_and_one_wobble_are_valid_style_choices():
    async def sample(t):
        scene = _scene(t)
        scene["smoke"] = [p for p in scene["smoke"] if int(p["id"].split("-")[1]) % 4 == 0]
        scene["smoke"].append({"id": "fully-faded", "x": 0, "y": 2, "radius": 1, "opacity": 0})
        if 5.5 < t < 6.5:
            phase = t - 5.5
            scene["rocket"]["angle"] = 0.06 * math.sin(2 * math.pi * phase) * math.sin(math.pi * phase) ** 2
        return scene

    assert (await grade_scene(sample)).reward == pytest.approx(1)


@pytest.mark.asyncio
async def test_a_hidden_launch_is_not_a_rendered_launch():
    async def sample(t):
        scene = _scene(t)
        scene["rocket"]["visible"] = False
        return scene

    assert (await grade_scene(sample)).reward == 0


@pytest.mark.asyncio
async def test_sideways_ascent_loses_upright_launch_credit():
    async def sample(t):
        scene = _scene(t)
        if 2 < t < 8:
            scene["rocket"]["angle"] += math.pi / 2
        return scene

    result = await grade_scene(sample)
    assert next(s.value for s in result.subscores if s.name == "launch_and_exhaust") < 1


@pytest.mark.asyncio
async def test_jumps_by_whole_revolutions_are_not_visible_tumbling():
    async def sample(t):
        scene = _scene(t)
        for piece in scene["debris"]:
            piece["angle"] = 2 * math.pi * math.floor(t * 10)
        return scene

    result = await grade_scene(sample)
    assert next(s.value for s in result.subscores if s.name == "coherent_breakup") < 1


@pytest.mark.asyncio
async def test_ballistic_fragments_can_settle_while_others_remain_airborne():
    async def sample(t):
        return _early_landings(t)

    landed = _early_landings(9.73)["debris"][:7]
    assert all(piece["y"] < 2 for piece in landed)
    result = await grade_scene(sample)
    assert next(score.value for score in result.subscores if score.name == "ballistic_gravity") == 1


@pytest.mark.asyncio
async def test_early_ground_teleport_cannot_hide_in_minority_landing_samples():
    async def sample(t):
        return _early_landings(t, teleport=True)

    result = await grade_scene(sample)
    scores = {score.name: score.value for score in result.subscores}
    assert scores["ballistic_gravity"] < 1


@pytest.mark.asyncio
async def test_ground_teleport_cannot_replace_airborne_acceleration_evidence():
    async def sample(t):
        scene = _early_landings(t)
        if t >= 8.2:
            for piece in scene["debris"]:
                piece["y"] = (
                    piece["width"] * math.sin(piece["angle"]) + piece["height"] * math.cos(piece["angle"])
                ) / 2
        return scene

    result = await grade_scene(sample)
    scores = {score.name: score.value for score in result.subscores}
    assert scores["ballistic_gravity"] == 0


@pytest.mark.asyncio
async def test_stationary_starter_receives_zero_reward():
    result = await _grade("static")
    assert result.reward == 0
    assert "launch" in result.content


@pytest.mark.asyncio
async def test_old_checkpoint_only_animation_cannot_earn_reward():
    old_checkpoints = {
        0.0,
        0.7,
        1.3,
        1.8,
        2.0,
        2.5,
        3.13,
        3.37,
        4.5,
        6.7,
        7.97,
        7.99,
        8.0,
        8.001,
        8.02,
        8.08,
        8.31,
        8.97,
        9.73,
        10.65,
        11.6,
        13.7,
        17.5,
        18.0,
    }
    old_checkpoints.update(round(index * 0.431, 6) for index in range(42))
    for start in (4.1, 5.5, 6.9):
        old_checkpoints.update(round(start + index * 0.047, 6) for index in range(22))
    observed = []

    async def sample(t):
        observed.append(t)
        return _scene(t) if t in old_checkpoints else _scene(t, defect="static")

    result = await grade_scene(sample)
    assert any(t not in old_checkpoints for t in observed)
    assert result.reward == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("gap", ["missing", "stationary", "resized"])
async def test_debris_cannot_only_work_inside_known_phase_windows(gap):
    windows = ((8, 8.086), (8.2, 8.4), (8.85, 9), (9.65, 9.8), (10.55, 10.75), (11.5, 11.7), (13.55, 13.7))

    async def sample(t):
        scene = _scene(t)
        if 8 <= t <= 13.7 and not any(low <= t <= high for low, high in windows):
            if gap == "missing":
                scene["debris"] = []
            elif gap == "stationary":
                scene["debris"] = _scene(8.1)["debris"]
            else:
                for piece in scene["debris"]:
                    piece["width"] = 0.0001
                    piece["height"] = 0.0001
        return scene

    result = await grade_scene(sample)
    assert result.reward < 1
    scores = {score.name: score.value for score in result.subscores}
    component = "coherent_breakup" if gap == "resized" else "ballistic_gravity"
    assert scores[component] < 1


@pytest.mark.asyncio
async def test_fragments_cannot_vanish_during_the_clip():
    async def sample(t):
        scene = _scene(t)
        if t >= 13.7:
            scene["debris"] = []
        return scene

    result = await grade_scene(sample)
    assert result.reward < 1


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["teleport", "no_inheritance", "no_gravity", "wrong_gravity", "lost_momentum"])
async def test_incorrect_debris_physics_loses_the_relevant_credit(defect):
    result = await _grade(defect)
    scores = {score.name: score.value for score in result.subscores}
    assert result.reward < (0.9 if defect == "teleport" else 0.85)
    if defect in ("no_inheritance", "lost_momentum"):
        assert scores["inherited_momentum"] < 0.5
    if defect == "teleport":
        assert scores["coherent_breakup"] == 0
    if defect in ("no_gravity", "wrong_gravity"):
        assert scores["ballistic_gravity"] < 0.5


@pytest.mark.asyncio
async def test_nonfading_smoke_loses_lifecycle_credit():
    result = await _grade("never_fades")
    scores = {score.name: score.value for score in result.subscores}
    assert scores["smoke_lifecycle"] <= 0.6


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["tiny_smoke", "offscreen_smoke"])
async def test_invisible_smoke_receives_no_lifecycle_credit(defect):
    result = await _grade(defect)
    scores = {score.name: score.value for score in result.subscores}
    assert scores["smoke_lifecycle"] == 0


@pytest.mark.asyncio
async def test_invalid_numeric_state_receives_zero():
    async def sample(t):
        result = _scene(t)
        result["rocket"]["x"] = float("nan")
        return result

    result = await grade_scene(sample)
    assert result.reward == 0
    assert "finite" in result.content


@pytest.mark.asyncio
async def test_state_changes_on_replay_receive_zero():
    calls = 0

    async def sample(t):
        nonlocal calls
        calls += 1
        result = _scene(t)
        result["rocket"]["x"] += calls * 0.0001
        return result

    result = await grade_scene(sample)
    assert result.reward == 0
    assert "deterministic" in result.content


@pytest.mark.asyncio
async def test_scene_script_errors_are_failed_contracts():
    async def sample(t):
        raise ValueError("sceneAt is not a function")

    result = await grade_scene(sample)
    assert result.reward == 0
    assert "sceneAt is not a function" in result.content
