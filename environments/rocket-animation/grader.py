"""Verify the animated outcome rendered by the supplied rocket viewer."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from itertools import pairwise
from math import cos, hypot, isfinite, pi, sin
from random import SystemRandom
from statistics import fmean
from typing import Any

from hud.graders import EvaluationResult, SubScore

Scene = dict[str, Any]
Sample = Callable[[float], Awaitable[Scene]]
GRAVITY = 9.81
SCALE = 4.4


def _number(value: Any, low: float = -1_000_000, high: float = 1_000_000) -> bool:
    return type(value) in (int, float) and isfinite(value) and low <= value <= high


def _validate(scene: Scene) -> None:
    if not isinstance(scene, dict) or not isinstance(scene.get("rocket"), dict):
        raise ValueError("sceneAt(t) must return a scene with a rocket object")
    rocket = scene["rocket"]
    if not all(_number(rocket.get(key)) for key in ("x", "y", "angle")):
        raise ValueError("rocket coordinates and angle must be finite numbers")
    if not _number(rocket.get("engine"), 0, 1) or type(rocket.get("visible")) is not bool:
        raise ValueError("rocket engine or visibility is invalid")
    for name, limit in (("smoke", 400), ("debris", 60)):
        items = scene.get(name)
        if not isinstance(items, list) or len(items) > limit:
            raise ValueError(f"{name} must be a bounded array")
        ids = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{name} entries must be objects")
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError(f"{name} IDs must be unique nonempty strings")
            ids.add(identifier)
            if not all(_number(item.get(key)) for key in ("x", "y")):
                raise ValueError(f"{name} coordinates must be finite numbers")
            if name == "smoke":
                valid = _number(item.get("radius"), 0.000001, 1000) and _number(item.get("opacity"), 0, 1)
            else:
                valid = (
                    _number(item.get("angle"))
                    and _number(item.get("width"), 0.000001, 1000)
                    and _number(item.get("height"), 0.000001, 1000)
                    and item.get("kind") in ("nose", "body", "fin", "engine")
                )
            if not valid:
                raise ValueError(f"{name} particle properties are invalid")


def _failure(reason: str) -> EvaluationResult:
    return EvaluationResult(reward=0, content=reason, subscores=[SubScore(name="scene", value=0, weight=1)])


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _similarity(error: float, numerical_margin: float, wrong: float) -> float:
    return _unit((wrong - abs(error)) / max(wrong - numerical_margin, 1e-9))


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _items(scene: Scene, name: str) -> dict[str, Any]:
    return {item["id"]: item for item in scene[name]}


def _angle_delta(a: float, b: float) -> float:
    return (b - a + pi) % (2 * pi) - pi


def _shake(times: list[float], scenes: dict[float, Scene]) -> float:
    angles = [0.0]
    for a, b in pairwise(times):
        angles.append(angles[-1] + _angle_delta(scenes[a]["rocket"]["angle"], scenes[b]["rocket"]["angle"]))
    residual = [
        angle - angles[-1] * (t - times[0]) / (times[-1] - times[0]) for angle, t in zip(angles, times, strict=True)
    ]
    return max(residual) - min(residual)


def _on_screen(x: float, y: float, radius: float = 0) -> bool:
    return (
        x + radius >= -585 / SCALE
        and x - radius <= (1280 - 585) / SCALE
        and y + radius >= (651 - 720) / SCALE
        and y - radius <= 651 / SCALE
    )


def _visible_smoke(puff: dict[str, Any]) -> bool:
    return puff["opacity"] * (puff["radius"] * SCALE) ** 2 >= 0.25 and _on_screen(puff["x"], puff["y"], puff["radius"])


def _ground_clearance(piece: dict[str, Any]) -> float:
    width, height = piece["width"] / 2, piece["height"] / 2
    if piece["kind"] == "nose":
        vertices = ((-width, height), (0, -height), (width, height))
    elif piece["kind"] == "fin":
        vertices = ((-width, height), (width, -height), (width, height))
    else:
        vertices = ((-width, height), (-width, -height), (width, -height), (width, height))
    return piece["y"] - max(x * sin(piece["angle"]) + y * cos(piece["angle"]) for x, y in vertices)


def _spread(pieces: dict[str, Any]) -> float:
    if not pieces:
        return 0
    x = fmean(p["x"] for p in pieces.values())
    y = fmean(p["y"] for p in pieces.values())
    return fmean(hypot(p["x"] - x, p["y"] - y) for p in pieces.values())


def _acceleration(points: list[tuple[float, dict[str, Any]]], axis: str) -> float | None:
    spaced = []
    for time, point in points:
        if not spaced or time - spaced[-1][0] >= 0.15:
            spaced.append((time, point))
    if len(spaced) < 3:
        return None
    velocities = [((a + b) / 2, (pb[axis] - pa[axis]) / (b - a)) for (a, pa), (b, pb) in pairwise(spaced)]
    center = fmean(t for t, _ in velocities)
    average = fmean(v for _, v in velocities)
    return sum((t - center) * (v - average) for t, v in velocities) / sum((t - center) ** 2 for t, _ in velocities)


def _smoke_score(
    times: list[float], scenes: dict[float, Scene], powered: list[float]
) -> tuple[float, dict[str, float]]:
    visible = _mean([float(any(_visible_smoke(p) for p in scenes[t]["smoke"])) for t in powered])
    tracks: dict[str, list[tuple[float, Any]]] = {}
    for time in times:
        for puff in scenes[time]["smoke"]:
            tracks.setdefault(puff["id"], []).append((time, puff))
    growth, drift, fading, origins = [], [], [], []
    for track in tracks.values():
        if len(track) < 2 or not any(_visible_smoke(p) for _, p in track):
            continue
        first_t, first = track[0]
        last_t, last = track[-1]
        growth.append(_unit((last["radius"] - first["radius"]) * SCALE / 0.5))
        drift.append(_unit(hypot(last["x"] - first["x"], last["y"] - first["y"]) * SCALE / 0.5))
        peak = max(p["opacity"] for _, p in track)
        decreasing = any(b["opacity"] < a["opacity"] - 1e-6 for (_, a), (_, b) in pairwise(track))
        next_times = [t for t in times if t > last_t]
        terminal_opacity = last["opacity"]
        if next_times:
            previous_t, previous = track[-2]
            terminal_opacity += (
                (last["opacity"] - previous["opacity"]) / (last_t - previous_t) * (next_times[0] - last_t)
            )
        fading.append(float(decreasing and terminal_opacity <= max(0.03, peak * 0.1)))
        if first_t < 8:
            previous_t = max((t for t in times if t < first_t), default=first_t)
            rocket = scenes[first_t]["rocket"]
            before = scenes[previous_t]["rocket"]
            distance = hypot(first["x"] - rocket["x"], first["y"] - (rocket["y"] - 8))
            travel = hypot(rocket["x"] - before["x"], rocket["y"] - before["y"])
            origins.append(float(distance <= travel + first["radius"] + 3))
    at_breakup = set(_items(scenes[8.0], "smoke"))
    stops = all(set(_items(scenes[t], "smoke")) <= at_breakup for t in times if t > 8)
    cleared = not any(_visible_smoke(p) for p in scenes[18.0]["smoke"])
    lifecycle = min(_mean(growth), _mean(drift), _mean(fading), float(stops), float(cleared))
    detail = {
        "visible_emission": visible,
        "growing": _mean(growth),
        "drifting": _mean(drift),
        "fading": _mean(fading),
        "no_new_emission_after_breakup": float(stops),
        "faded_by_end": float(cleared),
    }
    return visible * min(lifecycle, _mean(origins)), detail


def _flight_score(
    observations: list[tuple[float, dict[str, Any]]],
    identifier: str,
    start_time: float,
    start: dict[str, Any],
    velocity: tuple[float, float],
) -> float:
    if any(identifier not in pieces for _, pieces in observations):
        return 0.0
    points = [(t, pieces[identifier]) for t, pieces in observations]
    contact_time = next((t for t, p in points if _ground_clearance(p) <= 0.5), float("inf"))
    airborne = [(t, p) for t, p in points if t < contact_time]
    ay, ax = _acceleration(airborne, "y"), _acceleration(airborne, "x")
    if ay is None or ax is None:
        return 0.0
    acceleration = min(_similarity(ay + GRAVITY, 0.75, 1.75), _similarity(ax, 0.5, 1.5))
    path, landings = [], []
    vy = velocity[1] - GRAVITY * (start_time - 8)
    previous_contact = None
    for t, p in points:
        elapsed = t - start_time
        expected_y = start["y"] + vy * elapsed - GRAVITY * elapsed**2 / 2
        margin = 1 + 0.1 * elapsed**2
        clearance = _ground_clearance(p)
        if t >= contact_time and previous_contact is not None:
            previous_t, previous = previous_contact
            ground_speed = abs(p["x"] - previous["x"]) / (t - previous_t)
            incoming_speed = abs(velocity[0])
            landings.append(_similarity(ground_speed, incoming_speed + 1, incoming_speed * 3 + 8))
        if t >= contact_time:
            previous_contact = (t, p)
        if clearance <= 0.5:
            landings.append(
                min(_similarity(clearance, 0.5, 2), _similarity(max(0, expected_y - p["y"]), margin, margin * 3))
            )
        elif t > contact_time:
            size = max(p["width"], p["height"])
            landings.append(_similarity(clearance, size, size * 3))
        else:
            expected_x = start["x"] + velocity[0] * elapsed
            error = hypot(expected_x - p["x"], expected_y - p["y"])
            path.append(_similarity(error, margin, margin * 3))
    return min(acceleration, min(path, default=0), min(landings, default=1))


async def grade_scene(sample: Sample) -> EvaluationResult:
    """Score the client-visible outcome, independently of the agent's final prose."""
    random = SystemRandom()
    at = {
        "start": 0.0,
        "standby": random.uniform(0.3, 0.8),
        "ignite": random.uniform(1.15, 1.5),
        "liftoff": 2.0,
        "early": random.uniform(2.5, 3),
        "mid": random.uniform(4, 4.7),
        "late": random.uniform(6.7, 7.2),
        "pre_b": random.uniform(7.999, 7.9999),
        "split": 8.0,
        "velocity_a": random.uniform(8.015, 8.025),
        "end": 18.0,
    }
    at["pre_a"] = at["pre_b"] - 0.02
    at["velocity_b"] = at["velocity_a"] + 0.06
    periods = [[start + (i + random.uniform(0.1, 0.3)) / 24 for i in range(24)] for start in (4, 5.5, 6.8)]
    times = sorted(
        set(
            list(at.values())
            + [random.uniform(i * 0.4, (i + 1) * 0.4) for i in range(45)]
            + [t for period in periods for t in period]
        )
    )
    order = times.copy()
    random.shuffle(order)
    scenes = {}
    try:
        for t in order:
            scene = await sample(t)
            _validate(scene)
            scenes[t] = deepcopy(scene)
        for t in random.sample(times, 5):
            repeated = await sample(t)
            _validate(repeated)
            if repeated != scenes[t]:
                return _failure("sceneAt(t) changed on replay; the animation must be deterministic.")
    except (TypeError, ValueError) as error:
        return _failure(f"Scene contract failed: {error}")
    states = {name: scenes[t] for name, t in at.items()}
    rocket = {name: state["rocket"] for name, state in states.items()}
    vx = (rocket["pre_b"]["x"] - rocket["pre_a"]["x"]) / 0.02
    vy = (rocket["pre_b"]["y"] - rocket["pre_a"]["y"]) / 0.02
    center = (rocket["pre_b"]["x"] + vx * (8 - at["pre_b"]), rocket["pre_b"]["y"] + vy * (8 - at["pre_b"]))
    first = _items(states["split"], "debris")
    visible_launch = any(
        scenes[t]["rocket"]["visible"]
        and _on_screen(scenes[t]["rocket"]["x"], scenes[t]["rocket"]["y"])
        and scenes[t]["rocket"]["y"] > rocket["start"]["y"] + 0.5
        for t in times
        if t < 8
    )
    if not visible_launch or center[1] <= rocket["start"]["y"] + 0.5 or vy <= 0 or len(first) < 2:
        return _failure("No rising launch and mid-air breakup were demonstrated.")
    observations = [(t, _items(scenes[t], "debris")) for t in times if t >= 8]
    checks = {}
    initial = rocket["start"]
    height_error = max(60 - center[1], center[1] - 100, 0)
    speed_error = max(20 - vy, vy - 35, 0)
    powered = [at[name] for name in ("early", "mid", "late", "pre_b")]
    ascent = [rocket[name]["y"] for name in ("liftoff", "early", "mid", "late", "pre_b")]
    checks["launch_and_exhaust"] = _mean(
        [
            _mean(
                [
                    _similarity(hypot(scenes[t]["rocket"]["x"], scenes[t]["rocket"]["y"] - 10), 0.25, 4)
                    for t in times
                    if t <= 2
                ]
            ),
            _similarity(_angle_delta(initial["angle"], 0), 0.02, 0.3),
            float(all(a < b for a, b in pairwise(ascent))),
            _similarity(height_error, 0.1, 10),
            _similarity(speed_error, 0.2, 5),
            _similarity(rocket["standby"]["engine"], 0.01, 0.15),
            float(rocket["ignite"]["engine"] > 0),
            _unit(rocket["liftoff"]["engine"] / 0.85),
            _mean([_unit(scenes[t]["rocket"]["engine"] * 14 * SCALE / 2) for t in powered]),
            float(
                all(
                    not scenes[t]["rocket"]["visible"] and scenes[t]["rocket"]["engine"] <= 0.01
                    for t in times
                    if t >= 8
                )
            ),
        ]
    ) * _mean(
        [
            float(scenes[t]["rocket"]["visible"] and _on_screen(scenes[t]["rocket"]["x"], scenes[t]["rocket"]["y"]))
            for t in times
            if t < 8
        ]
    )
    checks["launch_and_exhaust"] *= _mean(
        [_similarity(_angle_delta(scenes[t]["rocket"]["angle"], 0), pi / 9, pi / 2) for t in times if t < 8]
    )
    smoke, smoke_detail = _smoke_score(times, scenes, powered)
    checks["smoke_lifecycle"] = smoke
    calm, max_q, settled = periods
    shake = _shake(max_q, scenes)
    baseline = max(_shake(calm, scenes), _shake(settled, scenes))
    checks["max_q_shake"] = _unit((shake - baseline) * 8 * SCALE)
    coherent, persistent, rotating, visible = [], [], [], []
    for identifier, piece in first.items():
        dx, dy = piece["x"] - center[0], piece["y"] - center[1]
        distance = hypot(max(0, abs(dx) - 4), max(0, abs(dy) - 8))
        coherent.append(_similarity(distance, min(hypot(piece["width"], piece["height"]) / 2 + 0.25, 8), 16))
        track = [pieces.get(identifier) for _, pieces in observations]
        stable = all(
            p is not None and (p["kind"], p["width"], p["height"]) == (piece["kind"], piece["width"], piece["height"])
            for p in track
        )
        persistent.append(float(stable))
        if stable:
            travel = sum(
                abs(_angle_delta(a["angle"], b["angle"]))
                for a, b in pairwise(track)
                if _ground_clearance(a) > 0.5 and _ground_clearance(b) > 0.5
            )
            rotating.append(_unit(travel * min(piece["width"], piece["height"]) * SCALE))
            visible.append(
                _mean([float(_on_screen(p["x"], p["y"])) for p in track])
                * _unit(min(piece["width"], piece["height"]) * SCALE / 2)
            )
        else:
            rotating.append(0)
            visible.append(0)
    kinds = {p["kind"] for p in first.values()}
    airborne_spreads = [
        _spread(pieces)
        for _, pieces in observations
        if pieces and all(_ground_clearance(p) > 0.5 for p in pieces.values())
    ]
    spread = _unit((max(airborne_spreads, default=_spread(first)) - _spread(first)) * SCALE / 2)
    geometry = min(_mean(coherent), _mean(persistent), _mean(visible), spread)
    checks["coherent_breakup"] = geometry * _mean([len(kinds & {"nose", "body", "fin", "engine"}) / 4, _mean(rotating)])
    a, b = (_items(states[name], "debris") for name in ("velocity_a", "velocity_b"))
    midpoint = (at["velocity_a"] + at["velocity_b"]) / 2 - 8
    velocities = {
        key: ((b[key]["x"] - a[key]["x"]) / 0.06, (b[key]["y"] - a[key]["y"]) / 0.06 + GRAVITY * midpoint)
        for key in first.keys() & a.keys() & b.keys()
    }
    speed = max(hypot(vx, vy), 1)
    if velocities:
        mean_vx, mean_vy = fmean(v[0] for v in velocities.values()), fmean(v[1] for v in velocities.values())
        mean_motion = _similarity(hypot(mean_vx - vx, mean_vy - vy), 0.15 * speed + 0.5, 0.35 * speed + 0.5)
        kicks = _mean([_similarity(hypot(x - vx, y - vy), 0.3 * speed + 0.5, speed) for x, y in velocities.values()])
        checks["inherited_momentum"] = min(mean_motion, kicks) * len(velocities) / len(first)
    else:
        checks["inherited_momentum"] = 0
    checks["ballistic_gravity"] = _mean(
        [
            _flight_score(observations, key, at["velocity_a"], a[key], velocities[key]) if key in velocities else 0
            for key in first
        ]
    )
    weights = {
        "launch_and_exhaust": 0.10,
        "smoke_lifecycle": 0.20,
        "max_q_shake": 0.10,
        "coherent_breakup": 0.15,
        "inherited_momentum": 0.20,
        "ballistic_gravity": 0.25,
    }
    subscores = [SubScore(name=name, value=value, weight=weights[name]) for name, value in checks.items()]
    subscores[1].info = smoke_detail
    failed = [name for name, value in checks.items() if value < 0.99]
    return EvaluationResult(
        reward=sum(s.value * s.weight for s in subscores),
        subscores=subscores,
        content="All scene outcomes demonstrated." if not failed else "Review: " + ", ".join(failed),
        info={"sample_count": len(times)},
    )
