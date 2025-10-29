# ruff: noqa: S311
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.live import Live
from rich.text import Text

from hud.utils.hud_console import hud_console

if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class Particle:
    """A confetti particle with physics."""

    x: float
    y: float
    vx: float  # velocity x
    vy: float  # velocity y
    char: str
    color: str

    def update(self, gravity: float = 0.5, fps: float = 30.0) -> None:
        """Update particle position and velocity."""
        dt = 1.0 / fps
        self.x += self.vx * dt
        self.vy += gravity  # Apply gravity
        self.y += self.vy * dt


class ConfettiSystem:
    """Minimal confetti system inspired by confetty."""

    # Confetty-style colors
    COLORS: ClassVar[list[str]] = ["#a864fd", "#29cdff", "#78ff44", "#ff718d", "#fdff6a"]
    # Confetty-style characters
    CHARS: ClassVar[list[str]] = ["█", "▓", "▒", "░", "▄", "▀"]

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.particles: list[Particle] = []

    def spawn_burst(self, num_particles: int = 75) -> None:
        """Spawn a burst of confetti particles from the top center."""
        center_x = self.width / 2

        for _ in range(num_particles):
            # Start from top center with some horizontal spread
            x = center_x + (self.width / 4) * (random.random() - 0.5)
            y = 0

            # Random velocities - horizontal spread and upward/slight downward initial velocity
            vx = (random.random() - 0.5) * 100
            vy = random.random() * 50 - 25  # Some go up first

            particle = Particle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                char=random.choice(self.CHARS),
                color=random.choice(self.COLORS),
            )
            self.particles.append(particle)

    def update(self) -> None:
        """Update all particles and remove off-screen ones."""
        # Update physics
        for particle in self.particles:
            particle.update()

        # Remove particles that are off-screen
        self.particles = [p for p in self.particles if 0 <= p.x < self.width and p.y < self.height]

    def render(self) -> str:
        """Render the particle system to a string."""
        # Create empty grid
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Place particles
        for particle in self.particles:
            x, y = int(particle.x), int(particle.y)
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = particle.char

        # Convert to string
        return "\n".join("".join(row) for row in grid)

    def render_with_colors(self) -> Text:
        """Render the particle system with colors for Rich."""
        text = Text()

        # Create empty grid with color info
        grid: list[list[tuple[str, str] | None]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]

        # Place particles with their colors
        for particle in self.particles:
            x, y = int(particle.x), int(particle.y)
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = (particle.char, particle.color)

        # Build colored text
        for row in grid:
            for cell in row:
                if cell:
                    char, color = cell
                    text.append(char, style=color)
                else:
                    text.append(" ")
            text.append("\n")

        return text


def show_confetti(console: Console, seconds: float = 2.5) -> None:
    """Display celebratory confetti animation inspired by confetty.

    Shows "Starting training!" message first, then creates two bursts of
    falling confetti particles that fall away completely.

    Args:
        console: Rich console instance
        seconds: Duration to show confetti
    """
    # Show celebratory message first
    console.print(
        "[bold green]🎉 Starting training! See your model on https://hud.ai/models[/bold green]"
    )
    time.sleep(0.3)  # Brief pause to see the message

    width = min(console.size.width, 120)  # Cap width for performance
    height = min(console.size.height - 2, 30)  # Leave room for message

    # Create confetti system
    system = ConfettiSystem(width, height)

    fps = 30
    frame_time = 1.0 / fps

    # First burst at the beginning
    system.spawn_burst(num_particles=60)

    # Track when to spawn second burst
    second_burst_frame = int(fps * 0.4)  # Second burst after 0.4 seconds

    with Live("", refresh_per_second=fps, console=console, transient=True) as live:
        frame = 0
        # Keep running until all particles have fallen off screen
        while frame < seconds * fps or len(system.particles) > 0:
            # Spawn second burst
            if frame == second_burst_frame:
                system.spawn_burst(num_particles=60)

            system.update()
            live.update(system.render_with_colors())
            time.sleep(frame_time)
            frame += 1


def show_confetti_async(console: Console, seconds: float = 2.5) -> None:
    """Non-blocking confetti animation that runs in a background thread.

    The animation will run independently while training starts immediately.
    """
    import threading

    def _run_confetti() -> None:
        try:
            show_confetti(console, seconds)
        except Exception:
            hud_console.info("Launching training...")

    thread = threading.Thread(target=_run_confetti, daemon=True)
    thread.start()
    # Don't wait - let training start immediately while confetti plays


__all__ = ["show_confetti", "show_confetti_async"]
