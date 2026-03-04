"""GPU-accelerated Kerr black hole ray tracing."""

from .animation import AnimationStats, render_animation
from .charged_particles import ChargedParticleAnimationStats, ChargedParticleOrbiter
from .config import RenderConfig
from .geometry import ISCOResult, isco_radius_general, isco_radius_grid
from .raytracer import KerrRayTracer
from .starship import Starship, StarshipThrustCommand, StarshipThrustSegment

__all__ = [
    "RenderConfig",
    "KerrRayTracer",
    "AnimationStats",
    "ChargedParticleAnimationStats",
    "ChargedParticleOrbiter",
    "render_animation",
    "ISCOResult",
    "isco_radius_general",
    "isco_radius_grid",
    "Starship",
    "StarshipThrustCommand",
    "StarshipThrustSegment",
]
