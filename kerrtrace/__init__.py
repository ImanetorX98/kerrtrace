"""GPU-accelerated Kerr black hole ray tracing."""

from .animation import AnimationStats, render_animation
from .charged_particles import ChargedParticleAnimationStats, ChargedParticleOrbiter
from .config import RenderConfig
from .geometry import ISCOResult, isco_radius_general, isco_radius_grid
from .particle_renderer import ParticleFrame, ParticleRenderer, RaytracedParticleRenderer
from .raytracer import KerrRayTracer, TraceFrame
from .starship import Starship, StarshipThrustCommand, StarshipThrustSegment

__all__ = [
    "RenderConfig",
    "KerrRayTracer",
    "TraceFrame",
    "AnimationStats",
    "ChargedParticleAnimationStats",
    "ChargedParticleOrbiter",
    "ParticleFrame",
    "ParticleRenderer",
    "RaytracedParticleRenderer",
    "render_animation",
    "ISCOResult",
    "isco_radius_general",
    "isco_radius_grid",
    "Starship",
    "StarshipThrustCommand",
    "StarshipThrustSegment",
]
