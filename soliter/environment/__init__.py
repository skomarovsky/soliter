"""Environment simulation: world, resources, physics, and sensors."""

from .world import World, WorldConfig, TimeOfDay, Season
from .resources import Resource, Feeder, Fountain, Heater, ResourceConfig, create_default_resources
from .physics import Physics, RaycastResult
from .sensors import SensorSystem, SensorConfig

__all__ = [
    'World',
    'WorldConfig',
    'TimeOfDay',
    'Season',
    'Resource',
    'ResourceConfig',
    'Feeder',
    'Fountain',
    'Heater',
    'create_default_resources',
    'Physics',
    'RaycastResult',
    'SensorSystem',
    'SensorConfig',
]
