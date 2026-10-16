"""Shared diagnostic prediction and fusion interfaces for EngineGuard sensors."""

from .schema import SchemaValidationError, validate_prediction
from .fusion import fuse_predictions, fuse_sensor_outputs

__all__ = ["SchemaValidationError", "validate_prediction", "fuse_predictions", "fuse_sensor_outputs"]
