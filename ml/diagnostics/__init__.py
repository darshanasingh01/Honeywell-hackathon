"""Shared diagnostic prediction and fusion interfaces for EngineGuard sensors."""

from .schema import SchemaValidationError, validate_prediction

__all__ = ["SchemaValidationError", "validate_prediction"]
