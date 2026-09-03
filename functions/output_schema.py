"""Structured output schema and validation.

Enables the model's final output to be validated against a Pydantic model
or JSON schema. Validation errors are fed back to the model for self-correction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .reflection import ReflectionConfig


@dataclass
class OutputSchema:
    """Schema for structured output validation.

    Attributes:
        model: Optional Pydantic model for validation.
        schema: JSON schema for validation.
        name: Human-readable name for the schema.
        description: Description of the expected output.
    """

    model: type[BaseModel] | None = None
    """Pydantic model for validation."""

    schema: dict[str, Any] | None = None
    """JSON schema for validation."""

    name: str = "structured_output"
    """Human-readable name for the schema."""

    description: str = ""
    """Description of the expected output."""

    def __post_init__(self) -> None:
        if self.model is not None and self.schema is None:
            self.schema = self.model.model_json_schema()
        if self.schema is not None and self.model is None:
            # Try to infer a model from the schema (best effort)
            pass

    @classmethod
    def from_model(cls, model: type[BaseModel], name: str = "", description: str = "") -> OutputSchema:
        """Create an OutputSchema from a Pydantic model.

        Args:
            model: Pydantic model class.
            name: Human-readable name.
            description: Description of the expected output.

        Returns:
            An OutputSchema instance.
        """
        return cls(
            model=model,
            name=name or model.__name__,
            description=description or f"Output matching {model.__name__} schema",
        )

    @classmethod
    def from_dict(cls, schema_dict: dict[str, Any], name: str = "", description: str = "") -> OutputSchema:
        """Create an OutputSchema from a JSON schema dict.

        Args:
            schema_dict: JSON schema dictionary.
            name: Human-readable name.
            description: Description of the expected output.

        Returns:
            An OutputSchema instance.
        """
        return cls(
            schema=schema_dict,
            name=name or "json_schema",
            description=description or "Output matching JSON schema",
        )

    def validate(self, output: str) -> tuple[bool, Any | str]:
        """Validate output against the schema.

        Args:
            output: The output string to validate.

        Returns:
            Tuple of (is_valid, result_or_error).
            If valid, result_or_result is the parsed data (or Pydantic model instance).
            If invalid, result_or_error is an error message string.
        """
        if not output or not output.strip():
            return False, "Empty output"

        # Try Pydantic model validation first
        if self.model is not None:
            try:
                data = json.loads(output)
                if isinstance(data, dict):
                    instance = self.model(**data)
                    return True, instance
                else:
                    return False, f"Expected object, got {type(data).__name__}"
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"
            except ValidationError as e:
                errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                return False, f"Validation failed: {'; '.join(errors)}"

        # Fall back to JSON schema validation
        if self.schema is not None:
            try:
                data = json.loads(output)
                # Simple schema validation (basic structure check)
                required = self.schema.get("required", [])
                properties = self.schema.get("properties", {})

                if not isinstance(data, dict):
                    return False, f"Expected object, got {type(data).__name__}"

                missing = [prop for prop in required if prop not in data]
                if missing:
                    return False, f"Missing required fields: {', '.join(missing)}"

                # Type checking for each property
                for key, value in data.items():
                    if key in properties:
                        prop_schema = properties[key]
                        expected_type = prop_schema.get("type")
                        if expected_type and not self._check_type(value, expected_type):
                            return False, f"Field '{key}' expected {expected_type}, got {type(value).__name__}"

                return True, data
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"

        # No schema to validate against
        return True, output

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected JSON schema type.

        Args:
            value: The value to check.
            expected_type: The expected JSON schema type.

        Returns:
            True if the value matches the expected type.
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected_python_type: Any = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        return isinstance(value, expected_python_type)

    def build_instruction(self) -> str:
        """Build an instruction for the model describing the expected output.

        Returns:
            A string instruction for the model.
        """
        if not self.description:
            return ""

        instruction = "Your response must match the following schema:\n"
        instruction += f"Name: {self.name}\n"
        instruction += f"Description: {self.description}\n"

        if self.schema:
            instruction += f"Schema:\n{json.dumps(self.schema, indent=2)}"

        return instruction


class OutputValidator:
    """Validates output against a schema with reflection/retry support.

    Attributes:
        schema: The OutputSchema to validate against.
        reflection_config: Configuration for the reflection loop.
    """

    def __init__(
        self,
        schema: OutputSchema,
        reflection_config: ReflectionConfig | None = None,
    ) -> None:
        self.schema = schema
        self.reflection_config = reflection_config or ReflectionConfig()

    def validate_with_reflection(
        self,
        output: str,
        max_retries: int | None = None,
    ) -> tuple[bool, Any | str, int]:
        """Validate output with reflection/retry support.

        Args:
            output: The output string to validate.
            max_retries: Maximum number of retries (overrides reflection_config).

        Returns:
            Tuple of (is_valid, result_or_error, retry_count).
        """
        retries = max_retries if max_retries is not None else self.reflection_config.max_output_retries
        retry_count = 0

        while retry_count <= retries:
            is_valid, result = self.schema.validate(output)
            if is_valid:
                return True, result, retry_count

            retry_count += 1
            if retry_count > retries:
                break

            # Build retry prompt
            error_msg = result if isinstance(result, str) else str(result)
            retry_prompt = self.reflection_config.output_error_prompt
            output = f"{retry_prompt}\n\nError: {error_msg}"

        return False, f"Validation failed after {retries} retries: {result}", retry_count
