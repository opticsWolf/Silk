"""Reflection loop for self-correction.

Enables the model to retry when validation errors occur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ReflectionConfig:
    """Configuration for the reflection loop.

    Attributes:
        max_retries: Maximum number of retries for tool validation errors.
        max_output_retries: Maximum number of retries for output validation errors.
        retry_prompt: Prompt to send when retrying after a tool validation error.
        tool_error_prompt: Prompt to send when retrying after a tool call error.
        output_error_prompt: Prompt to send when retrying after an output validation error.
    """

    max_retries: int = 3
    """Maximum number of retries for tool validation errors."""

    max_output_retries: int = 2
    """Maximum number of retries for output validation errors."""

    retry_prompt: str = (
        "Your previous response had validation errors. "
        "Please fix the errors and try again."
    )
    """Prompt to send when retrying after a tool validation error."""

    tool_error_prompt: str = (
        "The tool call arguments did not match the required schema. "
        "A corrected schema is provided in the system message above. "
        "Fix the arguments and try again."
    )
    """Prompt to send when retrying after a tool call error."""

    output_error_prompt: str = (
        "Your output did not match the required schema. "
        "Please fix the output and try again."
    )
    """Prompt to send when retrying after an output validation error."""


class ModelRetry(Exception):
    """Raised to signal the model should retry.

    Attributes:
        message: The error message to send back to the model.
        retry_count: Number of retries so far.
        max_retries: Maximum number of retries allowed.
    """

    def __init__(self, message: str, retry_count: int = 0, max_retries: int = 3):
        super().__init__(message)
        self.message = message
        self.retry_count = retry_count
        self.max_retries = max_retries


def parse_tool_error(content: str) -> tuple[bool, str]:
    """Parse a tool error response for validation errors.

    Args:
        content: The tool error content.

    Returns:
        Tuple of (is_error, error_message).
        ``is_error`` is True if the content contains a validation error.
        ``error_message`` is the parsed error message, or empty string if not an error.
    """
    import json

    if not content:
        return False, ""

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # Truthiness, NOT key presence. Many successful tools include an
            # ``"error": null`` field in their result payload (read_file,
            # write_file, view_file, …). Treating the mere presence of the key
            # as a failure flagged every successful call as an error and fired
            # a spurious reflection retry ("arguments did not match the
            # schema") on content the model had already received.
            err = data.get("error")
            if err:
                details = data.get("details", "")
                suggestion = data.get("suggestion", "")
                if details:
                    return True, f"{err}\nDetails: {details}"
                if suggestion:
                    return True, f"{err}\n{suggestion}"
                return True, str(err)
    except (json.JSONDecodeError, TypeError):
        pass

    # Check for plain text error
    if content.startswith("Error:") or content.startswith("error:"):
        return True, content
    if content.startswith("Failed:") or content.startswith("failed:"):
        return True, content

    return False, ""


#: error_type tags that reflection must NOT retry — feeding these back for a
#: retry cannot succeed (e.g. the active role hard-denies the tool).
# role_denied: the role will not change mid-run. budget_exceeded: the
# per-run budget cannot recover within the same run either.
NON_RETRYABLE_ERROR_TYPES = frozenset({"role_denied", "budget_exceeded"})


def is_retryable_tool_error(content: str) -> bool:
    """Whether a tool error result is worth a reflection retry.

    Structured errors tagged with a ``NON_RETRYABLE_ERROR_TYPES`` entry (see
    ``ToolBox._error(error_type=...)``) are permanent — retrying burns
    reflection budget without any chance of success.
    """
    import json

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return True
    if isinstance(data, dict) and data.get("error_type") in NON_RETRYABLE_ERROR_TYPES:
        return False
    return True


def parse_output_error(content: str, schema: dict | None = None) -> tuple[bool, str]:
    """Parse an output for validation errors.

    Args:
        content: The output content.
        schema: Optional schema for more detailed error messages.

    Returns:
        Tuple of (is_error, error_message).
        ``is_error`` is True if the content contains a validation error.
        ``error_message`` is the parsed error message, or empty string if not an error.
    """
    import json

    if not content:
        return True, "Empty output"

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # Truthiness, not key presence — a payload carrying ``error: null``
            # is a success, not a validation failure (see parse_tool_error).
            if data.get("error"):
                return True, str(data["error"])
    except json.JSONDecodeError:
        return True, f"Output is not valid JSON: {content[:200]}..."

    return False, ""


def build_retry_message(
    error_type: str,
    error_message: str,
    retry_prompt: str | None = None,
) -> str:
    """Build a retry message for the model.

    Args:
        error_type: The type of error (e.g. "tool", "output").
        error_message: The error message.
        retry_prompt: Optional retry prompt.

    Returns:
        The retry message.
    """
    if retry_prompt:
        return f"{retry_prompt}\n\nError: {error_message}"
    return f"Error: {error_message}"
