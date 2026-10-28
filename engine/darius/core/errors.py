"""Errors that cross process boundaries.

Pipeline stages run in separate cloud containers, and an exception raised in one
is pickled back to the orchestrator. Third-party exception types often can't be
unpickled there (their module isn't installed in that image), so stages re-raise
them as plain RuntimeErrors built with `describe`. `InputError` is ours and is
importable everywhere, so it survives the trip and reaches the user verbatim.
"""

from __future__ import annotations


class InputError(ValueError):
    """The video can't produce a usable scan; the message is shown to the user."""


def describe(error: BaseException | None) -> str:
    """A plain-string error summary.

    Duck-typed on purpose: this module is imported by every image, including ones
    without the OpenAI SDK.
    """
    status, message = getattr(error, "status_code", None), getattr(error, "message", None)
    if status is not None and message is not None:
        return f"HTTP {status}: {message}"
    return f"{type(error).__name__}: {error}"
