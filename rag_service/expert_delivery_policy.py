from typing import Callable, Any


def maybe_consume_answered_expert_response(
    phone_number: str,
    consumer: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Expert responses are played only by outbound callbacks, never inline."""
    return None
