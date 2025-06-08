from functools import lru_cache
from typing import Any, Literal, Set, Type, TypeVar, Union, get_args, get_origin

from pyhub.llm.utils.enums import TextChoices


def type_to_flatten_set(type_: TypeVar) -> set[str]:
    """Convert a type to a flattened set of string values."""
    result = set()

    for arg in get_args(type_):
        if isinstance(arg, str):
            result.add(arg.lower())
        elif hasattr(arg, "__args__"):
            for val in arg.__args__:
                if isinstance(val, str):
                    result.add(val.lower())

    return result


def enum_to_flatten_set(enum: Type[TextChoices]) -> set[str]:
    """Convert a TextChoices enum to a flattened set of string values."""
    return set(map(lambda s: s.lower(), enum.values()))


@lru_cache(maxsize=32)
def get_literal_values(*type_hints: Any) -> Set[Any]:
    """
    Extract all literal values from nested type structures (Union, Literal, etc.)

    Args:
        *type_hints: Type hints (can be nested Union, Literal, etc.)

    Returns:
        Set of all literal values
    """
    values = set()

    for type_hint in type_hints:
        # Skip None
        if type_hint is None:
            continue

        # Handle type(None)
        if type_hint is type(None):
            values.add(None)
            continue

        # Handle actual values (not type objects)
        if not isinstance(type_hint, type) and not hasattr(type_hint, "__origin__"):
            values.add(type_hint)
            continue

        # Check type origin (Union, Literal, etc.)
        origin = get_origin(type_hint)

        # Handle Literal types
        if origin is Literal:
            values.update(get_args(type_hint))

        # Handle Union types (typing.Union or | operator)
        elif origin is Union:
            for arg in get_args(type_hint):
                values.update(get_literal_values(arg))

        # Handle other complex types
        elif origin is not None:
            for arg in get_args(type_hint):
                values.update(get_literal_values(arg))

    return values
