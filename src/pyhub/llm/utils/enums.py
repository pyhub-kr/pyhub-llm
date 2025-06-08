"""Custom Enum classes to replace Django's TextChoices"""

from enum import Enum


class StrEnum(str, Enum):
    """String-based Enum that mimics Django's TextChoices behavior"""

    def __str__(self):
        return self.value

    @classmethod
    def values(cls):
        """Return all values (for compatibility with TextChoices)"""
        return [member.value for member in cls]

    @classmethod
    def choices(cls):
        """Return choices as tuples (for compatibility with TextChoices)"""
        return [(member.value, member.name) for member in cls]


class TextChoices(StrEnum):
    """Compatibility class for Django's TextChoices"""

    def __new__(cls, value, label=None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._label = label or value
        return obj

    @property
    def label(self):
        return getattr(self, "_label", self.value)
