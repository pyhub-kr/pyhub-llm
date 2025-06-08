from datetime import date, datetime
from decimal import Decimal
from json import JSONDecodeError
from json import JSONEncoder as BaseJSONEncoder
from json import dumps, loads
from typing import Any

from pyhub.llm.types import Embed, EmbedList


class JSONEncoder(BaseJSONEncoder):
    def default(self, o):
        if hasattr(o, "to_dict"):
            return o.to_dict()

        if isinstance(o, Embed):
            return o.array
        if isinstance(o, EmbedList):
            return [embed.array for embed in o.arrays]

        # Handle datetime and date objects
        if isinstance(o, (datetime, date)):
            return o.isoformat()

        # Handle Decimal
        if isinstance(o, Decimal):
            return float(o)

        return super().default(o)


def json_loads(s, **kwargs) -> Any:
    return loads(s, **kwargs)


def json_dumps(obj, **kwargs) -> str:
    kwargs.setdefault("ensure_ascii", False)
    return dumps(obj, cls=JSONEncoder, **kwargs)


__all__ = ["JSONDecodeError", "JSONEncoder", "json_loads", "json_dumps"]
