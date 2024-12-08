from typing_extensions import Protocol


class DataclassProtocol(Protocol):
    __dataclass_fields__: dict


__all__ = ["DataclassProtocol"]
