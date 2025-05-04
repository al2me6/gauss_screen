from dataclasses import dataclass
from typing import LiteralString


@dataclass
class AtomCoords:
    element: LiteralString
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"{self.element:8}{self.x:14.8f}{self.y:14.8f}{self.z:14.8f}"
