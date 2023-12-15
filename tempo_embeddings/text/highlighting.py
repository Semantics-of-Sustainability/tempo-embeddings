from dataclasses import dataclass


@dataclass(frozen=True)
class Highlighting:
    start: int
    end: int
