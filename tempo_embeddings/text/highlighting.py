from dataclasses import dataclass


@dataclass(frozen=True)
class Highlighting:
    start: int
    end: int

    def __str__(self):
        return f"{self.start}_{self.end}"
