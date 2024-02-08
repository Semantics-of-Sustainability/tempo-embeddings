from dataclasses import dataclass


@dataclass(frozen=True)
class Highlighting:
    start: int
    end: int

    def get_span(self, stringify: str = False):
        if stringify:
            return f"{self.start}_{self.end}"
        return (self.start, self.end)
