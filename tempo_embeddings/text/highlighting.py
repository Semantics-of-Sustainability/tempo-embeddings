from dataclasses import dataclass


@dataclass(frozen=True)
class Highlighting:
    start: int
    end: int

    def __str__(self):
        return f"{self.start}_{self.end}"

    @classmethod
    def from_string(cls, string: str) -> "Highlighting":
        """Create a Highlighting object from a string representation.

        Expects a string in the format "start_end".

        Args:
            string: The string representation.
        """
        start, end = string.split("_")
        return cls(int(start), int(end))
