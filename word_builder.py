"""Simple, explicit word buffer; letters are appended only on user action."""
from time import monotonic
from config import LETTERS

class WordBuilder:
    def __init__(self, cooldown_seconds: float = 1.0):
        self.characters: list[str] = []
        self.cooldown_seconds = cooldown_seconds
        self._last_accept = float("-inf")

    @property
    def text(self) -> str:
        return "".join(self.characters)

    def add_letter(self, letter: str, now: float | None = None) -> bool:
        letter = letter.upper().strip()
        if letter not in LETTERS:
            raise ValueError("Only A-Z can be added")
        now = monotonic() if now is None else now
        if now - self._last_accept < self.cooldown_seconds:
            return False
        self.characters.append(letter)
        self._last_accept = now
        return True

    def space(self):
        if self.characters and self.characters[-1] != " ":
            self.characters.append(" ")

    def delete(self):
        if self.characters:
            self.characters.pop()

    def clear(self):
        self.characters.clear()

    def save(self, path):
        from pathlib import Path
        Path(path).write_text(self.text, encoding="utf-8")
