"""Engineering-side blinding wrapper.

Until Stage 4 validation run, all Arcade().make() calls during development
must go through this wrapper. It raises HeldOutGameError if a caller
touches any game outside the Stage 1 selection set.

See testing-plan.md §7 measurement validity.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

from arc_agi import Arcade, EnvironmentWrapper


class HeldOutGameError(RuntimeError):
    """Raised when code tries to touch a held-out game during development."""


class BlindedArcade:
    """Thin facade over Arcade that enforces the held-out set.

    Every make() call is also appended to a session log so we can audit
    which game IDs were touched during development.
    """

    def __init__(
        self,
        arcade: Arcade,
        allowed_games: Iterable[str],
        access_log: str | os.PathLike = "logs/game_access.log",
    ) -> None:
        self._arcade = arcade
        self._allowed = {g for g in allowed_games}
        self._log_path = Path(access_log)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def make(self, game_id: str, **kw) -> EnvironmentWrapper:
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{game_id}\n")
        if game_id not in self._allowed:
            raise HeldOutGameError(
                f"Refusing to make() held-out game {game_id!r}. "
                f"Allowed set: {sorted(self._allowed)}."
            )
        return self._arcade.make(game_id, **kw)

    # passthroughs we still allow during development
    def get_environments(self):
        return self._arcade.get_environments()
