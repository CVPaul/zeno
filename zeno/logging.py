from __future__ import annotations

import sys
from collections.abc import Callable


VerboseLogger = Callable[[str], None]


def verbose_logger(enabled: bool) -> VerboseLogger | None:
    if not enabled:
        return None

    def log(message: str) -> None:
        print(f"[zeno] {message}", file=sys.stderr)

    return log
