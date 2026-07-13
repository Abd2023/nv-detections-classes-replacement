from __future__ import annotations

from typing import Any


class PackageInitPlaceholderExecutor:
    """Temporary executor used only to verify the package scaffold."""

    def __init__(self, request: Any | None = None, bootstrap: dict[str, Any] | None = None) -> None:
        self.request = request
        self.bootstrap_data = bootstrap or {}

    @staticmethod
    def bootstrap() -> dict[str, str]:
        return {"executor": "PackageInitPlaceholderExecutor", "status": "ready"}

    def run(self) -> dict[str, str]:
        return {"status": "package scaffold ready"}

