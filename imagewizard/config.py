"""Paths and configuration.

Everything image-wizard writes lives under two roots:

* data dir (`~/Library/Application Support/image-wizard` on macOS) — the
  SQLite database, any persistent state.
* cache dir (`~/Library/Caches/image-wizard`) — thumbnails and downloaded
  model weights. Safe to delete.

Both can be overridden with env vars `IMAGEWIZARD_DATA_DIR` and
`IMAGEWIZARD_CACHE_DIR` for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir

APP_NAME = "image-wizard"


@dataclass(frozen=True)
class Config:
    data_dir: Path
    cache_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "imagewizard.sqlite"

    @property
    def thumbs_dir(self) -> Path:
        return self.cache_dir / "thumbs"

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    def ensure(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.thumbs_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


def load() -> Config:
    data = Path(os.environ.get("IMAGEWIZARD_DATA_DIR") or user_data_dir(APP_NAME))
    cache = Path(os.environ.get("IMAGEWIZARD_CACHE_DIR") or user_cache_dir(APP_NAME))
    cfg = Config(data_dir=data, cache_dir=cache)
    cfg.ensure()
    return cfg
