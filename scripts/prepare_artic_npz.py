#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hohs_mano_regressor.data.artic_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())

