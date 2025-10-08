from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
def pytest_configure(config):
    mutant = os.environ.get("MUTANT_UNDER_TEST")
    if mutant:
        plugin = config.pluginmanager.get_plugin("_cov")
        if plugin is not None:
            config.pluginmanager.unregister(plugin)
