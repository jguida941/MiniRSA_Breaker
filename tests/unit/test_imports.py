from __future__ import annotations

import importlib
from typing import Iterable

PUBLIC_MODULES: Iterable[str] = (
    "mini_rsa",
    "mini_rsa.core",
    "rsa",
)


def test_import_public_modules() -> None:
    for module_name in PUBLIC_MODULES:
        importlib.import_module(module_name)
