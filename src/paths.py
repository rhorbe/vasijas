from __future__ import annotations
from pathlib import Path
from .config import project_root

def paths() -> dict[str, Path]:
    root = project_root()
    return {
        "root": root,
        "data_originales": root / "data" / "originales",
        "data_originales_corregidos": root / "data" / "corregidos",
        "data_clasificados": root / "data" / "clasificados",
        "data_mitades": root / "data" / "mitades",
        "data_processed": root / "data" / "processed",
        "models": root / "models",
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
        "tables": root / "reports" / "tables",
    }
