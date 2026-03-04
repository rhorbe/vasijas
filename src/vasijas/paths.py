from __future__ import annotations
from pathlib import Path
from .config import project_root


def paths() -> dict[str, Path]:
    root = project_root()
    data = root / "data"
    models = root / "models"
    reports = root / "reports"
    return {
        "root": root,
        "data": root / "data",
        "data_originales": data / "01.originales",
        "data_originales_corregidos": data / "corregidos",
        "data_clasificados": data / "clasificados",
        "data_mitades": data / "mitades",
        "models": models,
        "reports": reports,
        "figures": reports / "figures",
        "tables": root / "reports" / "tables",
    }
