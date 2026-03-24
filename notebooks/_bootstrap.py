from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

@dataclass(frozen=True)
class ProjectPaths:
    ROOT: Path
    DATA: Path
    DATA_ORIGINALES: Path
    DATA_CORREGIDOS: Path
    DATA_CLASIFICADOS: Path
    DATA_MITADES: Path
    MODELS: Path
    REPORTS: Path
    FIGURES: Path
    NOTEBOOKS: Path


def get_paths() -> ProjectPaths:
    root = _project_root()
    structure = _build_structure(root)
    _ensure_directories(structure)
    return ProjectPaths(**structure)

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _ensure_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

def _build_structure(root: Path) -> dict[str, Path]:
    data = root / "data"
    reports = root / "reports"

    return {
        "ROOT": root,
        "DATA": data,
        "DATA_ORIGINALES": data / "01.originales",
        "DATA_CORREGIDOS": data / "02.corregidos",
        "DATA_UBICADOS": data / "03.ubicados",
        "DATA_CLASIFICADOS": data / "04.clasificados",
        "DATA_MITADES": data / "05.mitades",
        "DATA_FRACCIONES": data / "06.fracciones",
        "DATA_CENTRADOS": data / "07.centrados",
        "MODELS": root / "models",
        "REPORTS": reports,
        "FIGURES": reports / "figures",
        "NOTEBOOKS": root / "notebooks",
    }

def setup():
    root = _project_root()
    structure = _build_structure(root)
    _ensure_directories(structure)
    return SimpleNamespace(**structure)

__all__ = ["setup"]
