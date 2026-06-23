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
    DATA_UNIFICADO: Path
    MODELS: Path
    REPORTS: Path
    FIGURES: Path
    NOTEBOOKS: Path

def _build_structure(root: Path) -> dict[str, Path]:
    data = root / "data"
    notebooks = root / "notebooks"
    output = root / "output"

    return {
        "ROOT": root,
        "DATA_ORIGINALES": data / "01.originales",
        "DATA_CORREGIDOS": data / "02.corregidos",
        "DATA_UBICADOS": data / "03.ubicados",
        "DATA_CLASIFICADOS": data / "04.clasificados",
        "DATA_CUADRADOS": data / "05.cuadrados",
        "DATA_MITADES": data / "06.mitades",
        "DATA_FRACCIONES": data / "07.fracciones",
        "DATA_CENTRADOS": data / "08.centrados",
        "DATA_UNIFICADO": data / "09.unificado",
        "NOTEBOOKS": notebooks,
        "OUTPUT_MODELS": output / "models",
        "OUTPUT_REPORTS": output / "reports",
    }

def get_paths() -> ProjectPaths:
    root = _project_root()
    structure = _build_structure(root)
    _ensure_directories(structure)
    return ProjectPaths(**structure)

def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("No se pudo encontrar la raíz del proyecto")

def _ensure_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)



def setup():
    root = _project_root()
    structure = _build_structure(root)
    _ensure_directories(structure)
    return SimpleNamespace(**structure)

__all__ = ["setup"]
