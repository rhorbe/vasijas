from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from vasijas.config import load_config


def ruta_relativa(ruta_destino: str | Path) -> Path:
    """Devuelve una ruta relativa a PROJECT_ROOT para mostrar en reportes.

    Las rutas relativas se interpretan respecto de la raíz del proyecto.
    Lanza ValueError si el destino queda fuera de esa raíz.
    """
    root = load_config(mount_colab_drive=False).project_root
    destino = Path(ruta_destino).expanduser()
    if not destino.is_absolute():
        destino = root / destino
    return destino.resolve().relative_to(root)


@dataclass(frozen=True)
class ProjectPaths:
    ROOT: Path
    DATA_ORIGINALES: Path
    DATA_CORREGIDOS: Path
    DATA_UBICADOS: Path
    DATA_CLASIFICADOS: Path
    DATA_CUADRADOS: Path
    DATA_MITADES: Path
    DATA_FRACCIONES: Path
    DATA_CENTRADOS: Path
    DATA_CUADRADOS_DIVIDIDO: Path
    DATA_UNIFICADO: Path
    DATA_UNIFICADO_DIVIDIDO: Path
    NOTEBOOKS: Path
    OUTPUT_MODELS: Path
    OUTPUT_REPORTS: Path

    @classmethod
    def from_root(cls, root: Path) -> Self:
        """Construye las rutas del proyecto sin modificar el filesystem."""
        root = root.expanduser().resolve()
        data = root / "data"
        output = root / "output"

        return cls(
            ROOT=root,
            DATA_ORIGINALES=data / "01.originales",
            DATA_CORREGIDOS=data / "02.corregidos",
            DATA_UBICADOS=data / "03.ubicados",
            DATA_CLASIFICADOS=data / "04.clasificados",
            DATA_CUADRADOS=data / "05.cuadrados",
            DATA_MITADES=data / "06.mitades",
            DATA_FRACCIONES=data / "07.fracciones",
            DATA_CENTRADOS=data / "08.centrados",
            DATA_CUADRADOS_DIVIDIDO=data / "10.cuadrados_divididos_en_train_val_y_test",
            DATA_UNIFICADO=data / "11.unificado",
            DATA_UNIFICADO_DIVIDIDO=data / "12.unificado_divididos_en_train_val_y_test",
            NOTEBOOKS=root / "notebooks",
            OUTPUT_MODELS=output / "models",
            OUTPUT_REPORTS=output / "reports",
        )

    def ensure_outputs(self) -> None:
        """Crea únicamente los directorios destinados a results generados."""
        self.OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


__all__ = ["ProjectPaths", "ruta_relativa"]
