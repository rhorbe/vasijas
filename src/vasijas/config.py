from __future__ import annotations

import os
from pathlib import Path

def is_colab() -> bool:
    return os.environ.get("COLAB_GPU") is not None or os.path.exists("/content")

def mount_drive_if_colab() -> None:
    """
    Monta Google Drive solo si estás en Colab.
    En local no hace nada.
    """
    if is_colab():
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        except Exception as e:
            raise RuntimeError(
                "Parece Colab, pero no se pudo montar Drive. "
                "¿Estás ejecutando realmente en un notebook de Colab?"
            ) from e

def load_env() -> None:
    """
    Carga variables desde .env si existe.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

def project_root() -> Path:
    """
    Devuelve la raíz del proyecto a partir de PROJECT_ROOT (en .env o variables de entorno).
    """
    root = os.getenv("PROJECT_ROOT")
    if not root:
        raise RuntimeError(
            "No está definida la variable PROJECT_ROOT. "
            "Creá un .env (a partir de .env.example) y definila."
        )
    return Path(root)

def setup() -> Path:
    """
    Setup completo: monta drive si corresponde, carga .env y devuelve PROJECT_ROOT.
    """
    mount_drive_if_colab()
    load_env()
    return project_root()
