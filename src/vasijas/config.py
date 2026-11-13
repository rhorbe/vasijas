from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    is_colab: bool


def is_colab() -> bool:
    return os.environ.get("COLAB_GPU") is not None or Path("/content").exists()


def mount_drive_if_colab() -> None:
    """Monta Google Drive cuando la ejecución tiene lugar en Colab."""
    if not is_colab():
        return

    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "El entorno parece ser Colab, pero google.colab no está disponible."
        ) from exc

    drive.mount("/content/drive")


def load_env() -> None:
    """Carga las variables del archivo .env, si existe."""
    from dotenv import load_dotenv

    load_dotenv()


def _project_root_from_env() -> Path:
    value = os.getenv("PROJECT_ROOT")
    if not value:
        raise RuntimeError(
            "No está definida la variable PROJECT_ROOT. "
            "Creá un .env a partir de .env.example y definila."
        )

    root = Path(value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def load_config(*, mount_colab_drive: bool = True) -> RuntimeConfig:
    """Prepara el entorno y devuelve la configuración de ejecución."""
    running_in_colab = is_colab()
    if mount_colab_drive and running_in_colab:
        mount_drive_if_colab()

    load_env()
    return RuntimeConfig(
        project_root=_project_root_from_env(),
        is_colab=running_in_colab,
    )


__all__ = [
    "RuntimeConfig",
    "is_colab",
    "load_config",
    "load_env",
    "mount_drive_if_colab",
]
