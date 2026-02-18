# _bootstrap.py (en la raíz del repo)
from __future__ import annotations

import sys
from pathlib import Path

# Asegura que /src esté en sys.path para poder importar el paquete
SRC = (Path(__file__).resolve().parent.parent / "src").resolve()
sys_path_resolved = {Path(p).resolve() for p in sys.path}
if SRC not in sys_path_resolved:
    sys.path.insert(0, str(SRC))

from config import setup  # noqa: E402  (import después de tocar sys.path)

__all__ = ["setup"]
