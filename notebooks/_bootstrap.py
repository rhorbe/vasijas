# _bootstrap.py (en la raíz del repo)
from __future__ import annotations

import sys
from pathlib import Path

# Asegura que /src esté en sys.path para poder importar el paquete
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import setup  # noqa: E402  (import después de tocar sys.path)

__all__ = ["setup"]
