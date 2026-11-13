import re
import unicodedata
from pathlib import Path

PATTERN = r"^[A-Za-z]{2}_[0-9]{2}[A-Za-z0-9_]*$"


def es_formato_nombre_valido(cadena: str) -> bool:
    """
    Valida que una cadena cumpla con el formato especificado.

    Formato:
    - Comienza con 2 letras
    - Continúa con un guión bajo
    - Continúa con 2 dígitos
    - Seguido de letras, números y guiones bajos (sin espacios en blanco)

    Args:
        cadena: La cadena a validar

    Returns:
        True si la cadena cumple con el formato, False en caso contrario
    """
    return bool(re.match(PATTERN, cadena))


def normalizar_nombre(nombre: str) -> str:
    """
    Normaliza un nombre de archivo:
    - minúsculas
    - sin acentos
    - caracteres no alfanuméricos -> _
    - colapsa múltiples _
    - conserva extensión
    """

    path = Path(nombre)
    stem = path.stem.lower()
    suffix = path.suffix.lower()

    # Quitar acentos
    stem = unicodedata.normalize("NFKD", stem)
    stem = stem.encode("ascii", "ignore").decode("ascii")

    # Reemplazar caracteres no alfanuméricos por _
    stem = re.sub(r"[^a-z0-9]+", "_", stem)

    # Quitar múltiples _
    stem = re.sub(r"_+", "_", stem)

    # Quitar _ al inicio y final
    stem = stem.strip("_")

    # Insertar _ después de las 2 primeras letras si no existe
    if len(stem) > 2 and stem[2] != "_":
        stem = stem[:2] + "_" + stem[2:]

    return f"{stem}{suffix}"
