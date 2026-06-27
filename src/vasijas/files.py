import re
import shutil
import unicodedata
from pathlib import Path

PATTERN = r"^[A-Za-z0-9_-]+$"


def es_formato_nombre_valido(cadena: str) -> bool:
    """
    Valida que una cadena cumpla con el formato especificado.

    Formato:
    - Comienza con 2 letras en mayúscula
    - Continúa con 2 dígitos
    - Un guión bajo
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


def copiar_png_normalizados_por_ubicacion(
    ruta_origen: Path,
    ruta_destino: Path,
) -> list[Path]:
    """
    Copia archivos PNG desde subdirectorios de origen hacia destino,
    preservando cada ubicación y normalizando los nombres de archivo.
    """
    destinos: list[Path] = []

    rutas_ubicaciones = [ruta for ruta in ruta_origen.iterdir() if ruta.is_dir()]

    for ruta_ubicacion in rutas_ubicaciones:
        nombre_ubicacion = ruta_ubicacion.name

        ruta_destino_ubicacion = ruta_destino / nombre_ubicacion
        ruta_destino_ubicacion.mkdir(parents=True, exist_ok=True)

        archivos_png = list(ruta_ubicacion.glob("*.png"))

        for archivo in archivos_png:
            nombre_normalizado = normalizar_nombre(archivo.name)
            destino = ruta_destino_ubicacion / nombre_normalizado
            shutil.copy2(archivo, destino)
            destinos.append(destino)

    return destinos
