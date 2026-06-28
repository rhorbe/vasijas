from pathlib import Path
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def cargar_metadata(
    ruta_base: Path,
    nombre_archivo: str = "metadata.csv",
) -> "pd.DataFrame":
    """Carga el archivo de metadata desde una carpeta base."""
    import pandas as pd

    return pd.read_csv(ruta_base / nombre_archivo)


def guardar_metadata(
    df: "pd.DataFrame",
    ruta_base: Path,
    nombre_archivo: str = "metadata.csv",
) -> tuple[Path, str]:
    """Guarda el dataset de metadata en CSV sin índice."""
    ruta_salida = ruta_base / nombre_archivo
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_salida, index=False)

    return ruta_salida, f"Archivo generado: {ruta_salida}"


def agregar_ubicacion(
    df: "pd.DataFrame",
    archivo_ubicacion: dict[str, str],
    columna_identificador: str = "Identificador",
    columna_ubicacion: str = "Ubicacion",
) -> tuple["pd.DataFrame", str]:
    """
    Agrega la ubicación al dataset a partir del identificador de archivo.

    Devuelve una copia del DataFrame y un mensaje con el resultado de la validación.
    """
    df_ubicado = df.copy()
    df_ubicado[columna_ubicacion] = (
        df_ubicado[columna_identificador].astype(str).map(archivo_ubicacion)
    )

    no_encontrados = df_ubicado.loc[
        df_ubicado[columna_ubicacion].isna(), columna_identificador
    ].tolist()
    if no_encontrados:
        mensaje = (
            f"{len(no_encontrados)} identificadores sin ubicación: {no_encontrados}"
        )
    else:
        mensaje = "Todos los identificadores tienen ubicación asignada."

    return df_ubicado, mensaje


def copiar_estructura_archivos(
    ruta_origen: Path,
    ruta_destino: Path,
) -> tuple[list[Path], str]:
    """
    Copia todos los archivos de una estructura preservando rutas relativas.

    Devuelve las rutas destino copiadas y un mensaje de resumen.
    """
    archivos_copiados: list[Path] = []

    for archivo in sorted(ruta_origen.rglob("*")):
        if not archivo.is_file():
            continue

        ruta_relativa = archivo.relative_to(ruta_origen)
        destino = ruta_destino / ruta_relativa
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archivo, destino)
        archivos_copiados.append(destino)

    mensaje = (
        f"{len(archivos_copiados)} archivos copiados desde "
        f"{ruta_origen} hacia {ruta_destino}"
    )
    return archivos_copiados, mensaje


def mapear_archivos_a_ubicacion(
    ruta_base: Path,
    patron: str = "*.png",
) -> dict[str, str]:
    """
    Construye un mapa entre el nombre base de cada archivo y su ubicación.

    La ubicación corresponde al nombre de la carpeta padre del archivo.
    """
    return {
        archivo.stem: archivo.parent.name for archivo in sorted(ruta_base.rglob(patron))
    }
