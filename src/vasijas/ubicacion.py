from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
