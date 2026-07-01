import pandas as pd
from pathlib import Path

from vasijas.files import copiar_archivo


def cargar_metadata(
    carpeta_origen_data: Path, nombre_archivo="metadata.csv"
) -> pd.DataFrame:
    archivo_csv = carpeta_origen_data / nombre_archivo
    return pd.read_csv(archivo_csv)


def guardar_metadata(
    df: pd.DataFrame, carpeta_destino: Path, nombre_archivo: str = "metadata.csv"
) -> Path:
    ruta_destino = carpeta_destino / nombre_archivo
    df.to_csv(ruta_destino, index=False)
    return ruta_destino


def contar_registros_metadata(df: pd.DataFrame) -> int:
    return len(df)


def get_preview_metadata(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)


def get_identificadores_metadata(
    df: pd.DataFrame,
    columna_identificador: str = "Identificador",
) -> set[str]:
    return set(df[columna_identificador].astype(str).str.strip())


def copiar_metadata(
    ruta_origen: Path,
    ruta_destino: Path,
    nombre_archivo: str = "metadata.csv",
) -> Path:
    return copiar_archivo(
        origen=ruta_origen / nombre_archivo,
        destino=ruta_destino / nombre_archivo,
    )
