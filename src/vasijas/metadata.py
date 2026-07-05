import pandas as pd
from pathlib import Path

from vasijas.archivos import copiar_archivo
from vasijas.nombres import normalizar_nombre


COLUMNA_IDENTIFICADOR = "Identificador"
COLUMNAS_METADATA_DERIVADA = ["archivo_origen", "mitad", "porcentaje"]


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
    columna_identificador: str = COLUMNA_IDENTIFICADOR,
) -> set[str]:
    return set(df[columna_identificador].astype(str).str.strip())


def get_columnas_adicionales_metadata(
    df_base: pd.DataFrame,
    df_referencia: pd.DataFrame,
) -> list[str]:
    """
    Devuelve las columnas presentes en la metadata de referencia y ausentes en la base.
    """
    return [
        columna for columna in df_referencia.columns if columna not in df_base.columns
    ]


def completar_metadata_cuadrados(
    df_metadata_cuadrados: pd.DataFrame,
    df_metadata_referencia: pd.DataFrame,
    columna_identificador: str = COLUMNA_IDENTIFICADOR,
    columnas_derivadas: list[str] | None = None,
    porcentaje: float = 100.00,
) -> pd.DataFrame:
    """
    Completa la metadata de imagenes cuadradas con las columnas derivadas.

    Las columnas adicionales se toman de la metadata de referencia para conservar el
    mismo orden. Para cuadrados, archivo_origen coincide con Identificador, mitad queda
    vacia y porcentaje representa el 100% de la imagen.
    """
    columnas_adicionales = columnas_derivadas or get_columnas_adicionales_metadata(
        df_metadata_cuadrados,
        df_metadata_referencia,
    )

    df_resultado = df_metadata_cuadrados.copy()
    valores_por_defecto = {
        "archivo_origen": df_resultado[columna_identificador],
        "mitad": "",
        "porcentaje": porcentaje,
    }

    columnas_no_soportadas = [
        columna
        for columna in columnas_adicionales
        if columna not in valores_por_defecto
    ]
    if columnas_no_soportadas:
        raise ValueError(
            "No hay valor por defecto definido para las columnas: "
            + ", ".join(columnas_no_soportadas)
        )

    for columna in columnas_adicionales:
        df_resultado[columna] = valores_por_defecto[columna]

    return df_resultado[list(df_metadata_cuadrados.columns) + columnas_adicionales]


def copiar_metadata(
    ruta_origen: Path,
    ruta_destino: Path,
    nombre_archivo: str = "metadata.csv",
) -> Path:
    return copiar_archivo(
        origen=ruta_origen / nombre_archivo,
        destino=ruta_destino / nombre_archivo,
    )


def normalizar_identificadores(
    df: pd.DataFrame,
    columna_identificador: str = COLUMNA_IDENTIFICADOR,
) -> pd.DataFrame:
    """
    Normaliza la columna de identificadores sin modificar el DataFrame original.
    """
    df_resultado = df.copy()
    df_resultado[columna_identificador] = (
        df_resultado[columna_identificador].astype(str).apply(normalizar_nombre)
    )
    return df_resultado


def cargar_metadata_con_identificadores_normalizados(
    ruta_csv: Path,
    columna_identificador: str = COLUMNA_IDENTIFICADOR,
) -> pd.DataFrame:
    """
    Carga metadata desde CSV y normaliza la columna de identificadores.
    """
    df = pd.read_csv(ruta_csv)
    return normalizar_identificadores(df, columna_identificador)
