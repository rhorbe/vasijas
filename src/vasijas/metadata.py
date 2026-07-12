import pandas as pd
from pathlib import Path
from typing import Iterable, Protocol

from vasijas.archivos import copiar_archivo
from vasijas.nombres import normalizar_nombre

COLUMNA_IDENTIFICADOR = "Identificador"
COLUMNAS_METADATA_DERIVADA = ["archivo_origen", "mitad", "porcentaje"]
COLUMNAS_REGISTRO_CENTRADO = ["archivo_origen", "archivo_generado"]


class RegistroCentradoLike(Protocol):
    archivo_origen: str
    archivo_generado: str


def cargar_metadata(
        carpeta_origen_data: Path, nombre_archivo="metadata.csv"
) -> pd.DataFrame:
    archivo_csv = carpeta_origen_data / nombre_archivo
    return pd.read_csv(archivo_csv)


def guardar_metadata(
        df: pd.DataFrame, carpeta_destino: Path, nombre_archivo: str = "metadata.csv"
) -> Path:
    ruta_destino = carpeta_destino / nombre_archivo
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
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


def construir_metadata_mitades(
        metadata_original: pd.DataFrame,
        registros_derivados: pd.DataFrame,
        columna_identificador: str = COLUMNA_IDENTIFICADOR,
        how: str = "inner",
) -> pd.DataFrame:
    columnas_registros = ["archivo_origen", "archivo_generado", "mitad"]
    if "porcentaje" in registros_derivados.columns:
        columnas_registros.append("porcentaje")

    derivados = registros_derivados[columnas_registros].copy()
    derivados["archivo_origen_stem"] = derivados["archivo_origen"].map(
        lambda nombre: Path(nombre).stem
    )

    metadata_derivada = derivados.merge(
        metadata_original,
        left_on="archivo_origen_stem",
        right_on=columna_identificador,
        how=how,
        validate="many_to_one",
    )
    metadata_derivada[columna_identificador] = metadata_derivada[
        "archivo_generado"
    ].map(lambda nombre: Path(nombre).stem)

    columnas_finales = list(metadata_original.columns) + ["archivo_origen", "mitad"]
    if "porcentaje" in metadata_derivada.columns:
        columnas_finales.append("porcentaje")

    return metadata_derivada[columnas_finales]


def construir_metadata_centrados(
        metadata_original: pd.DataFrame,
        registros_centrados: pd.DataFrame | Iterable[RegistroCentradoLike],
        columna_identificador: str = COLUMNA_IDENTIFICADOR,
        how: str = "inner",
) -> pd.DataFrame:
    """Replica la metadata original con identificadores de imagenes centradas."""
    if not isinstance(registros_centrados, pd.DataFrame):
        registros_centrados = pd.DataFrame(
            [
                {
                    "archivo_origen": registro.archivo_origen,
                    "archivo_generado": registro.archivo_generado,
                }
                for registro in registros_centrados
            ],
            columns=COLUMNAS_REGISTRO_CENTRADO,
        )

    registros = registros_centrados[COLUMNAS_REGISTRO_CENTRADO].copy()
    if registros.empty:
        return metadata_original.iloc[0:0].copy()

    registros["_identificador_original"] = registros["archivo_origen"].map(
        lambda nombre: Path(nombre).stem
    )
    registros["_identificador_centrado"] = registros["archivo_generado"].map(
        lambda nombre: Path(nombre).stem
    )

    metadata_centrada = metadata_original.merge(
        registros[["_identificador_original", "_identificador_centrado"]],
        left_on=columna_identificador,
        right_on="_identificador_original",
        how=how,
        validate="one_to_one",
    )
    metadata_centrada[columna_identificador] = metadata_centrada[
        "_identificador_centrado"
    ]

    return metadata_centrada[list(metadata_original.columns)]


def guardar_metadata_centrados(
        ruta_metadata_origen: Path,
        ruta_metadata_destino: Path,
        registros_centrados: pd.DataFrame | Iterable[RegistroCentradoLike],
        columna_identificador: str = COLUMNA_IDENTIFICADOR,
        how: str = "inner",
) -> Path:
    """Crea el CSV de metadata para imagenes ya centradas."""
    metadata_original = pd.read_csv(ruta_metadata_origen)
    metadata_centrada = construir_metadata_centrados(
        metadata_original=metadata_original,
        registros_centrados=registros_centrados,
        columna_identificador=columna_identificador,
        how=how,
    )
    ruta_metadata_destino.parent.mkdir(parents=True, exist_ok=True)
    metadata_centrada.to_csv(ruta_metadata_destino, index=False)

    return ruta_metadata_destino


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
