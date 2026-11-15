from pathlib import Path
from typing import TypeVar

import pandas as pd

from vasijas.nombres import es_formato_nombre_valido

T = TypeVar("T")


def get_ubicaciones(data: Path) -> list[str]:
    return sorted(item.name for item in data.iterdir() if item.is_dir())


def contar_ubicaciones(ubicaciones: list[str]) -> int:
    return len(ubicaciones)


def contar_perfiles_por_ubicacion(data: Path) -> tuple[int, list[tuple[str, int]]]:
    conteo_png = {
        ruta_ubicacion.name: len(list(ruta_ubicacion.glob("*.png")))
        for ruta_ubicacion in data.iterdir()
        if ruta_ubicacion.is_dir()
    }

    total_png = sum(conteo_png.values())
    conteo_png_ordenado = sorted(conteo_png.items(), key=lambda item: item[1])
    return total_png, conteo_png_ordenado


def get_nombres_archivos_perfiles(data: Path) -> set[str]:
    return {p.stem for p in data.rglob("*.png")}


def get_nombres_invalidos(nombres: set[str]) -> list[str]:
    return sorted(nombre for nombre in nombres if not es_formato_nombre_valido(nombre))


def contar_nombres_invalidos(nombres_invalidos: list[str]) -> int:
    return len(nombres_invalidos)


def agregar_marca_si_id_esta_en_fs(
    df: pd.DataFrame,
    nombres_archivos: set[str],
    columna_identificador: str = "Identificador",
) -> pd.DataFrame:
    df_resultado = df.copy()
    df_resultado["id_encontrado_en_fs"] = (
        df_resultado[columna_identificador]
        .astype(str)
        .str.strip()
        .isin(nombres_archivos)
    )

    return df_resultado


def eliminar_marca_id_esta_en_fs(df: pd.DataFrame) -> pd.DataFrame:
    df_resultado = df.copy()
    if "id_encontrado_en_fs" in df_resultado.columns:
        df_resultado.drop(columns=["id_encontrado_en_fs"], inplace=True)
    return df_resultado


def contar_identificadores_encontrados(df: pd.DataFrame) -> int:
    return int(df["id_encontrado_en_fs"].sum())


def contar_identificadores_no_encontrados(df: pd.DataFrame) -> int:
    return int((~df["id_encontrado_en_fs"]).sum())


def get_identificadores_no_encontrados(
    df: pd.DataFrame,
    columna_identificador: str = "Identificador",
) -> list[str]:
    ids_no_encontrados = set(
        df.loc[~df["id_encontrado_en_fs"], columna_identificador]
        .astype(str)
        .str.strip()
    )
    return sorted(ids_no_encontrados)


def contar_identificadores_unicos_no_encontrados(df: pd.DataFrame) -> int:
    return len(get_identificadores_no_encontrados(df))


def calcular_ancho_maximo(textos: list[str]) -> int:
    return max((len(texto) for texto in textos), default=0)


def agrupar_en_filas(
    elementos: list[T],
    cantidad_columnas: int = 3,
) -> list[list[T]]:
    """Agrupa los elementos en filas conservando su orden original.

    Args:
        elementos: Lista de elementos que se desea agrupar.
        cantidad_columnas: Cantidad máxima de elementos por fila. Por defecto, 3.

    Returns:
        Lista de filas. La última puede tener menos elementos que
        ``cantidad_columnas``. Si ``elementos`` está vacío, devuelve una lista vacía.

    Raises:
        ValueError: Si ``cantidad_columnas`` es menor que 1.

    Examples:
        >>> agrupar_en_filas([1, 2, 3, 4, 5], cantidad_columnas=2)
        [[1, 2], [3, 4], [5]]
    """
    if cantidad_columnas < 1:
        raise ValueError("cantidad_columnas debe ser mayor que cero")

    return [
        elementos[inicio : inicio + cantidad_columnas]
        for inicio in range(0, len(elementos), cantidad_columnas)
    ]


def mostrar_en_columnas(
    textos: list[str],
    cantidad_columnas: int = 5,
) -> None:
    """Imprime textos en columnas alineadas, conservando el orden por filas.

    El ancho se ajusta al texto más largo y las columnas se separan con dos
    espacios. Por defecto muestra cinco columnas. Una lista vacía no imprime
    nada; una cantidad de columnas menor que uno produce un ValueError.
    """
    filas = agrupar_en_filas(textos, cantidad_columnas=cantidad_columnas)
    ancho = calcular_ancho_maximo(textos)
    for fila in filas:
        print("  " + "  ".join(f"{texto:<{ancho}}" for texto in fila).rstrip())


def mostrar_duplas(duplas: list[tuple[str, T]]) -> None:
    """Imprime una dupla por fila, alineando la columna del segundo dato.

    El primer dato es un texto y el segundo puede ser, por ejemplo, un conteo.
    Conserva el orden recibido. Una lista vacía no imprime nada.
    """
    ancho = calcular_ancho_maximo([texto for texto, _ in duplas])
    for texto, valor in duplas:
        print(f"  {texto:<{ancho}} {valor}")


def get_archivos_sin_registro(
    nombres_archivos: set[str],
    ids_metadata: set[str],
) -> list[str]:
    return sorted(nombres_archivos - ids_metadata)


def contar_archivos_sin_registro(
    nombres_archivos: set[str],
    ids_metadata: set[str],
) -> int:
    return len(get_archivos_sin_registro(nombres_archivos, ids_metadata))


def corregir_identificadores(
    df: pd.DataFrame,
    correcciones: dict[str, str],
    columna_identificador: str = "Identificador",
) -> pd.DataFrame:
    df_resultado = df.copy()
    df_resultado[columna_identificador] = df_resultado[columna_identificador].replace(
        correcciones
    )
    return df_resultado
