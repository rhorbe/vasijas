from pathlib import Path

import pandas as pd

from vasijas.files import es_formato_nombre_valido


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


def cargar_metadata(data_originales: Path) -> pd.DataFrame:
    archivo_csv = data_originales / "metadata.csv"
    return pd.read_csv(archivo_csv)


def contar_registros_metadata(df: pd.DataFrame) -> int:
    return len(df)


def get_preview_metadata(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)


def agregar_marca_identificador_en_fs(
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


def get_identificadores_metadata(
    df: pd.DataFrame,
    columna_identificador: str = "Identificador",
) -> set[str]:
    return set(df[columna_identificador].astype(str).str.strip())


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
