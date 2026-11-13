from dataclasses import replace
from pathlib import Path
from typing import Iterable

from vasijas.imagenes import (
    ResultadoCentrado,
    ResultadoCorte,
    cortar_imagenes_clasificadas_en_fracciones,
    cortar_imagenes_en_mitades,
    centrar_imagenes,
)
from vasijas.metadata import (
    guardar_metadata,
    cargar_metadata,
    construir_metadata_mitades,
    construir_metadata_fracciones,
    construir_metadata_centrados,
)


def cortar_dataset_clasificado_en_mitades(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoCorte:
    resultado = cortar_imagenes_en_mitades(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        extension=extension,
        valor_fondo=valor_fondo,
    )

    metadata_original = cargar_metadata(ruta_origen, nombre_metadata)
    metadata_mitades = construir_metadata_mitades(
        metadata_original=metadata_original,
        registros_derivados=resultado.dataframe_registros,
    )
    ruta_metadata_destino = guardar_metadata(
        df=metadata_mitades,
        carpeta_destino=ruta_destino,
        nombre_archivo=nombre_metadata,
    )

    return replace(resultado, metadata_destino=ruta_metadata_destino)


def cortar_dataset_clasificado_en_fracciones(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    porcentaje_avance: float,
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoCorte:
    resultado = cortar_imagenes_clasificadas_en_fracciones(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        porcentaje_avance=porcentaje_avance,
        extension=extension,
        valor_fondo=valor_fondo,
    )
    metadata_original = cargar_metadata(ruta_origen, nombre_metadata)
    metadata_fracciones = construir_metadata_fracciones(
        metadata_original=metadata_original,
        registros_derivados=resultado.dataframe_registros,
    )
    ruta_metadata_destino = guardar_metadata(
        df=metadata_fracciones,
        carpeta_destino=ruta_destino,
        nombre_archivo=nombre_metadata,
    )

    return replace(resultado, metadata_destino=ruta_metadata_destino)


def centrar_dataset_clasificado_de_fracciones(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> ResultadoCentrado:
    resultado = centrar_imagenes(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        extension=extension,
        tamano_salida=tamano_salida,
        valor_fondo=valor_fondo,
    )
    metadata_original = cargar_metadata(ruta_origen)

    metadata_centrada = construir_metadata_centrados(
        metadata_original=metadata_original,
        registros_centrados=resultado.registros,
    )
    ruta_metadata_destino = guardar_metadata(
        df=metadata_centrada,
        carpeta_destino=ruta_destino,
    )

    return replace(resultado, metadata_destino=ruta_metadata_destino)
