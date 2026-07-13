from dataclasses import replace
from pathlib import Path
from typing import Iterable

from vasijas.imagenes import (
    ResultadoCentrado,
    ResultadoCorteFracciones,
    ResultadoCorteMitades,
    cortar_imagenes_clasificadas_en_fracciones,
    cortar_imagenes_en_mitades,
)
from vasijas.metadata import (
    guardar_metadata,
    cargar_metadata,
    construir_metadata_mitades,
)


def cortar_dataset_clasificado_en_mitades(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoCorteMitades:
    resultado = cortar_imagenes_en_mitades(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        extension=extension,
        valor_fondo=valor_fondo,
    )

    metadata_original = cargar_metadata(ruta_origen)
    metadata_mitades = construir_metadata_mitades(
        metadata_original=metadata_original,
        registros_derivados=resultado.dataframe_registros,
    )
    guardar_metadata(df=metadata_mitades, carpeta_destino=ruta_destino)

    return replace(resultado, metadata_destino=metadata_mitades)


def cortar_dataset_clasificado_en_fracciones(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    porcentaje_avance: float,
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoCorteFracciones:
    resultado = cortar_imagenes_clasificadas_en_fracciones(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        porcentaje_avance=porcentaje_avance,
        extension=extension,
        valor_fondo=valor_fondo,
    )
    metadata_original = cargar_metadata(ruta_origen)
    metadata_mitades = construir_metadata_mitades(
        metadata_original=metadata_original,
        registros_mitades=resultado.dataframe_registros,
    )
    guardar_metadata(df=metadata_mitades, carpeta_destino=ruta_destino)
    ruta_metadata_destino = ruta_destino + nombre_metadata

    return replace(resultado, metadata_destino=ruta_metadata_destino)


def centrar_dataset_imagenes(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> ResultadoCentrado:
    pass
    # resultado = centrar_imagenes(
    #     ruta_origen=ruta_origen,
    #     ruta_destino=ruta_destino,
    #     clases=clases,
    #     extension=extension,
    #     tamano_salida=tamano_salida,
    #     valor_fondo=valor_fondo,
    # )
    # metadata_original = cargar_metadata(ruta_origen)

    # metadata_centrada = construir_metadata_centrados(
    #     metadata_original=metadata_original,
    #     registros_centrados=registros_centrados,
    #     columna_identificador=columna_identificador,
    #     how=how,
    # )
    # ruta_metadata_destino.parent.mkdir(parents=True, exist_ok=True)
    # metadata_centrada.to_csv(ruta_metadata_destino, index=False)
    #
    # ruta_metadata_destino = guardar_metadata_centrados(
    #     ruta_metadata_origen=ruta_origen / nombre_metadata,
    #     ruta_metadata_destino=ruta_destino / nombre_metadata,
    #     registros_centrados=resultado.registros,
    # )
    #
    # return replace(resultado, metadata_destino=ruta_metadata_destino)
