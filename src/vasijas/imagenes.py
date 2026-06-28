from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from vasijas.files import (
    copiar_metadata,
    obtener_nombre_carpeta_clase,
    obtener_o_crear_directorio,
)


class OrientacionCorte(StrEnum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@dataclass(frozen=True)
class ErrorConversionImagen:
    ruta_origen: Path
    mensaje: str


@dataclass(frozen=True)
class ResultadoConversionCuadrados:
    imagenes_procesadas: int
    rutas_destino: list[Path]
    clases_sin_imagenes: list[str]
    errores: list[ErrorConversionImagen]
    metadata_destino: Path | None = None

    @property
    def imagenes_con_error(self) -> int:
        return len(self.errores)

    @property
    def mensaje_resumen(self) -> str:
        partes = [
            f"Imagenes procesadas: {self.imagenes_procesadas}",
            f"Clases sin imagenes: {len(self.clases_sin_imagenes)}",
            f"Errores: {self.imagenes_con_error}",
        ]
        if self.metadata_destino is not None:
            partes.append(f"Metadata copiada: {self.metadata_destino}")
        return " | ".join(partes)


def leer_imagen_grises(ruta_imagen: Path) -> np.ndarray:
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")
    return imagen


def guardar_imagen(ruta_destino: Path, imagen: np.ndarray) -> Path:
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    guardada = cv2.imwrite(str(ruta_destino), imagen)

    if not guardada:
        raise OSError(f"No se pudo guardar la imagen: {ruta_destino}")

    return ruta_destino


def obtener_nombre_archivo(ruta: Path) -> str:
    return ruta.stem


def crear_ruta_archivo(
    ruta_base: Path,
    nombre_archivo: str,
    sufijo: str = "",
    extension: str = "png",
) -> Path:
    extension_normalizada = extension.lstrip(".")
    return ruta_base / f"{nombre_archivo}{sufijo}.{extension_normalizada}"


def crear_lienzo_cuadrado(
    imagen: np.ndarray,
    lado: int,
    valor_fondo: int = 255,
) -> np.ndarray:
    return np.full((lado, lado), valor_fondo, dtype=imagen.dtype)


def cuadrar_imagen(imagen: np.ndarray, valor_fondo: int = 255) -> np.ndarray:
    if imagen.ndim != 2:
        raise ValueError(
            "La conversion a cuadrado espera una imagen en escala de grises"
        )

    alto, ancho = imagen.shape
    lado = max(alto, ancho)
    offset_y = (lado - alto) // 2
    offset_x = (lado - ancho) // 2

    lienzo = crear_lienzo_cuadrado(imagen, lado, valor_fondo)
    lienzo[offset_y : offset_y + alto, offset_x : offset_x + ancho] = imagen

    return lienzo


def determinar_orientacion_corte(clase: int | str) -> OrientacionCorte:
    try:
        numero_clase = int(float(str(clase)))
    except ValueError:
        return OrientacionCorte.HORIZONTAL

    if 1 <= numero_clase <= 9:
        return OrientacionCorte.VERTICAL

    return OrientacionCorte.HORIZONTAL


def obtener_cortes_fuente(
    alto: int,
    ancho: int,
    orientacion_corte: OrientacionCorte,
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    if orientacion_corte == OrientacionCorte.VERTICAL:
        return (
            (slice(0, alto // 2), slice(0, ancho)),
            (slice(alto // 2, alto), slice(0, ancho)),
        )

    return (
        (slice(0, alto), slice(0, ancho // 2)),
        (slice(0, alto), slice(ancho // 2, ancho)),
    )


def trasladar_slice(segmento: slice, offset: int, limite: int) -> slice:
    inicio = offset if segmento.start is None else segmento.start + offset
    fin = limite if segmento.stop is None else segmento.stop + offset
    return slice(inicio, fin)


def trasladar_corte_a_lienzo(
    corte_src: tuple[slice, slice],
    offset_y: int,
    offset_x: int,
    lado_lienzo: int,
) -> tuple[slice, slice]:
    slice_y, slice_x = corte_src

    return (
        trasladar_slice(slice_y, offset_y, lado_lienzo),
        trasladar_slice(slice_x, offset_x, lado_lienzo),
    )


def obtener_rutas_perfiles(
    ruta_origen: Path,
    clase: int | str,
    extension: str = "png",
) -> list[Path]:
    nombre_carpeta = obtener_nombre_carpeta_clase(clase)
    carpeta_clase = ruta_origen / nombre_carpeta
    return sorted(carpeta_clase.glob(f"*.{extension.lstrip('.')}"))


def convertir_perfil_a_cuadrado(
    ruta_perfil: Path,
    ruta_destino_clase: Path,
    valor_fondo: int = 255,
) -> Path:
    imagen = leer_imagen_grises(ruta_perfil)
    imagen_cuadrada = cuadrar_imagen(imagen, valor_fondo)
    nombre_original = obtener_nombre_archivo(ruta_perfil)
    ruta_destino = crear_ruta_archivo(ruta_destino_clase, nombre_original)

    return guardar_imagen(ruta_destino, imagen_cuadrada)


def convertir_clase_a_cuadrados(
    ruta_origen: Path,
    ruta_destino: Path,
    clase: int | str,
    extension: str = "png",
    valor_fondo: int = 255,
) -> tuple[list[Path], list[ErrorConversionImagen]]:
    perfiles = obtener_rutas_perfiles(ruta_origen, clase, extension)
    if not perfiles:
        return [], []

    ruta_destino_clase = obtener_o_crear_directorio(
        ruta_destino,
        obtener_nombre_carpeta_clase(clase),
    )
    rutas_destino: list[Path] = []
    errores: list[ErrorConversionImagen] = []

    for ruta_perfil in perfiles:
        try:
            rutas_destino.append(
                convertir_perfil_a_cuadrado(
                    ruta_perfil=ruta_perfil,
                    ruta_destino_clase=ruta_destino_clase,
                    valor_fondo=valor_fondo,
                )
            )
        except Exception as exc:
            errores.append(ErrorConversionImagen(ruta_perfil, str(exc)))

    return rutas_destino, errores


def convertir_imagenes_clasificadas_a_cuadradas(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoConversionCuadrados:
    rutas_destino: list[Path] = []
    errores: list[ErrorConversionImagen] = []
    clases_sin_imagenes: list[str] = []

    for clase in clases:
        rutas_clase, errores_clase = convertir_clase_a_cuadrados(
            ruta_origen=ruta_origen,
            ruta_destino=ruta_destino,
            clase=clase,
            extension=extension,
            valor_fondo=valor_fondo,
        )

        if not rutas_clase and not errores_clase:
            clases_sin_imagenes.append(obtener_nombre_carpeta_clase(clase))

        rutas_destino.extend(rutas_clase)
        errores.extend(errores_clase)

    return ResultadoConversionCuadrados(
        imagenes_procesadas=len(rutas_destino),
        rutas_destino=rutas_destino,
        clases_sin_imagenes=clases_sin_imagenes,
        errores=errores,
    )


def convertir_dataset_clasificado_a_cuadrado(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoConversionCuadrados:
    resultado = convertir_imagenes_clasificadas_a_cuadradas(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        extension=extension,
        valor_fondo=valor_fondo,
    )
    metadata_destino = copiar_metadata(ruta_origen, ruta_destino, nombre_metadata)

    return replace(resultado, metadata_destino=metadata_destino)
