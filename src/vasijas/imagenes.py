from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Iterable, TypeAlias

import cv2
import numpy as np
import pandas as pd

from vasijas.archivos import (
    obtener_nombre_carpeta_clase,
    obtener_o_crear_directorio,
)

from vasijas.metadata import copiar_metadata


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


@dataclass(frozen=True)
class RegistroMitad:
    archivo_origen: str
    archivo_generado: str
    mitad: str


@dataclass(frozen=True)
class RegistroFraccion:
    archivo_origen: str
    archivo_generado: str
    mitad: str
    porcentaje: float


RegistroCorte: TypeAlias = RegistroMitad | RegistroFraccion


@dataclass(frozen=True)
class ResultadoCorte:
    imagenes_origen_procesadas: int
    cortes_generados: int
    registros: list[RegistroCorte]
    clases_sin_imagenes: list[str]
    errores: list[ErrorConversionImagen]
    nombre_cortes: str
    metadata_destino: Path | None = None

    @property
    def imagenes_con_error(self) -> int:
        return len(self.errores)

    @property
    def mitades_generadas(self) -> int:
        return self.cortes_generados

    @property
    def fracciones_generadas(self) -> int:
        return self.cortes_generados

    @property
    def dataframe_registros(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    {
                        "id_perfil_completo": Path(registro.archivo_origen).stem,
                        "archivo_generado": registro.archivo_generado,
                        "borde_origen": registro.mitad,
                    }
                    | (
                        {"porcentaje": registro.porcentaje}
                        if hasattr(registro, "porcentaje")
                        else {}
                    )
                )
                for registro in self.registros
            ]
        )

    @property
    def mensaje_resumen(self) -> str:
        partes = [
            f"Imagenes origen procesadas: {self.imagenes_origen_procesadas}",
            f"{self.nombre_cortes} generadas: {self.cortes_generados}",
            f"Clases sin imagenes: {len(self.clases_sin_imagenes)}",
            f"Errores: {self.imagenes_con_error}",
        ]
        if self.metadata_destino is not None:
            partes.append(f"Metadata generada: {self.metadata_destino}")
        return " | ".join(partes)


ResultadoCorteMitades: TypeAlias = ResultadoCorte
ResultadoCorteFracciones: TypeAlias = ResultadoCorte


@dataclass(frozen=True)
class RegistroCentrado:
    archivo_origen: str
    archivo_generado: str


COLUMNAS_REGISTRO_CENTRADO = ["archivo_origen", "archivo_generado"]


@dataclass(frozen=True)
class ResultadoCentrado:
    imagenes_procesadas: int
    registros: list[RegistroCentrado]
    clases_sin_imagenes: list[str]
    errores: list[ErrorConversionImagen]
    metadata_destino: Path | None = None

    @property
    def imagenes_con_error(self) -> int:
        return len(self.errores)

    @property
    def dataframe_registros(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "archivo_origen": registro.archivo_origen,
                    "archivo_generado": registro.archivo_generado,
                }
                for registro in self.registros
            ],
            columns=COLUMNAS_REGISTRO_CENTRADO,
        )

    @property
    def mensaje_resumen(self) -> str:
        partes = [
            f"Imagenes procesadas: {self.imagenes_procesadas}",
            f"Clases sin imagenes: {len(self.clases_sin_imagenes)}",
            f"Errores: {self.imagenes_con_error}",
        ]
        if self.metadata_destino is not None:
            partes.append(f"Metadata generada: {self.metadata_destino}")
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


def cortar_imagen_en_mitades(
    imagen: np.ndarray,
    orientacion_corte: OrientacionCorte,
    valor_fondo: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    if imagen.ndim != 2:
        raise ValueError("El corte en mitades espera una imagen en escala de grises")

    alto, ancho = imagen.shape
    lado_lienzo = max(alto, ancho)
    offset_y = (lado_lienzo - alto) // 2
    offset_x = (lado_lienzo - ancho) // 2
    corte_tope_src, corte_base_src = obtener_cortes_fuente(
        alto,
        ancho,
        orientacion_corte,
    )

    imagen_tope = crear_lienzo_cuadrado(imagen, lado_lienzo, valor_fondo)
    imagen_base = crear_lienzo_cuadrado(imagen, lado_lienzo, valor_fondo)
    corte_tope_dst = trasladar_corte_a_lienzo(
        corte_tope_src,
        offset_y,
        offset_x,
        lado_lienzo,
    )
    corte_base_dst = trasladar_corte_a_lienzo(
        corte_base_src,
        offset_y,
        offset_x,
        lado_lienzo,
    )

    imagen_tope[corte_tope_dst] = imagen[corte_tope_src]
    imagen_base[corte_base_dst] = imagen[corte_base_src]

    return imagen_tope, imagen_base


def obtener_porcentajes_fraccion(porcentaje_avance: float) -> list[float]:
    if porcentaje_avance <= 0 or porcentaje_avance >= 100:
        raise ValueError("porcentaje_avance debe estar entre 0 y 100")

    porcentajes: list[float] = []
    actual = porcentaje_avance
    while actual < 100:
        porcentajes.append(round(actual, 4))
        actual += porcentaje_avance

    if not porcentajes:
        raise ValueError("No se generaron fracciones con el porcentaje indicado")

    return porcentajes


def formatear_porcentaje_para_nombre(porcentaje: float) -> str:
    if float(porcentaje).is_integer():
        return str(int(porcentaje))
    return str(porcentaje).replace(".", "_")


def generar_fracciones(
    orientacion_corte: OrientacionCorte,
    porcentaje_avance: float,
) -> list[tuple[str, float]]:
    porcentajes = obtener_porcentajes_fraccion(porcentaje_avance)

    if orientacion_corte == OrientacionCorte.HORIZONTAL:
        return [("base", porcentaje) for porcentaje in porcentajes] + [
            ("tope", porcentaje) for porcentaje in porcentajes
        ]

    return [("tope", porcentaje) for porcentaje in porcentajes] + [
        ("base", porcentaje) for porcentaje in porcentajes
    ]


def calcular_tamanio_fraccion(total: int, porcentaje: float) -> int:
    return max(1, min(total, int(round(total * (porcentaje / 100)))))


def obtener_corte_fuente_fraccion(
    alto: int,
    ancho: int,
    orientacion_corte: OrientacionCorte,
    mitad: str,
    porcentaje: float,
) -> tuple[slice, slice]:
    if orientacion_corte == OrientacionCorte.VERTICAL:
        altura_fraccion = calcular_tamanio_fraccion(alto, porcentaje)
        if mitad == "base":
            return (slice(alto - altura_fraccion, alto), slice(0, ancho))
        return (slice(0, altura_fraccion), slice(0, ancho))

    ancho_fraccion = calcular_tamanio_fraccion(ancho, porcentaje)
    if mitad == "base":
        return (slice(0, alto), slice(ancho - ancho_fraccion, ancho))
    return (slice(0, alto), slice(0, ancho_fraccion))


def cortar_imagen_en_fraccion(
    imagen: np.ndarray,
    orientacion_corte: OrientacionCorte,
    mitad: str,
    porcentaje: float,
    valor_fondo: int = 255,
) -> np.ndarray:
    if imagen.ndim != 2:
        raise ValueError("El corte en fracciones espera una imagen en escala de grises")

    alto, ancho = imagen.shape
    lado_lienzo = max(alto, ancho)
    offset_y = (lado_lienzo - alto) // 2
    offset_x = (lado_lienzo - ancho) // 2
    corte_src = obtener_corte_fuente_fraccion(
        alto,
        ancho,
        orientacion_corte,
        mitad,
        porcentaje,
    )

    imagen_fraccion = crear_lienzo_cuadrado(imagen, lado_lienzo, valor_fondo)
    corte_dst = trasladar_corte_a_lienzo(
        corte_src,
        offset_y,
        offset_x,
        lado_lienzo,
    )
    imagen_fraccion[corte_dst] = imagen[corte_src]

    return imagen_fraccion


def centrar_trazo_bbox(
    imagen: np.ndarray,
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> np.ndarray:
    if imagen.ndim != 2:
        raise ValueError("El centrado espera una imagen en escala de grises")

    _, binaria_invertida = cv2.threshold(
        imagen,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    ys, xs = np.where(binaria_invertida > 0)
    if len(xs) == 0 or len(ys) == 0:
        return imagen.copy()

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    region_interes = imagen[y0:y1, x0:x1]

    alto_salida, ancho_salida = tamano_salida or imagen.shape
    alto_region, ancho_region = region_interes.shape
    alto_copia = min(alto_region, alto_salida)
    ancho_copia = min(ancho_region, ancho_salida)
    region_interes = region_interes[:alto_copia, :ancho_copia]

    lienzo = np.full(
        (alto_salida, ancho_salida),
        valor_fondo,
        dtype=imagen.dtype,
    )
    offset_y = (alto_salida - alto_copia) // 2
    offset_x = (ancho_salida - ancho_copia) // 2
    lienzo[offset_y : offset_y + alto_copia, offset_x : offset_x + ancho_copia] = (
        region_interes
    )

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


def cortar_perfil_en_mitades(
    ruta_perfil: Path,
    orientacion: OrientacionCorte,
    ruta_destino_clase: Path,
    valor_fondo: int = 255,
) -> list[RegistroMitad]:
    imagen = leer_imagen_grises(ruta_perfil)
    tope, base = cortar_imagen_en_mitades(imagen, orientacion, valor_fondo)
    nombre_original = obtener_nombre_archivo(ruta_perfil)
    ruta_tope = crear_ruta_archivo(ruta_destino_clase, nombre_original, "_tope")
    ruta_base = crear_ruta_archivo(ruta_destino_clase, nombre_original, "_base")

    guardar_imagen(ruta_tope, tope)
    guardar_imagen(ruta_base, base)

    return [
        RegistroMitad(
            archivo_origen=ruta_perfil.name,
            archivo_generado=ruta_tope.name,
            mitad="tope",
        ),
        RegistroMitad(
            archivo_origen=ruta_perfil.name,
            archivo_generado=ruta_base.name,
            mitad="base",
        ),
    ]


def cortar_clase_en_mitades(
    ruta_origen: Path,
    ruta_destino: Path,
    clase: int | str,
    extension: str = "png",
    valor_fondo: int = 255,
) -> tuple[list[RegistroMitad], list[ErrorConversionImagen]]:
    perfiles = obtener_rutas_perfiles(ruta_origen, clase, extension)
    if not perfiles:
        return [], []

    orientacion = determinar_orientacion_corte(clase)
    ruta_destino_clase = obtener_o_crear_directorio(
        ruta_destino,
        obtener_nombre_carpeta_clase(clase),
    )
    registros: list[RegistroMitad] = []
    errores: list[ErrorConversionImagen] = []

    for ruta_perfil in perfiles:
        try:
            registros.extend(
                cortar_perfil_en_mitades(
                    ruta_perfil=ruta_perfil,
                    orientacion=orientacion,
                    ruta_destino_clase=ruta_destino_clase,
                    valor_fondo=valor_fondo,
                )
            )
        except Exception as exc:
            errores.append(ErrorConversionImagen(ruta_perfil, str(exc)))

    return registros, errores


def cortar_imagenes_en_mitades(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoCorte:
    registros: list[RegistroMitad] = []
    errores: list[ErrorConversionImagen] = []
    clases_sin_imagenes: list[str] = []

    for clase in clases:
        registros_clase, errores_clase = cortar_clase_en_mitades(
            ruta_origen=ruta_origen,
            ruta_destino=ruta_destino,
            clase=clase,
            extension=extension,
            valor_fondo=valor_fondo,
        )

        if not registros_clase and not errores_clase:
            clases_sin_imagenes.append(obtener_nombre_carpeta_clase(clase))

        registros.extend(registros_clase)
        errores.extend(errores_clase)

    return ResultadoCorte(
        imagenes_origen_procesadas=len(registros) // 2,
        cortes_generados=len(registros),
        registros=registros,
        clases_sin_imagenes=clases_sin_imagenes,
        errores=errores,
        nombre_cortes="Mitades",
    )


def construir_metadata_mitades(
    metadata_original: pd.DataFrame,
    registros_mitades: pd.DataFrame,
    columna_identificador: str = "Identificador",
    how: str = "inner",
) -> pd.DataFrame:
    from vasijas.metadata import construir_metadata_mitades as construir

    return construir(
        metadata_original=metadata_original,
        registros_derivados=registros_mitades,
        columna_identificador=columna_identificador,
        how=how,
    )


def cortar_perfil_en_fracciones(
    ruta_perfil: Path,
    orientacion: OrientacionCorte,
    ruta_destino_clase: Path,
    porcentaje_avance: float,
    valor_fondo: int = 255,
) -> list[RegistroFraccion]:
    imagen = leer_imagen_grises(ruta_perfil)
    nombre_original = obtener_nombre_archivo(ruta_perfil)
    registros: list[RegistroFraccion] = []

    for mitad, porcentaje in generar_fracciones(orientacion, porcentaje_avance):
        imagen_fraccion = cortar_imagen_en_fraccion(
            imagen=imagen,
            orientacion_corte=orientacion,
            mitad=mitad,
            porcentaje=porcentaje,
            valor_fondo=valor_fondo,
        )
        porcentaje_nombre = formatear_porcentaje_para_nombre(porcentaje)
        ruta_fraccion = crear_ruta_archivo(
            ruta_destino_clase,
            nombre_original,
            f"_{mitad}_{porcentaje_nombre}",
        )

        guardar_imagen(ruta_fraccion, imagen_fraccion)
        registros.append(
            RegistroFraccion(
                archivo_origen=ruta_perfil.name,
                archivo_generado=ruta_fraccion.name,
                mitad=mitad,
                porcentaje=porcentaje,
            )
        )

    return registros


def cortar_clase_en_fracciones(
    ruta_origen: Path,
    ruta_destino: Path,
    clase: int | str,
    porcentaje_avance: float,
    extension: str = "png",
    valor_fondo: int = 255,
) -> tuple[list[RegistroFraccion], list[ErrorConversionImagen]]:
    perfiles = obtener_rutas_perfiles(ruta_origen, clase, extension)
    if not perfiles:
        return [], []

    orientacion = determinar_orientacion_corte(clase)
    ruta_destino_clase = obtener_o_crear_directorio(
        ruta_destino,
        obtener_nombre_carpeta_clase(clase),
    )
    registros: list[RegistroFraccion] = []
    errores: list[ErrorConversionImagen] = []

    for ruta_perfil in perfiles:
        try:
            registros.extend(
                cortar_perfil_en_fracciones(
                    ruta_perfil=ruta_perfil,
                    orientacion=orientacion,
                    ruta_destino_clase=ruta_destino_clase,
                    porcentaje_avance=porcentaje_avance,
                    valor_fondo=valor_fondo,
                )
            )
        except Exception as exc:
            errores.append(ErrorConversionImagen(ruta_perfil, str(exc)))

    return registros, errores


def cortar_imagenes_clasificadas_en_fracciones(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    porcentaje_avance: float,
    extension: str = "png",
    valor_fondo: int = 255,
) -> ResultadoCorte:
    registros: list[RegistroFraccion] = []
    errores: list[ErrorConversionImagen] = []
    clases_sin_imagenes: list[str] = []

    for clase in clases:
        registros_clase, errores_clase = cortar_clase_en_fracciones(
            ruta_origen=ruta_origen,
            ruta_destino=ruta_destino,
            clase=clase,
            porcentaje_avance=porcentaje_avance,
            extension=extension,
            valor_fondo=valor_fondo,
        )

        if not registros_clase and not errores_clase:
            clases_sin_imagenes.append(obtener_nombre_carpeta_clase(clase))

        registros.extend(registros_clase)
        errores.extend(errores_clase)

    fracciones_por_imagen = len(
        generar_fracciones(OrientacionCorte.VERTICAL, porcentaje_avance)
    )
    imagenes_origen_procesadas = (
        len(registros) // fracciones_por_imagen if fracciones_por_imagen else 0
    )

    return ResultadoCorte(
        imagenes_origen_procesadas=imagenes_origen_procesadas,
        cortes_generados=len(registros),
        registros=registros,
        clases_sin_imagenes=clases_sin_imagenes,
        errores=errores,
        nombre_cortes="Fracciones",
    )


def centrar_imagen_fraccion(
    ruta_imagen: Path,
    ruta_destino_clase: Path,
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> RegistroCentrado:
    imagen = leer_imagen_grises(ruta_imagen)
    nombre_original = obtener_nombre_archivo(ruta_imagen)
    ruta_destino = crear_ruta_archivo(
        ruta_base=ruta_destino_clase,
        nombre_archivo=nombre_original,
        sufijo="_centrado",
    )
    imagen_centrada = centrar_trazo_bbox(
        imagen=imagen,
        tamano_salida=tamano_salida,
        valor_fondo=valor_fondo,
    )

    guardar_imagen(ruta_destino, imagen_centrada)

    return RegistroCentrado(
        archivo_origen=ruta_imagen.name,
        archivo_generado=ruta_destino.name,
    )


def centrar_clase_fracciones(
    ruta_origen: Path,
    ruta_destino: Path,
    clase: int | str,
    extension: str = "png",
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> tuple[list[RegistroCentrado], list[ErrorConversionImagen]]:
    rutas_imagenes = obtener_rutas_perfiles(ruta_origen, clase, extension)
    if not rutas_imagenes:
        return [], []

    ruta_destino_clase = obtener_o_crear_directorio(
        ruta_destino,
        obtener_nombre_carpeta_clase(clase),
    )
    registros: list[RegistroCentrado] = []
    errores: list[ErrorConversionImagen] = []

    for ruta_imagen in rutas_imagenes:
        try:
            registros.append(
                centrar_imagen_fraccion(
                    ruta_imagen=ruta_imagen,
                    ruta_destino_clase=ruta_destino_clase,
                    tamano_salida=tamano_salida,
                    valor_fondo=valor_fondo,
                )
            )
        except Exception as exc:
            errores.append(ErrorConversionImagen(ruta_imagen, str(exc)))

    return registros, errores


def centrar_imagenes(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    extension: str = "png",
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> ResultadoCentrado:
    registros: list[RegistroCentrado] = []
    errores: list[ErrorConversionImagen] = []
    clases_sin_imagenes: list[str] = []

    for clase in clases:
        registros_clase, errores_clase = centrar_clase_fracciones(
            ruta_origen=ruta_origen,
            ruta_destino=ruta_destino,
            clase=clase,
            extension=extension,
            tamano_salida=tamano_salida,
            valor_fondo=valor_fondo,
        )

        if not registros_clase and not errores_clase:
            clases_sin_imagenes.append(obtener_nombre_carpeta_clase(clase))

        registros.extend(registros_clase)
        errores.extend(errores_clase)

    return ResultadoCentrado(
        imagenes_procesadas=len(registros),
        registros=registros,
        clases_sin_imagenes=clases_sin_imagenes,
        errores=errores,
    )


def construir_metadata_centrados(
    metadata_original: pd.DataFrame,
    registros_centrados: pd.DataFrame | Iterable[RegistroCentrado],
    columna_identificador: str = "Identificador",
    how: str = "inner",
) -> pd.DataFrame:
    from vasijas.metadata import construir_metadata_centrados as construir

    return construir(
        metadata_original=metadata_original,
        registros_centrados=registros_centrados,
        columna_identificador=columna_identificador,
        how=how,
    )


def guardar_metadata_centrados(
    ruta_metadata_origen: Path,
    ruta_metadata_destino: Path,
    registros_centrados: pd.DataFrame | Iterable[RegistroCentrado],
    columna_identificador: str = "Identificador",
    how: str = "inner",
) -> Path:
    from vasijas.metadata import guardar_metadata_centrados as guardar

    return guardar(
        ruta_metadata_origen=ruta_metadata_origen,
        ruta_metadata_destino=ruta_metadata_destino,
        registros_centrados=registros_centrados,
        columna_identificador=columna_identificador,
        how=how,
    )


def centrar_dataset_imagenes(
    ruta_origen: Path,
    ruta_destino: Path,
    clases: Iterable[int | str],
    nombre_metadata: str = "metadata.csv",
    extension: str = "png",
    tamano_salida: tuple[int, int] | None = None,
    valor_fondo: int = 255,
) -> ResultadoCentrado:
    """Wrapper de compatibilidad; preferir vasijas.pipelines.centrar_dataset_imagenes."""
    from vasijas.pipelines import centrar_dataset_clasificado_de_fracciones as ejecutar

    return ejecutar(
        ruta_origen=ruta_origen,
        ruta_destino=ruta_destino,
        clases=clases,
        nombre_metadata=nombre_metadata,
        extension=extension,
        tamano_salida=tamano_salida,
        valor_fondo=valor_fondo,
    )
