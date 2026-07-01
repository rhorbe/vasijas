import shutil
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from vasijas.nombres import normalizar_nombre


@dataclass(frozen=True)
class PerfilClasificacion:
    identificador: str
    clase: str | int | float
    ubicacion: str


@dataclass(frozen=True)
class RutasPerfilClasificado:
    origen: Path
    destino: Path
    carpeta_destino: Path


@dataclass(frozen=True)
class ResultadoCopiaClasificada:
    archivos_copiados: int
    rutas_no_encontradas: list[Path]
    conteo_por_clase: dict[Path, int]

    @property
    def archivos_no_encontrados(self) -> int:
        return len(self.rutas_no_encontradas)


def obtener_o_crear_directorio(ruta_base: Path, nombre_directorio: str) -> Path:
    ruta_completa = ruta_base / nombre_directorio
    ruta_completa.mkdir(parents=True, exist_ok=True)

    if not ruta_completa.is_dir():
        raise RuntimeError(f"No se pudo crear el directorio: {ruta_completa}")

    return ruta_completa


def obtener_nombre_carpeta_clase(valor_clase: str | int | float) -> str:
    """
    Devuelve el nombre de la carpeta segun la clase.

    Si el valor contiene un numero, devuelve nombres como ``clase01`` o
    ``clase11``. Si esta vacio o no puede interpretarse como numero, devuelve
    ``sin_clase``.
    """
    if pd.isna(valor_clase) or str(valor_clase).strip() == "":
        return "sin_clase"

    try:
        numero = int(float(valor_clase))
    except (TypeError, ValueError):
        return "sin_clase"

    return f"clase{numero:02d}"


def crear_perfil_clasificacion(
    fila: pd.Series,
    columna_identificador: str = "Identificador",
    columna_clase: str = "Clase Ana",
    columna_ubicacion: str = "Ubicacion",
) -> PerfilClasificacion:
    return PerfilClasificacion(
        identificador=str(fila[columna_identificador]).strip(),
        clase=fila[columna_clase],
        ubicacion=str(fila[columna_ubicacion]).strip(),
    )


def obtener_perfiles_clasificacion(
    df: pd.DataFrame,
    columna_identificador: str = "Identificador",
    columna_clase: str = "Clase Ana",
    columna_ubicacion: str = "Ubicacion",
) -> list[PerfilClasificacion]:
    return [
        crear_perfil_clasificacion(
            fila=fila,
            columna_identificador=columna_identificador,
            columna_clase=columna_clase,
            columna_ubicacion=columna_ubicacion,
        )
        for _, fila in df.iterrows()
    ]


def obtener_nombre_archivo_perfil(
    identificador: str,
    extension: str = "png",
) -> str:
    return f"{identificador}.{extension.lstrip('.')}"


def obtener_rutas_perfil_clasificado(
    perfil: PerfilClasificacion,
    ruta_origen: Path,
    ruta_destino: Path,
    extension: str = "png",
) -> RutasPerfilClasificado:
    nombre_carpeta_clase = obtener_nombre_carpeta_clase(perfil.clase)
    carpeta_destino = ruta_destino / nombre_carpeta_clase
    nombre_archivo = obtener_nombre_archivo_perfil(perfil.identificador, extension)

    return RutasPerfilClasificado(
        origen=ruta_origen / perfil.ubicacion / nombre_archivo,
        destino=carpeta_destino / nombre_archivo,
        carpeta_destino=carpeta_destino,
    )


def copiar_perfil_clasificado(rutas: RutasPerfilClasificado) -> bool:
    if not rutas.origen.exists():
        return False

    rutas.carpeta_destino.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rutas.origen, rutas.destino)
    return True


def copiar_perfiles_por_clase(
    df: pd.DataFrame,
    ruta_origen: Path,
    ruta_destino: Path,
    columna_identificador: str = "Identificador",
    columna_clase: str = "Clase Ana",
    columna_ubicacion: str = "Ubicacion",
    extension: str = "png",
) -> ResultadoCopiaClasificada:
    perfiles = obtener_perfiles_clasificacion(
        df=df,
        columna_identificador=columna_identificador,
        columna_clase=columna_clase,
        columna_ubicacion=columna_ubicacion,
    )
    archivos_copiados = 0
    rutas_no_encontradas: list[Path] = []
    conteo_por_clase: dict[Path, int] = {}

    for perfil in perfiles:
        rutas = obtener_rutas_perfil_clasificado(
            perfil=perfil,
            ruta_origen=ruta_origen,
            ruta_destino=ruta_destino,
            extension=extension,
        )

        if not copiar_perfil_clasificado(rutas):
            rutas_no_encontradas.append(rutas.origen)
            continue

        archivos_copiados += 1
        conteo_por_clase[rutas.carpeta_destino] = (
            conteo_por_clase.get(rutas.carpeta_destino, 0) + 1
        )

    return ResultadoCopiaClasificada(
        archivos_copiados=archivos_copiados,
        rutas_no_encontradas=rutas_no_encontradas,
        conteo_por_clase=conteo_por_clase,
    )


def copiar_archivo(
    origen: Path,
    destino: Path,
) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)
    return destino


def copiar_png_normalizados_por_ubicacion(
    ruta_origen: Path,
    ruta_destino: Path,
) -> list[Path]:
    """
    Copia archivos PNG desde subdirectorios de origen hacia destino,
    preservando cada ubicación y normalizando los nombres de archivo.
    """
    destinos: list[Path] = []

    rutas_ubicaciones = [ruta for ruta in ruta_origen.iterdir() if ruta.is_dir()]

    for ruta_ubicacion in rutas_ubicaciones:
        nombre_ubicacion = ruta_ubicacion.name

        ruta_destino_ubicacion = ruta_destino / nombre_ubicacion
        ruta_destino_ubicacion.mkdir(parents=True, exist_ok=True)

        archivos_png = list(ruta_ubicacion.glob("*.png"))

        for archivo in archivos_png:
            nombre_normalizado = normalizar_nombre(archivo.name)
            destino = ruta_destino_ubicacion / nombre_normalizado
            shutil.copy2(archivo, destino)
            destinos.append(destino)

    return destinos


