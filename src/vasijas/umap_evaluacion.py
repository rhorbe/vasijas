"""Extracción de características y visualización exploratoria del test persistido."""

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.datasets.folder import default_loader

from vasijas.evaluacion import (
    indexar_metadata_por_ruta,
    validar_identificadores_metadata,
)
from vasijas.paths import ruta_relativa

DIMENSION_CARACTERISTICAS_RESNET18 = 512
COLUMNAS_COORDENADAS_UMAP = ["UMAP_1", "UMAP_2"]
COLUMNAS_IDENTIDAD_TEST = {"Identificador", "ruta_relativa", "clase_carpeta", "split"}


@dataclass(frozen=True)
class ConfiguracionUmap:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


class DatasetTestConRuta(Dataset[tuple[torch.Tensor, str]]):
    """Carga RGB y devuelve la ruta para verificar la identidad de cada vector."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        data_dir: Path,
        transformacion: Callable[[Image], torch.Tensor],
    ) -> None:
        self.rutas = metadata["ruta_relativa"].tolist()
        self.data_dir = data_dir
        self.transformacion = transformacion

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, str]:
        ruta = self.rutas[indice]
        imagen = default_loader(str(self.data_dir / ruta))
        return self.transformacion(imagen), ruta


def validar_columnas_metadata(
    metadata_split: pd.DataFrame,
    columnas_requeridas: set[str],
) -> None:
    """Exige las columnas indicadas antes de preparar los datos para UMAP."""
    faltan_columnas = columnas_requeridas - set(metadata_split.columns)
    if faltan_columnas:
        raise ValueError(
            f"Faltan columnas requeridas en metadata: {sorted(faltan_columnas)}"
        )


def seleccionar_metadata(
    metadata_split: pd.DataFrame,
    conjuntos: str | Iterable[str],
) -> pd.DataFrame:
    """Copia las filas de los splits elegidos y reinicia el índice.

    Acepta un nombre ("test") o varios (["train", "val"]). Conserva todas
    las columnas y el orden original de las filas, sin duplicar muestras.
    Si no hay coincidencias, devuelve una tabla vacía con las mismas columnas.
    """
    if isinstance(conjuntos, str):
        conjuntos = [conjuntos]
    return (
        metadata_split.loc[metadata_split["split"].isin(conjuntos)]
        .copy()
        .reset_index(drop=True)
    )


def contar_imagenes_por_grupo(
    metadata: pd.DataFrame,
    columna: str,
    nombre_conteo: str = "imagenes",
) -> pd.DataFrame:
    """Cuenta filas por grupo y devuelve una tabla indexada por sus valores."""
    return metadata.groupby(columna).size().rename(nombre_conteo).to_frame()


def obtener_muestra_metadata(
    metadata: pd.DataFrame,
    cantidad: int = 5,
) -> pd.DataFrame:
    """Devuelve las primeras filas de la metadata para su inspección."""
    return metadata.head(cantidad)


def validar_metadata_test(
    metadata_test: pd.DataFrame,
    columnas_requeridas: set[str],
    data_dir: Path,
) -> None:
    """Valida identidad, campos requeridos y archivos de las imágenes de test."""
    columnas_identidad = COLUMNAS_IDENTIDAD_TEST | columnas_requeridas
    validar_columnas_metadata(metadata_test, columnas_identidad)
    if metadata_test.empty:
        raise ValueError("No hay imágenes con split == 'test' en la metadata.")
    validar_identificadores_metadata(metadata_test)
    if metadata_test[list(columnas_identidad)].isna().any().any():
        raise ValueError(
            "Cada imagen de test debe tener identificador, ruta, clase y split."
        )
    if not metadata_test["ruta_relativa"].is_unique:
        raise ValueError("Las rutas de las imágenes de test deben ser únicas.")
    if not metadata_test["split"].eq("test").all():
        raise ValueError("La metadata debe contener solamente imágenes del split test.")
    validar_archivos_imagenes_test(metadata_test["ruta_relativa"], data_dir)


def validar_archivos_imagenes_test(rutas_relativas: pd.Series, data_dir: Path) -> None:
    """Exige que todas las rutas pertenezcan al directorio y sean archivos existentes."""
    directorio = data_dir.resolve()
    rutas_test = [(data_dir / ruta).resolve() for ruta in rutas_relativas]
    if any(not ruta.is_relative_to(directorio) for ruta in rutas_test):
        raise ValueError("Las rutas de test deben estar dentro de DATA_DIR.")
    faltantes = [
        relativa
        for relativa, ruta in zip(rutas_relativas, rutas_test)
        if not ruta.is_file()
    ]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan {len(faltantes)} imágenes de test: {faltantes[:10]}"
        )


def cargar_resnet18_entrenada(
    ruta_modelo: Path,
    dispositivo: torch.device,
) -> nn.Module:
    """Restaura pesos locales y el número de clases, sin descargar pesos."""
    state_dict = torch.load(ruta_modelo, map_location="cpu", weights_only=True)
    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, state_dict["fc.weight"].shape[0])
    modelo.load_state_dict(state_dict, strict=True)
    return modelo.to(dispositivo).eval()


def crear_extractor_resnet18(modelo: nn.Module) -> nn.Sequential:
    """Comparte las capas hasta avgpool en evaluación y aplana su salida; excluye fc."""
    return nn.Sequential(*list(modelo.children())[:-1], nn.Flatten(1)).eval()


def _extraer_vectores_y_rutas(
    extractor: nn.Module,
    lotes: Iterable[tuple[torch.Tensor, Iterable[str]]],
    dispositivo: torch.device,
    al_finalizar_lote: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Motor de inferencia común; el callback recibe número de lote e imágenes acumuladas."""
    bloques = []
    rutas_extraidas = []
    with torch.inference_mode():
        for numero_lote, (imagenes, rutas) in enumerate(lotes, start=1):
            vectores = extractor(imagenes.to(dispositivo))
            bloques.append(vectores.cpu().numpy())
            rutas_extraidas.extend(rutas)
            if al_finalizar_lote is not None:
                al_finalizar_lote(numero_lote, len(rutas_extraidas))
    if not bloques:
        raise ValueError("No se extrajeron características de ningún lote.")
    return np.concatenate(bloques, axis=0), rutas_extraidas


def extraer_caracteristicas_por_lotes(
    extractor: nn.Module,
    loader: DataLoader,
    dispositivo: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Extrae pares (imágenes, rutas) e informa progreso cada cien lotes y al final.

    El extractor debe estar en modo evaluación y en el dispositivo indicado.
    """

    def informar_avance(numero_lote: int, cantidad_imagenes: int) -> None:
        if numero_lote % 100 == 0 or numero_lote == len(loader):
            print(f"Características: {cantidad_imagenes}/{len(loader.dataset)}")

    return _extraer_vectores_y_rutas(extractor, loader, dispositivo, informar_avance)


def validar_orden_rutas_extraidas(
    rutas_extraidas: list[str],
    metadata: pd.DataFrame,
) -> None:
    """Exige cobertura completa y el mismo orden de rutas que la metadata."""
    if rutas_extraidas != metadata["ruta_relativa"].tolist():
        raise ValueError(
            "Los vectores no corresponden al orden completo de la metadata."
        )


def validar_caracteristicas_resnet18(
    caracteristicas: np.ndarray,
    cantidad_imagenes: int,
) -> None:
    """Exige un vector finito de 512 características por imagen."""
    if caracteristicas.shape != (cantidad_imagenes, DIMENSION_CARACTERISTICAS_RESNET18):
        raise ValueError("Debe existir un vector de 512 características por imagen.")
    if not np.isfinite(caracteristicas).all():
        raise ValueError("Las características contienen valores no finitos.")


def _iterar_lotes_test_validados(
    loader: DataLoader,
    metadata_por_ruta: pd.DataFrame,
    nombres_clases: list[str],
) -> Iterable[tuple[torch.Tensor, Iterable[str]]]:
    """Adapta lotes etiquetados tras comprobar pertenencia y clase de cada imagen."""
    for imagenes, etiquetas, rutas in loader:
        for ruta, etiqueta in zip(rutas, etiquetas.tolist(), strict=True):
            if ruta not in metadata_por_ruta.index:
                raise ValueError(f"La ruta {ruta!r} no pertenece al test.")
            if metadata_por_ruta.at[ruta, "clase_carpeta"] != nombres_clases[etiqueta]:
                raise ValueError(f"La clase de {ruta!r} no coincide con la metadata.")
        yield imagenes, rutas


def _validar_cobertura_rutas_test(
    rutas_extraidas: list[str], metadata_por_ruta: pd.DataFrame
) -> None:
    """Exige exactamente una aparición de cada ruta del test, en cualquier orden."""
    rutas_unicas = set(rutas_extraidas)
    if (
        len(rutas_extraidas) != len(metadata_por_ruta)
        or len(rutas_unicas) != len(rutas_extraidas)
        or rutas_unicas != set(metadata_por_ruta.index)
    ):
        raise ValueError("Debe existir exactamente un vector por imagen de test.")


def extraer_caracteristicas_test(
    modelo: nn.Module,
    test_loader: DataLoader,
    metadata_split: pd.DataFrame,
    nombres_clases: list[str],
    dispositivo: torch.device,
    informar_progreso: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extrae vectores de avgpool de ResNet18 sin modificar su clasificador.

    Conserva la identidad por ruta, independientemente del orden del loader.
    Exige exactamente una muestra por imagen de test, incluido el último lote.
    """
    validar_columnas_metadata(metadata_split, COLUMNAS_IDENTIDAD_TEST)
    metadata_test = seleccionar_metadata(metadata_split, "test")
    requeridas = ["Identificador", "ruta_relativa", "clase_carpeta"]
    if metadata_test.empty or metadata_test[requeridas].isna().any().any():
        raise ValueError("El test debe contener imágenes con identidad, ruta y clase.")
    metadata_por_ruta = indexar_metadata_por_ruta(metadata_test)
    if not metadata_por_ruta.index.is_unique:
        raise ValueError("Las rutas de test deben ser únicas.")

    estados_entrenamiento = [(capa, capa.training) for capa in modelo.modules()]

    def informar_avance(numero_lote: int, cantidad_imagenes: int) -> None:
        if informar_progreso is not None:
            informar_progreso(cantidad_imagenes, len(test_loader.dataset))

    try:
        extractor = crear_extractor_resnet18(modelo)
        lotes = _iterar_lotes_test_validados(
            test_loader, metadata_por_ruta, nombres_clases
        )
        vectores, rutas_extraidas = _extraer_vectores_y_rutas(
            extractor, lotes, dispositivo, informar_avance
        )
    finally:
        for capa, entrenando in estados_entrenamiento:
            capa.training = entrenando

    _validar_cobertura_rutas_test(rutas_extraidas, metadata_por_ruta)
    tabla = metadata_por_ruta.loc[rutas_extraidas].reset_index()
    tabla = tabla[list(metadata_split.columns)]
    return vectores, tabla


def validar_entrada_proyeccion_umap(
    caracteristicas: np.ndarray,
    metadata_test: pd.DataFrame,
) -> None:
    """Exige vectores finitos alineados con al menos tres imágenes de test."""
    if caracteristicas.ndim != 2 or caracteristicas.shape[1] == 0:
        raise ValueError("Las características deben ser una matriz de vectores.")
    if len(caracteristicas) != len(metadata_test) or len(caracteristicas) < 3:
        raise ValueError(
            "Se requieren al menos tres vectores alineados con la metadata."
        )
    if not np.isfinite(caracteristicas).all():
        raise ValueError("Las características contienen valores no finitos.")
    if not metadata_test["split"].eq("test").all():
        raise ValueError("UMAP debe recibir solamente imágenes del split test.")


def validar_configuracion_umap(
    configuracion: ConfiguracionUmap,
    cantidad_imagenes: int,
) -> None:
    """Comprueba los límites de vecinos y distancia mínima de la proyección."""
    if not 2 <= configuracion.n_neighbors < cantidad_imagenes:
        raise ValueError(
            "n_neighbors debe ser al menos 2 y menor que el número de imágenes."
        )
    if not 0 <= configuracion.min_dist <= 1:
        raise ValueError("min_dist debe estar entre cero y uno.")


def ajustar_umap(
    caracteristicas: np.ndarray,
    configuracion: ConfiguracionUmap,
) -> np.ndarray:
    """Ajusta UMAP en dos dimensiones sin etiquetas, con un único trabajo."""
    from umap import UMAP

    return UMAP(
        n_components=2,
        **asdict(configuracion),
        n_jobs=1,
        # También permite proyectar conjuntos pequeños (tres muestras).
        init="random",
    ).fit_transform(caracteristicas)


def validar_coordenadas_umap(
    coordenadas: np.ndarray,
    cantidad_imagenes: int,
) -> None:
    """Exige exactamente dos coordenadas finitas por imagen."""
    if (
        coordenadas.shape != (cantidad_imagenes, 2)
        or not np.isfinite(coordenadas).all()
    ):
        raise ValueError("UMAP debe producir dos coordenadas finitas por imagen.")


def crear_tabla_umap(
    metadata: pd.DataFrame,
    coordenadas: np.ndarray,
) -> pd.DataFrame:
    """Añade las coordenadas por posición a una copia de la metadata."""
    tabla = metadata.copy()
    tabla[COLUMNAS_COORDENADAS_UMAP] = coordenadas
    return tabla


def proyectar_caracteristicas_umap(
    caracteristicas: np.ndarray,
    metadata_test: pd.DataFrame,
    configuracion: ConfiguracionUmap,
) -> pd.DataFrame:
    """Valida y proyecta test sin etiquetas; devuelve metadata con coordenadas."""
    vectores = np.asarray(caracteristicas)
    validar_entrada_proyeccion_umap(vectores, metadata_test)
    validar_configuracion_umap(configuracion, len(vectores))
    coordenadas = ajustar_umap(vectores, configuracion)
    validar_coordenadas_umap(coordenadas, len(metadata_test))
    return crear_tabla_umap(metadata_test.reset_index(drop=True), coordenadas)


def validar_tabla_vistas_fracciones(tabla: pd.DataFrame) -> None:
    """Exige puntos de test con metadatos completos y coordenadas finitas."""
    requeridas = ["split", "borde_origen", "porcentaje_avance", "UMAP_1", "UMAP_2"]
    faltantes = set(requeridas) - set(tabla.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas para las vistas de fracciones: {sorted(faltantes)}"
        )
    if tabla.empty or tabla[requeridas].isna().any().any():
        raise ValueError("Se requieren puntos y metadatos completos para colorear.")
    if not tabla["split"].eq("test").all():
        raise ValueError("Las vistas deben contener solamente imágenes del split test.")
    if not np.isfinite(tabla[COLUMNAS_COORDENADAS_UMAP].to_numpy(dtype=float)).all():
        raise ValueError("Las coordenadas deben ser finitas.")


def normalizar_origenes_recorte(origenes: pd.Series) -> pd.Series:
    """Quita espacios, convierte a minúsculas y rechaza orígenes desconocidos."""
    origen = origenes.astype("string").str.strip().str.lower()
    inesperados = set(origen) - {"completo", "tope", "base"}
    if inesperados:
        raise ValueError(f"Orígenes desconocidos: {sorted(inesperados)}")
    return origen


def validar_porcentajes_perfil(porcentaje: pd.Series, origen: pd.Series) -> None:
    """Exige porcentajes finitos: 100 para completas y entre 0 y 100 para recortes."""
    if not np.isfinite(porcentaje.to_numpy(dtype=float)).all():
        raise ValueError("Los porcentajes deben ser finitos.")
    if not porcentaje.loc[origen.eq("completo")].eq(100).all():
        raise ValueError("Las imágenes completas deben indicar 100 %.")
    recortes = porcentaje.loc[origen.isin(["tope", "base"])]
    if not recortes.between(0, 100, inclusive="neither").all():
        raise ValueError(
            "Los recortes deben indicar un porcentaje entre 0 y 100 exclusivo."
        )


def validar_presencia_tope_y_base(origen: pd.Series) -> None:
    """Exige ambas clases de recorte para comparar sus distribuciones."""
    if not {"tope", "base"}.issubset(set(origen)):
        raise ValueError("Se necesitan fracciones de tope y de base para comparar.")


def preparar_datos_vistas_fracciones(
    tabla_umap: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Valida y devuelve orígenes, porcentajes numéricos y filas de recortes.

    Conserva la tabla original, sus índices y coordenadas; no vuelve a ajustar UMAP.
    """
    validar_tabla_vistas_fracciones(tabla_umap)
    origen = normalizar_origenes_recorte(tabla_umap["borde_origen"])
    porcentaje = pd.to_numeric(tabla_umap["porcentaje_avance"], errors="raise")
    validar_porcentajes_perfil(porcentaje, origen)
    validar_presencia_tope_y_base(origen)
    fracciones = tabla_umap.loc[origen.isin(["tope", "base"])]
    return origen, porcentaje, fracciones


def obtener_limites_grafico_umap(
    figura: Figure,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Obtiene los límites de ambos ejes para reutilizarlos en otras vistas."""
    eje = figura.axes[0]
    return eje.get_xlim(), eje.get_ylim()


def imprimir_resumen_fracciones(origen: pd.Series) -> None:
    """Informa los recuentos por origen y los totales de fracciones y de test."""
    for categoria in ("completo", "tope", "base"):
        print(f"{categoria}: {int(origen.eq(categoria).sum())} imágenes de test")
    cantidad_fracciones = int(origen.isin(["tope", "base"]).sum())
    print(f"Fracciones: {cantidad_fracciones} · Total: {len(origen)}")


def configurar_leyenda_interior_umap(
    figura: Figure,
    titulo: str = "Clase real",
) -> None:
    """Ubica la leyenda dentro del eje donde menos tape los puntos."""
    figura.axes[0].legend(
        title=titulo,
        loc="best",
        markerscale=2.5,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )


def configurar_ejes_umap(
    eje: Axes,
    titulo: str,
    *,
    limites_x: tuple[float, float] | None = None,
    limites_y: tuple[float, float] | None = None,
) -> None:
    """Rotula los ejes UMAP y aplica límites compartidos cuando se indican."""
    eje.set(title=titulo, xlabel="UMAP 1", ylabel="UMAP 2")
    if limites_x is not None:
        eje.set_xlim(limites_x)
    if limites_y is not None:
        eje.set_ylim(limites_y)


def _dibujar_grupo_umap(
    eje: Axes,
    grupo: pd.DataFrame,
    etiqueta: str,
    color: str | tuple[float, float, float, float],
) -> None:
    """Aplica el estilo común a un grupo de puntos con color discreto."""
    eje.scatter(
        grupo["UMAP_1"],
        grupo["UMAP_2"],
        s=14,
        alpha=0.65,
        color=color,
        linewidths=0,
        label=f"{etiqueta} (n={len(grupo)})",
    )


def graficar_umap_test(
    tabla: pd.DataFrame,
    *,
    posicion_leyenda: Literal["interior", "exterior"] = "exterior",
) -> Figure:
    """Colorea cada imagen por su clase real con una leyenda discreta."""
    if posicion_leyenda not in {"interior", "exterior"}:
        raise ValueError("La posición de la leyenda debe ser 'interior' o 'exterior'.")
    figura, eje = plt.subplots(figsize=(11, 8))
    clases = sorted(tabla["clase_carpeta"].unique())
    colores = plt.get_cmap("tab10" if len(clases) <= 10 else "tab20", len(clases))
    for indice, clase in enumerate(clases):
        grupo = tabla.loc[tabla["clase_carpeta"].eq(clase)]
        _dibujar_grupo_umap(eje, grupo, clase, colores(indice))
    configurar_ejes_umap(
        eje, f"UMAP de características ResNet18 · test (n={len(tabla)})"
    )
    if posicion_leyenda == "interior":
        configurar_leyenda_interior_umap(figura)
    else:
        eje.legend(title="Clase real", bbox_to_anchor=(1.02, 1), loc="upper left")
    figura.tight_layout()
    return figura


def dibujar_fracciones_por_origen(
    eje: Axes,
    tabla_umap: pd.DataFrame,
    origen: pd.Series,
) -> None:
    """Dibuja tope en naranja y base en azul usando los orígenes normalizados."""
    for categoria, etiqueta, color in [
        ("tope", "Tope (borde)", "#E69F00"),
        ("base", "Base", "#0072B2"),
    ]:
        grupo = tabla_umap.loc[origen.eq(categoria)]
        _dibujar_grupo_umap(eje, grupo, etiqueta, color)


def graficar_umap_por_origen(
    tabla_umap: pd.DataFrame,
    origen: pd.Series,
    limites_x: tuple[float, float],
    limites_y: tuple[float, float],
) -> Figure:
    """Crea la vista de recortes con coordenadas y orígenes ya preparados.

    La serie de orígenes debe estar normalizada y alineada con la tabla.
    Las imágenes completas se excluyen; no se vuelve a ajustar UMAP.
    """
    figura, eje = plt.subplots(figsize=(11, 8))
    dibujar_fracciones_por_origen(eje, tabla_umap, origen)
    cantidad_fracciones = int(origen.isin(["tope", "base"]).sum())
    configurar_ejes_umap(
        eje,
        f"UMAP test · fracciones por origen (n={cantidad_fracciones})",
        limites_x=limites_x,
        limites_y=limites_y,
    )
    configurar_leyenda_interior_umap(figura, titulo="Origen del recorte")
    figura.tight_layout()
    return figura


def dibujar_puntos_por_porcentaje(
    eje: Axes,
    tabla_umap: pd.DataFrame,
    porcentaje: pd.Series,
) -> PathCollection:
    """Dibuja de menor a mayor porcentaje; conserva el orden de los empates."""
    orden = np.argsort(porcentaje.to_numpy(), kind="stable")
    return eje.scatter(
        tabla_umap.iloc[orden]["UMAP_1"],
        tabla_umap.iloc[orden]["UMAP_2"],
        c=porcentaje.iloc[orden],
        cmap="Blues",
        norm=Normalize(vmin=0, vmax=100),
        s=14,
        alpha=1.0,
        linewidths=0,
    )


def agregar_barra_porcentaje_conservado(
    figura: Figure,
    eje: Axes,
    puntos: PathCollection,
) -> None:
    """Añade la escala de color de 0 a 100 % y explica el perfil completo."""
    barra = figura.colorbar(puntos, ax=eje)
    barra.set_label("Perfil conservado (%) · 100 % = imagen completa")
    barra.set_ticks([0, 25, 50, 75, 100])
    barra.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])


def graficar_umap_por_porcentaje(
    tabla_umap: pd.DataFrame,
    porcentaje: pd.Series,
    limites_x: tuple[float, float],
    limites_y: tuple[float, float],
) -> Figure:
    """Crea la vista por porcentaje con coordenadas y porcentajes ya preparados.

    La serie numérica debe seguir el orden de las filas de la tabla.
    Incluye imágenes completas y recortes, sin volver a ajustar UMAP.
    """
    figura, eje = plt.subplots(figsize=(11, 8))
    puntos = dibujar_puntos_por_porcentaje(eje, tabla_umap, porcentaje)
    agregar_barra_porcentaje_conservado(figura, eje, puntos)
    configurar_ejes_umap(
        eje,
        f"UMAP test · porcentaje conservado (n={len(tabla_umap)})",
        limites_x=limites_x,
        limites_y=limites_y,
    )
    figura.tight_layout()
    return figura


def validar_metadata_resultado_umap(
    tabla_umap: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Comprueba que UMAP conserve columnas, valores, tipos y orden de la metadata."""
    validar_columnas_metadata(tabla_umap, set(metadata.columns))
    esperada = metadata.reset_index(drop=True)
    obtenida = tabla_umap[list(metadata.columns)]
    if (
        not obtenida.equals(esperada)
        or obtenida.index.names != esperada.index.names
        or obtenida.columns.names != esperada.columns.names
    ):
        raise ValueError("La proyección debe conservar intacta la metadata y su orden.")


def obtener_muestra_resultado_umap(
    tabla_umap: pd.DataFrame,
    cantidad: int = 10,
) -> pd.DataFrame:
    """Selecciona identidad, clase, split y coordenadas de las primeras imágenes."""
    columnas = [
        "Identificador",
        "ruta_relativa",
        "clase_carpeta",
        "split",
        "UMAP_1",
        "UMAP_2",
    ]
    return obtener_muestra_metadata(tabla_umap[columnas], cantidad)


def guardar_figura_umap(figura: Figure, ruta_figura: Path) -> Path:
    """Guarda una figura PNG a 200 dpi, ajustando el borde al contenido."""
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_figura, dpi=200, bbox_inches="tight")
    return ruta_figura


def guardar_vistas_adicionales_umap(
    figuras: Iterable[tuple[str, Figure]],
    directorio_salida: Path,
    ruta_modelo: Path,
) -> tuple[Path, ...]:
    """Guarda las vistas con el nombre del modelo y devuelve las rutas en orden."""
    rutas = []
    for nombre, figura in figuras:
        ruta_figura = directorio_salida / f"umap_test_{nombre}_{ruta_modelo.stem}.png"
        rutas.append(guardar_figura_umap(figura, ruta_figura))
    return tuple(rutas)


def imprimir_archivos_generados(rutas: Iterable[Path]) -> None:
    """Informa los archivos guardados con rutas relativas a la raíz del proyecto."""
    for ruta in rutas:
        print(f"Archivo generado: {ruta_relativa(ruta)}")


def guardar_resultado_umap(
    tabla: pd.DataFrame,
    figura: Figure,
    ruta_csv: Path,
    ruta_figura: Path,
) -> tuple[Path, Path]:
    """Guarda metadata con coordenadas y la figura usada en el reporte."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    guardar_figura_umap(figura, ruta_figura)
    return ruta_csv, ruta_figura
