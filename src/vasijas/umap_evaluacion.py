"""Extracción de características y visualización exploratoria del test persistido."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from vasijas.evaluacion import indexar_metadata_por_ruta


@dataclass(frozen=True)
class ConfiguracionUmap:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42


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
    metadata_test = metadata_split.loc[metadata_split["split"].eq("test")].copy()
    requeridas = ["Identificador", "ruta_relativa", "clase_carpeta"]
    if metadata_test.empty or metadata_test[requeridas].isna().any().any():
        raise ValueError("El test debe contener imágenes con identidad, ruta y clase.")
    metadata_por_ruta = indexar_metadata_por_ruta(metadata_test)
    if not metadata_por_ruta.index.is_unique:
        raise ValueError("Las rutas de test deben ser únicas.")

    # Se reutilizan los pesos locales hasta avgpool; fc queda intacta.
    extractor = nn.Sequential(*list(modelo.children())[:-1], nn.Flatten(1))
    estaba_entrenando = modelo.training
    bloques, rutas_extraidas = [], []
    try:
        extractor.eval()
        with torch.inference_mode():
            for imagenes, etiquetas, rutas in test_loader:
                for ruta, etiqueta in zip(rutas, etiquetas.tolist(), strict=True):
                    if ruta not in metadata_por_ruta.index:
                        raise ValueError(f"La ruta {ruta!r} no pertenece al test.")
                    if (
                        metadata_por_ruta.at[ruta, "clase_carpeta"]
                        != nombres_clases[etiqueta]
                    ):
                        raise ValueError(
                            f"La clase de {ruta!r} no coincide con la metadata."
                        )
                vectores = extractor(imagenes.to(dispositivo))
                bloques.append(vectores.cpu().numpy())
                rutas_extraidas.extend(rutas)
                if informar_progreso is not None:
                    informar_progreso(len(rutas_extraidas), len(test_loader.dataset))
    finally:
        modelo.train(estaba_entrenando)

    if (
        len(rutas_extraidas) != len(metadata_test)
        or len(set(rutas_extraidas)) != len(rutas_extraidas)
        or set(rutas_extraidas) != set(metadata_por_ruta.index)
    ):
        raise ValueError("Debe existir exactamente un vector por imagen de test.")
    tabla = metadata_por_ruta.loc[rutas_extraidas].reset_index()
    tabla = tabla[list(metadata_split.columns)]
    return np.concatenate(bloques), tabla


def proyectar_caracteristicas_umap(
    caracteristicas: np.ndarray,
    metadata_test: pd.DataFrame,
    configuracion: ConfiguracionUmap,
) -> pd.DataFrame:
    """Ajusta UMAP solo sobre vectores de test, sin usar clases como objetivo.

    Las etiquetas y demás columnas se conservan para interpretar la proyección;
    ninguna columna de metadata interviene en el ajuste.
    """
    from umap import UMAP

    vectores = np.asarray(caracteristicas)
    if vectores.ndim != 2 or vectores.shape[1] == 0:
        raise ValueError("Las características deben ser una matriz de vectores.")
    if len(vectores) != len(metadata_test) or len(vectores) < 3:
        raise ValueError(
            "Se requieren al menos tres vectores alineados con la metadata."
        )
    if not np.isfinite(vectores).all():
        raise ValueError("Las características contienen valores no finitos.")
    if not metadata_test["split"].eq("test").all():
        raise ValueError("UMAP debe recibir solamente imágenes del split test.")
    if not 2 <= configuracion.n_neighbors < len(vectores):
        raise ValueError(
            "n_neighbors debe ser al menos 2 y menor que el tamaño del test."
        )
    if not 0 <= configuracion.min_dist <= 1:
        raise ValueError("min_dist debe estar entre cero y uno.")
    coordenadas = UMAP(
        n_components=2,
        n_neighbors=configuracion.n_neighbors,
        min_dist=configuracion.min_dist,
        metric=configuracion.metric,
        random_state=configuracion.random_state,
        n_jobs=1,
        # También permite proyectar conjuntos pequeños (tres muestras).
        init="random",
    ).fit_transform(vectores)
    if not np.isfinite(coordenadas).all():
        raise ValueError("UMAP produjo coordenadas no finitas.")
    tabla = metadata_test.copy().reset_index(drop=True)
    tabla[["UMAP_1", "UMAP_2"]] = coordenadas
    return tabla


def graficar_umap_test(tabla: pd.DataFrame) -> Figure:
    """Colorea cada imagen por su clase real con una leyenda discreta."""
    figura, eje = plt.subplots(figsize=(11, 8))
    clases = sorted(tabla["clase_carpeta"].unique())
    colores = plt.get_cmap("tab10" if len(clases) <= 10 else "tab20", len(clases))
    for indice, clase in enumerate(clases):
        grupo = tabla.loc[tabla["clase_carpeta"].eq(clase)]
        eje.scatter(
            grupo["UMAP_1"],
            grupo["UMAP_2"],
            s=14,
            alpha=0.65,
            color=colores(indice),
            label=f"{clase} (n={len(grupo)})",
            linewidths=0,
        )
    eje.set(
        title=f"UMAP de características ResNet18 · test (n={len(tabla)})",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
    )
    eje.legend(title="Clase real", bbox_to_anchor=(1.02, 1), loc="upper left")
    figura.tight_layout()
    return figura


def guardar_resultado_umap(
    tabla: pd.DataFrame,
    figura: Figure,
    ruta_csv: Path,
    ruta_figura: Path,
) -> tuple[Path, Path]:
    """Guarda metadata con coordenadas y la figura usada en el reporte."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    figura.savefig(ruta_figura, dpi=200, bbox_inches="tight")
    return ruta_csv, ruta_figura
