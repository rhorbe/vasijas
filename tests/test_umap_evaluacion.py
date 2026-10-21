import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from vasijas.umap_evaluacion import (
    ConfiguracionUmap,
    extraer_caracteristicas_test,
    proyectar_caracteristicas_umap,
)


def test_extraccion_alinea_rutas_y_conserva_ultimo_lote_y_clasificador():
    metadata = pd.DataFrame(
        {
            "Identificador": ["a", "b", "c", "entrenamiento"],
            "ruta_relativa": [
                "clase01/a.png",
                "clase01/b.png",
                "clase01/c.png",
                "train.png",
            ],
            "clase_carpeta": ["clase01"] * 4,
            "split": ["test", "test", "test", "train"],
            "dato_opcional": [None, 2.0, 3.0, 4.0],
        }
    )
    # Orden distinto al CSV y último lote de una sola imagen.
    orden = [2, 0, 1]
    loader = DataLoader(
        [
            (torch.full((1, 2, 2), float(i)), 0, metadata.at[i, "ruta_relativa"])
            for i in orden
        ],
        batch_size=2,
    )
    modelo = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(1, 1))
    clasificador = modelo[-1]
    pesos = clasificador.weight.detach().clone()
    avances = []
    vectores, tabla = extraer_caracteristicas_test(
        modelo,
        loader,
        metadata,
        ["clase01"],
        torch.device("cpu"),
        informar_progreso=lambda actual, total: avances.append((actual, total)),
    )
    np.testing.assert_array_equal(vectores[:, 0], orden)
    pd.testing.assert_frame_equal(tabla, metadata.iloc[orden].reset_index(drop=True))
    assert avances == [(2, 3), (3, 3)]
    assert modelo[-1] is clasificador
    torch.testing.assert_close(clasificador.weight, pesos)
    assert modelo.training and modelo[0].training


def test_extraccion_rechaza_cobertura_incompleta():
    metadata = pd.DataFrame(
        {
            "Identificador": ["a", "b"],
            "ruta_relativa": ["a.png", "b.png"],
            "clase_carpeta": ["clase01", "clase01"],
            "split": ["test", "test"],
        }
    )
    loader = DataLoader([(torch.ones(1, 2, 2), 0, "a.png")], batch_size=2)
    modelo = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Linear(1, 1))
    with pytest.raises(ValueError, match="exactamente un vector"):
        extraer_caracteristicas_test(
            modelo,
            loader,
            metadata,
            ["clase01"],
            torch.device("cpu"),
        )


def test_proyeccion_no_usa_etiquetas_y_preserva_metadata(monkeypatch):
    vectores = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = pd.DataFrame(
        {
            "Identificador": ["c", "a", "b"],
            "split": ["test"] * 3,
            "clase_carpeta": ["clase02", "clase01", "clase02"],
            "dato_opcional": [None, 2, 3],
        },
        index=[9, 2, 7],
    )
    original = metadata.copy(deep=True)

    class UmapSinEtiquetas:
        def __init__(self, **kwargs):
            assert kwargs["n_components"] == 2
            assert kwargs["random_state"] == 42
            assert kwargs["n_jobs"] == 1

        def fit_transform(self, x):
            np.testing.assert_array_equal(x, vectores)
            return np.array([[2, 3], [4, 5], [6, 7]])

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=UmapSinEtiquetas))
    config = ConfiguracionUmap(n_neighbors=2)
    tabla = proyectar_caracteristicas_umap(vectores, metadata, config)
    pd.testing.assert_frame_equal(metadata, original)
    pd.testing.assert_frame_equal(
        tabla[metadata.columns], original.reset_index(drop=True)
    )
    np.testing.assert_array_equal(tabla[["UMAP_1", "UMAP_2"]], [[2, 3], [4, 5], [6, 7]])
    with pytest.raises(ValueError, match="solamente imágenes"):
        proyectar_caracteristicas_umap(vectores, metadata.assign(split="train"), config)
    with pytest.raises(ValueError, match="alineados"):
        proyectar_caracteristicas_umap(vectores[:2], metadata, config)
    with pytest.raises(ValueError, match="no finitos"):
        proyectar_caracteristicas_umap(vectores * np.nan, metadata, config)
