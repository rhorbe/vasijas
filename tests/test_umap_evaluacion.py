import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from matplotlib.colors import to_rgba
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from vasijas.umap_evaluacion import (
    ConfiguracionUmap,
    cargar_resnet18_entrenada,
    crear_extractor_resnet18,
    crear_tabla_umap,
    extraer_caracteristicas_por_lotes,
    extraer_caracteristicas_test,
    graficar_umap_por_origen,
    graficar_umap_por_porcentaje,
    graficar_umap_test,
    guardar_resultado_umap,
    guardar_vistas_adicionales_umap,
    imprimir_archivos_generados,
    imprimir_resumen_fracciones,
    obtener_limites_grafico_umap,
    obtener_muestra_resultado_umap,
    preparar_datos_vistas_fracciones,
    proyectar_caracteristicas_umap,
    validar_caracteristicas_resnet18,
    validar_configuracion_umap,
    validar_coordenadas_umap,
    validar_metadata,
    validar_metadata_resultado_umap,
    validar_metadata_test,
    validar_orden_rutas_extraidas,
)


def test_carga_resnet18_restaura_pesos_clases_y_modo_evaluacion(tmp_path):
    original = models.resnet18(weights=None)
    original.fc = nn.Linear(original.fc.in_features, 3)
    ruta = tmp_path / "modelo.pth"
    torch.save(original.state_dict(), ruta)

    modelo = cargar_resnet18_entrenada(ruta, torch.device("cpu"))

    assert modelo.fc.out_features == 3
    assert all(not capa.training for capa in modelo.modules())
    assert next(modelo.parameters()).device.type == "cpu"
    for nombre, pesos in original.state_dict().items():
        torch.testing.assert_close(modelo.state_dict()[nombre], pesos)


def test_extractor_excluye_clasificador_y_aplana_avgpool():
    modelo = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Linear(512, 3))
    clasificador = modelo[-1]
    extractor = crear_extractor_resnet18(modelo)

    entrada = torch.arange(1024.0).reshape(2, 512, 1, 1).expand(-1, -1, 2, 2)
    torch.testing.assert_close(extractor(entrada), entrada[:, :, 0, 0])
    assert modelo[-1] is clasificador
    assert all(capa is not clasificador for capa in extractor.modules())
    assert not extractor.training


def test_extraccion_por_lotes_preserva_orden_inferencia_y_progreso(capsys):
    class ExtractorEnInferencia(nn.Module):
        def forward(self, imagenes):
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            return imagenes.flatten(1)

    rutas = [f"imagen_{i}.png" for i in reversed(range(201))]
    loader = DataLoader(
        [(torch.full((512, 1, 1), float(i)), ruta) for i, ruta in enumerate(rutas)],
        batch_size=2,
    )
    vectores, extraidas = extraer_caracteristicas_por_lotes(
        ExtractorEnInferencia(), loader, torch.device("cpu")
    )

    assert extraidas == rutas
    assert vectores.shape == (201, 512)
    np.testing.assert_array_equal(
        vectores, np.repeat(np.arange(201)[:, None], 512, axis=1)
    )
    assert capsys.readouterr().out.splitlines() == [
        "Características: 200/201",
        "Características: 201/201",
    ]


def test_extraccion_por_lotes_rechaza_loader_vacio():
    with pytest.raises(ValueError, match="ningún lote"):
        extraer_caracteristicas_por_lotes(
            nn.Identity(), DataLoader([]), torch.device("cpu")
        )


@pytest.mark.parametrize("rutas", [["b", "a"], ["a"], ["a", "a"], ["a", "b", "c"]])
def test_validacion_orden_rechaza_rutas_desalineadas(rutas):
    metadata = pd.DataFrame({"ruta_relativa": ["a", "b"]})
    with pytest.raises(ValueError, match="orden completo"):
        validar_orden_rutas_extraidas(rutas, metadata)


def test_validaciones_aceptan_vectores_alineados():
    validar_orden_rutas_extraidas(
        ["a", "b"], pd.DataFrame({"ruta_relativa": ["a", "b"]})
    )
    validar_caracteristicas_resnet18(np.ones((2, 512)), 2)


@pytest.mark.parametrize("forma", [(1, 512), (2, 511), (2, 512, 1)])
def test_validacion_caracteristicas_rechaza_dimensiones_incorrectas(forma):
    with pytest.raises(ValueError, match="512 características"):
        validar_caracteristicas_resnet18(np.ones(forma), 2)


@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_validacion_caracteristicas_rechaza_valores_no_finitos(valor):
    vectores = np.ones((2, 512))
    vectores[0, 0] = valor
    with pytest.raises(ValueError, match="no finitos"):
        validar_caracteristicas_resnet18(vectores, 2)


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
            assert kwargs == {
                "n_components": 2,
                "n_neighbors": 2,
                "min_dist": 0.1,
                "metric": "cosine",
                "random_state": 42,
                "n_jobs": 1,
                "init": "random",
            }

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


@pytest.mark.parametrize("vecinos", [1, 3, 4])
def test_configuracion_umap_rechaza_vecinos_fuera_de_rango(vecinos):
    with pytest.raises(ValueError, match="n_neighbors debe ser"):
        validar_configuracion_umap(ConfiguracionUmap(n_neighbors=vecinos), 3)


@pytest.mark.parametrize("distancia", [-0.1, 1.1, np.nan])
def test_configuracion_umap_rechaza_distancia_fuera_de_rango(distancia):
    with pytest.raises(ValueError, match="min_dist debe estar"):
        validar_configuracion_umap(
            ConfiguracionUmap(n_neighbors=2, min_dist=distancia), 3
        )


@pytest.mark.parametrize("distancia", [0.0, 1.0])
def test_configuracion_umap_acepta_limites_de_distancia(distancia):
    validar_configuracion_umap(ConfiguracionUmap(n_neighbors=2, min_dist=distancia), 3)


@pytest.mark.parametrize(
    "coordenadas",
    [
        np.ones((3, 1)),
        np.ones((2, 2)),
        np.ones((3, 2, 1)),
        np.full((3, 2), np.nan),
        np.full((3, 2), np.inf),
    ],
)
def test_proyeccion_rechaza_salida_umap_invalida(monkeypatch, coordenadas):
    class UmapConSalidaInvalida:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, x):
            return coordenadas

    monkeypatch.setitem(
        sys.modules, "umap", SimpleNamespace(UMAP=UmapConSalidaInvalida)
    )
    with pytest.raises(ValueError, match="dos coordenadas finitas por imagen"):
        proyectar_caracteristicas_umap(
            np.ones((3, 4)),
            pd.DataFrame({"split": ["test"] * 3}),
            ConfiguracionUmap(n_neighbors=2),
        )


def test_tabla_umap_conserva_indice_orden_y_metadata_original():
    metadata = pd.DataFrame({"clase_carpeta": ["b", "a"]}, index=[8, 3])
    original = metadata.copy(deep=True)
    coordenadas = np.array([[1, 2], [3, 4]])
    validar_coordenadas_umap(coordenadas, 2)
    tabla = crear_tabla_umap(metadata, coordenadas)
    pd.testing.assert_frame_equal(tabla[metadata.columns], original)
    np.testing.assert_array_equal(tabla[["UMAP_1", "UMAP_2"]], coordenadas)
    pd.testing.assert_frame_equal(metadata, original)


def test_figura_umap_conserva_titulo_y_leyenda_interior():
    tabla = pd.DataFrame(
        {
            "clase_carpeta": ["clase02", "clase01", "clase01"],
            "UMAP_1": [0, 1, 2],
            "UMAP_2": [2, 1, 0],
        }
    )
    figura = graficar_umap_test(tabla, posicion_leyenda="interior")
    try:
        eje = figura.axes[0]
        assert eje.get_title() == "UMAP de características ResNet18 · test (n=3)"
        leyenda = eje.get_legend()
        assert leyenda.get_title().get_text() == "Clase real"
        assert [texto.get_text() for texto in leyenda.get_texts()] == [
            "clase01 (n=2)",
            "clase02 (n=1)",
        ]
        assert leyenda._loc == 0  # Matplotlib: loc="best".
        assert leyenda.markerscale == 2.5
        assert leyenda.get_frame_on()
        assert leyenda.get_frame().get_alpha() == 0.9
        assert leyenda.get_frame().get_facecolor()[:3] == (1.0, 1.0, 1.0)
    finally:
        plt.close(figura)


@pytest.fixture
def tabla_vistas_fracciones():
    return pd.DataFrame(
        {
            "split": ["test"] * 3,
            "borde_origen": [" Completo ", " TOPE", "Base "],
            "porcentaje_avance": [100.0, 25.0, 75.0],
            "UMAP_1": [2.0, 1.0, 3.0],
            "UMAP_2": [4.0, 6.0, 5.0],
            "dato_opcional": [None, "a", "b"],
        },
        index=[9, 2, 7],
    )


def test_preparacion_fracciones_normaliza_sin_alterar_tabla(tabla_vistas_fracciones):
    tabla = tabla_vistas_fracciones
    tabla["porcentaje_avance"] = tabla["porcentaje_avance"].astype(str)
    original = tabla.copy(deep=True)
    origen, porcentaje, fracciones = preparar_datos_vistas_fracciones(tabla)

    assert origen.tolist() == ["completo", "tope", "base"]
    assert porcentaje.tolist() == [100.0, 25.0, 75.0]
    assert origen.index.tolist() == porcentaje.index.tolist() == [9, 2, 7]
    pd.testing.assert_frame_equal(fracciones, original.iloc[1:])
    pd.testing.assert_frame_equal(tabla, original)


@pytest.mark.parametrize(
    "modificar, mensaje",
    [
        (
            lambda t: t.drop(columns=["split", "UMAP_2"]),
            "Faltan columnas para las vistas de fracciones: ['UMAP_2', 'split']",
        ),
        (
            lambda t: t.iloc[:0],
            "Se requieren puntos y metadatos completos para colorear.",
        ),
        (
            lambda t: t.assign(borde_origen=None),
            "Se requieren puntos y metadatos completos para colorear.",
        ),
        (
            lambda t: t.assign(split="train"),
            "La metadata debe contener solamente imágenes de los conjuntos seleccionados: ['test'].",
        ),
        (lambda t: t.assign(UMAP_1=np.inf), "Las coordenadas deben ser finitas."),
        (lambda t: t.assign(borde_origen="otro"), "Orígenes desconocidos: ['otro']"),
        (
            lambda t: t.assign(porcentaje_avance=np.inf),
            "Los porcentajes deben ser finitos.",
        ),
        (
            lambda t: t.assign(porcentaje_avance=[99, 25, 75]),
            "Las imágenes completas deben indicar 100 %.",
        ),
        (
            lambda t: t.assign(porcentaje_avance=[100, 0, 75]),
            "Los recortes deben indicar un porcentaje entre 0 y 100 exclusivo.",
        ),
        (
            lambda t: t.assign(porcentaje_avance=[100, 25, 100]),
            "Los recortes deben indicar un porcentaje entre 0 y 100 exclusivo.",
        ),
        (
            lambda t: t.iloc[:2],
            "Se necesitan fracciones de tope y de base para comparar.",
        ),
        (
            lambda t: t.iloc[[0, 2]],
            "Se necesitan fracciones de tope y de base para comparar.",
        ),
    ],
)
def test_preparacion_fracciones_conserva_errores(
    tabla_vistas_fracciones, modificar, mensaje
):
    with pytest.raises(ValueError) as error:
        preparar_datos_vistas_fracciones(modificar(tabla_vistas_fracciones))
    assert str(error.value) == mensaje


def test_preparacion_fracciones_rechaza_porcentajes_no_numericos(
    tabla_vistas_fracciones,
):
    with pytest.raises(ValueError):
        preparar_datos_vistas_fracciones(
            tabla_vistas_fracciones.assign(porcentaje_avance="desconocido")
        )


def test_preparacion_fracciones_permite_ausencia_de_completas(tabla_vistas_fracciones):
    tabla = tabla_vistas_fracciones.iloc[1:]
    _, _, fracciones = preparar_datos_vistas_fracciones(tabla)
    pd.testing.assert_frame_equal(fracciones, tabla)


def test_resumen_fracciones_conserva_mensajes(tabla_vistas_fracciones, capsys):
    origen, _, _ = preparar_datos_vistas_fracciones(tabla_vistas_fracciones)
    imprimir_resumen_fracciones(origen)
    assert capsys.readouterr().out.splitlines() == [
        "completo: 1 imágenes de test",
        "tope: 1 imágenes de test",
        "base: 1 imágenes de test",
        "Fracciones: 2 · Total: 3",
    ]


def test_limites_umap_se_reutilizan_incluso_tras_cerrar_figura():
    figura, eje = plt.subplots()
    try:
        eje.set(xlim=(-3, 5), ylim=(-7, 11))
    finally:
        plt.close(figura)
    assert obtener_limites_grafico_umap(figura) == ((-3, 5), (-7, 11))


def test_grafico_por_origen_conserva_puntos_estilo_y_limites(tabla_vistas_fracciones):
    tabla = pd.concat(
        [
            tabla_vistas_fracciones,
            tabla_vistas_fracciones.loc[[2]].rename(index={2: 4}),
        ]
    )
    original = tabla.copy(deep=True)
    origen, _, fracciones = preparar_datos_vistas_fracciones(tabla)
    origen_original = origen.copy(deep=True)
    limites_x, limites_y = (-10.0, 20.0), (-30.0, 40.0)

    figura = graficar_umap_por_origen(tabla, origen, limites_x, limites_y)
    try:
        np.testing.assert_array_equal(figura.get_size_inches(), [11, 8])
        eje = figura.axes[0]
        assert obtener_limites_grafico_umap(figura) == (limites_x, limites_y)
        assert eje.get_title() == "UMAP test · fracciones por origen (n=3)"
        assert eje.get_xlabel() == "UMAP 1"
        assert eje.get_ylabel() == "UMAP 2"
        assert len(eje.collections) == 2
        assert sum(len(grupo.get_offsets()) for grupo in eje.collections) == len(
            fracciones
        )
        for grupo, categoria, etiqueta, color in zip(
            eje.collections,
            ["tope", "base"],
            ["Tope (borde) (n=2)", "Base (n=1)"],
            ["#E69F00", "#0072B2"],
            strict=True,
        ):
            esperadas = tabla.loc[origen.eq(categoria), ["UMAP_1", "UMAP_2"]]
            np.testing.assert_array_equal(grupo.get_offsets(), esperadas.to_numpy())
            np.testing.assert_allclose(grupo.get_facecolors()[0], to_rgba(color, 0.65))
            np.testing.assert_array_equal(grupo.get_sizes(), [14])
            np.testing.assert_array_equal(grupo.get_linewidths(), [0])
            assert grupo.get_label() == etiqueta
        leyenda = eje.get_legend()
        assert leyenda.get_title().get_text() == "Origen del recorte"
        assert [texto.get_text() for texto in leyenda.get_texts()] == [
            "Tope (borde) (n=2)",
            "Base (n=1)",
        ]
        assert leyenda._loc == 0  # Matplotlib: loc="best".
        assert leyenda.markerscale == 2.5
        assert leyenda.get_frame_on()
        assert leyenda.get_frame().get_alpha() == 0.9
        assert leyenda.get_frame().get_facecolor()[:3] == (1.0, 1.0, 1.0)
        pd.testing.assert_frame_equal(tabla, original)
        pd.testing.assert_series_equal(origen, origen_original)
    finally:
        plt.close(figura)


@pytest.mark.parametrize(
    "valores, orden_esperado",
    [
        ([100, 25, 25, 75, 100, 50], [1, 2, 5, 3, 0, 4]),
        ([100] * 6, [0, 1, 2, 3, 4, 5]),
    ],
)
def test_grafico_por_porcentaje_conserva_orden_estable_escala_y_limites(
    valores, orden_esperado
):
    tabla = pd.DataFrame(
        {"UMAP_1": [2, 1, 4, 3, 5, 6], "UMAP_2": [4, 6, 5, 7, 8, 9]},
        index=[9, 2, 7, 4, 8, 1],
    )
    porcentaje = pd.Series(valores, index=tabla.index, name="porcentaje_avance")
    tabla_original = tabla.copy(deep=True)
    porcentaje_original = porcentaje.copy(deep=True)
    limites_x, limites_y = (-10.0, 20.0), (-30.0, 40.0)

    figura = graficar_umap_por_porcentaje(tabla, porcentaje, limites_x, limites_y)
    try:
        np.testing.assert_array_equal(figura.get_size_inches(), [11, 8])
        eje = figura.axes[0]
        assert obtener_limites_grafico_umap(figura) == (limites_x, limites_y)
        assert eje.get_title() == "UMAP test · porcentaje conservado (n=6)"
        assert eje.get_xlabel() == "UMAP 1"
        assert eje.get_ylabel() == "UMAP 2"
        assert len(eje.collections) == 1
        puntos = eje.collections[0]
        np.testing.assert_array_equal(
            puntos.get_offsets(), tabla.iloc[orden_esperado].to_numpy()
        )
        np.testing.assert_array_equal(
            puntos.get_array(), porcentaje.iloc[orden_esperado].to_numpy()
        )
        assert puntos.get_cmap().name == "Blues"
        assert (puntos.norm.vmin, puntos.norm.vmax) == (0, 100)
        assert puntos.get_alpha() == 1.0
        np.testing.assert_array_equal(puntos.get_sizes(), [14])
        np.testing.assert_array_equal(puntos.get_linewidths(), [0])
        barra = puntos.colorbar
        assert len(figura.axes) == 2
        assert (
            barra.ax.get_ylabel() == "Perfil conservado (%) · 100 % = imagen completa"
        )
        np.testing.assert_array_equal(barra.get_ticks(), [0, 25, 50, 75, 100])
        assert [texto.get_text() for texto in barra.ax.get_yticklabels()] == [
            "0 %",
            "25 %",
            "50 %",
            "75 %",
            "100 %",
        ]
        pd.testing.assert_frame_equal(tabla, tabla_original)
        pd.testing.assert_series_equal(porcentaje, porcentaje_original)
    finally:
        plt.close(figura)


def test_validacion_metadata_resultado_conserva_valores_tipos_y_orden():
    metadata = pd.DataFrame(
        {"Identificador": ["b", "a"], "dato": [None, 2.0]}, index=[8, 3]
    )
    tabla = metadata.reset_index(drop=True).assign(UMAP_1=[1, 2], UMAP_2=[3, 4])
    validar_metadata_resultado_umap(tabla, metadata)
    with pytest.raises(ValueError, match="metadata y su orden"):
        validar_metadata_resultado_umap(
            tabla.iloc[::-1].reset_index(drop=True), metadata
        )
    with pytest.raises(ValueError, match="metadata y su orden"):
        validar_metadata_resultado_umap(
            tabla.assign(Identificador=["b", "c"]), metadata
        )
    with pytest.raises(ValueError, match="metadata y su orden"):
        validar_metadata_resultado_umap(
            tabla.assign(dato=tabla["dato"].astype("float32")), metadata
        )


@pytest.mark.parametrize("cantidad", [10, 2, 0])
def test_muestra_resultado_selecciona_columnas_y_primeras_filas(cantidad):
    columnas = [
        "Identificador",
        "ruta_relativa",
        "clase_carpeta",
        "split",
        "UMAP_1",
        "UMAP_2",
    ]
    tabla = pd.DataFrame({columna: list(range(12)) for columna in reversed(columnas)})
    tabla["extra"] = "dato"
    original = tabla.copy(deep=True)
    muestra = obtener_muestra_resultado_umap(tabla, cantidad)
    assert list(muestra.columns) == columnas
    assert len(muestra) == cantidad
    assert muestra["Identificador"].tolist() == list(range(cantidad))
    pd.testing.assert_frame_equal(tabla, original)


def test_guardado_resultado_conserva_csv_y_png(tmp_path):
    tabla = pd.DataFrame({"Identificador": ["á"], "UMAP_1": [1.0], "UMAP_2": [2.0]})
    ruta_csv = tmp_path / "datos" / "resultado.csv"
    ruta_png = tmp_path / "figuras" / "resultado.png"
    figura, _ = plt.subplots(figsize=(1, 1))
    plt.close(figura)
    rutas = guardar_resultado_umap(tabla, figura, ruta_csv, ruta_png)
    assert rutas == (ruta_csv, ruta_png)
    assert ruta_csv.read_bytes().startswith(b"\xef\xbb\xbf")
    pd.testing.assert_frame_equal(pd.read_csv(ruta_csv), tabla)
    with Image.open(ruta_png) as imagen:
        assert imagen.format == "PNG"
        assert imagen.info["dpi"] == pytest.approx((200, 200), abs=0.01)


def test_guardado_vistas_adicionales_conserva_nombres_y_orden(tmp_path):
    figura, _ = plt.subplots(figsize=(1, 1))
    plt.close(figura)
    destino = tmp_path / "vistas"
    rutas = guardar_vistas_adicionales_umap(
        [("origen_fracciones", figura), ("porcentaje", figura)],
        destino,
        Path("model_resnet18.pth"),
    )
    assert rutas == (
        destino / "umap_test_origen_fracciones_model_resnet18.png",
        destino / "umap_test_porcentaje_model_resnet18.png",
    )
    for ruta in rutas:
        with Image.open(ruta) as imagen:
            assert imagen.format == "PNG"
            assert imagen.info["dpi"] == pytest.approx((200, 200), abs=0.01)


def test_impresion_archivos_generados_conserva_orden_y_rutas_relativas(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "vasijas.umap_evaluacion.ruta_relativa", lambda ruta: ruta.relative_to(tmp_path)
    )
    relativas = [Path("output/resultado.csv"), Path("output/figura.png")]
    imprimir_archivos_generados(tmp_path / ruta for ruta in relativas)
    assert capsys.readouterr().out.splitlines() == [
        f"Archivo generado: {ruta}" for ruta in relativas
    ]


@pytest.mark.parametrize(
    "caso", ["valido", "ruta_ajena", "clase_incorrecta", "duplicada"]
)
def test_extraccion_etiquetada_restaura_modos_mixtos_y_valida_identidad(caso):
    metadata = pd.DataFrame(
        {
            "Identificador": ["a"],
            "ruta_relativa": ["a.png"],
            "clase_carpeta": ["clase01"],
            "split": ["test"],
        }
    )
    ruta = "ajena.png" if caso == "ruta_ajena" else "a.png"
    etiqueta = 1 if caso == "clase_incorrecta" else 0
    muestras = [(torch.ones(1, 2, 2), etiqueta, ruta)]
    if caso == "duplicada":
        muestras *= 2
    loader = DataLoader(muestras, batch_size=1)
    modelo = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(1, 2))
    modelo.train()
    modelo[0].eval()
    modos_originales = [capa.training for capa in modelo.modules()]
    pesos_originales = {
        nombre: peso.clone() for nombre, peso in modelo.state_dict().items()
    }

    def extraer():
        return extraer_caracteristicas_test(
            modelo, loader, metadata, ["clase01", "clase02"], torch.device("cpu")
        )

    if caso == "valido":
        vectores, tabla = extraer()
        assert vectores.shape == (1, 1)
        pd.testing.assert_frame_equal(tabla, metadata)
    else:
        mensaje = {
            "ruta_ajena": "no pertenece al test",
            "clase_incorrecta": "no coincide con la metadata",
            "duplicada": "exactamente un vector",
        }[caso]
        with pytest.raises(ValueError, match=mensaje):
            extraer()
    assert [capa.training for capa in modelo.modules()] == modos_originales
    for nombre, peso in modelo.state_dict().items():
        torch.testing.assert_close(peso, pesos_originales[nombre])


def test_extraccion_restaura_modos_si_falla_el_modelo():
    class CapaConFallo(nn.Module):
        def forward(self, imagenes):
            raise RuntimeError("Falló la inferencia")

    modelo = nn.Sequential(nn.Dropout(), CapaConFallo(), nn.Linear(1, 1))
    modelo[0].eval()
    modos_originales = [capa.training for capa in modelo.modules()]
    metadata = pd.DataFrame(
        {
            "Identificador": ["a"],
            "ruta_relativa": ["a.png"],
            "clase_carpeta": ["clase01"],
            "split": ["test"],
        }
    )
    loader = DataLoader([(torch.ones(1, 2, 2), 0, "a.png")])
    with pytest.raises(RuntimeError, match="Falló la inferencia"):
        extraer_caracteristicas_test(
            modelo, loader, metadata, ["clase01"], torch.device("cpu")
        )
    assert [capa.training for capa in modelo.modules()] == modos_originales


@pytest.mark.parametrize(
    "modificar, tipo_error, mensaje",
    [
        (
            lambda t: t.assign(split="train"),
            ValueError,
            "solamente imágenes de los conjuntos seleccionados",
        ),
        (lambda t: t.drop(columns="split"), ValueError, "Faltan columnas requeridas"),
        (
            lambda t: t.assign(clase_carpeta=None),
            ValueError,
            "identificador, ruta, clase y split",
        ),
        (
            lambda t: t.assign(ruta_relativa="../fuera.png"),
            ValueError,
            "dentro de DATA_DIR",
        ),
        (
            lambda t: t.assign(ruta_relativa="faltante.png"),
            FileNotFoundError,
            "Faltan 1 imágenes",
        ),
        (
            lambda t: pd.concat([t, t.assign(Identificador="b")]),
            ValueError,
            "rutas.*únicas",
        ),
    ],
)
def test_validacion_metadata_test_completa_antes_de_cargar_archivos(
    tmp_path, modificar, tipo_error, mensaje
):
    metadata = pd.DataFrame(
        {
            "Identificador": ["a"],
            "ruta_relativa": ["a.png"],
            "clase_carpeta": ["clase01"],
            "split": ["test"],
        }
    )
    (tmp_path / "a.png").touch()
    validar_metadata_test(metadata, set(), tmp_path)
    with pytest.raises(tipo_error, match=mensaje):
        validar_metadata_test(modificar(metadata), set(), tmp_path)


def test_grafico_clases_conserva_leyenda_exterior_por_defecto():
    tabla = pd.DataFrame({"clase_carpeta": ["a"], "UMAP_1": [0], "UMAP_2": [0]})
    figura = graficar_umap_test(tabla)
    try:
        assert figura.axes[0].get_legend()._loc == 2  # upper left
    finally:
        plt.close(figura)
    with pytest.raises(ValueError, match="posición de la leyenda"):
        graficar_umap_test(tabla, posicion_leyenda="desconocida")


@pytest.fixture
def metadata_todos_los_conjuntos(tmp_path):
    metadata = pd.DataFrame(
        {
            "Identificador": ["a", "b", "c"],
            "ruta_relativa": ["a.png", "b.png", "c.png"],
            "clase_carpeta": ["clase01", "clase02", "clase01"],
            "split": ["train", "val", "test"],
            "borde_origen": ["completo", "tope", "base"],
            "porcentaje_avance": [100, 25, 75],
            "dato_opcional": [None, None, None],
        }
    )
    for ruta in metadata["ruta_relativa"]:
        (tmp_path / ruta).touch()
    return metadata


@pytest.mark.parametrize(
    "conjuntos",
    [
        None,
        ["train", "val", "test"],
        ("train", "val", "test"),
        {"train", "val", "test"},
    ],
)
def test_validacion_generica_acepta_todos_los_conjuntos_sin_mutarlos(
    metadata_todos_los_conjuntos, tmp_path, conjuntos
):
    metadata = metadata_todos_los_conjuntos
    original = metadata.copy(deep=True)
    validar_metadata(
        metadata, {"borde_origen", "porcentaje_avance"}, tmp_path, conjuntos=conjuntos
    )
    pd.testing.assert_frame_equal(metadata, original)


def test_validacion_generica_restringe_conjuntos_y_columnas_requeridas(
    metadata_todos_los_conjuntos, tmp_path
):
    metadata = metadata_todos_los_conjuntos
    validar_metadata(metadata.iloc[:1], set(), tmp_path, conjuntos="train")
    with pytest.raises(ValueError, match="conjuntos seleccionados"):
        validar_metadata(metadata, set(), tmp_path, conjuntos="test")
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        validar_metadata(metadata, {"columna_ausente"}, tmp_path)
    with pytest.raises(ValueError, match="valores no nulos"):
        validar_metadata(metadata, {"dato_opcional"}, tmp_path)
    with pytest.raises(ValueError, match="no contiene imágenes"):
        validar_metadata(metadata.iloc[:0], set(), tmp_path)


def test_flujo_unificado_valida_proyecta_y_prepara_todos_los_splits(
    metadata_todos_los_conjuntos, tmp_path, monkeypatch
):
    metadata = metadata_todos_los_conjuntos
    conjuntos = ["train", "val", "test"]
    caracteristicas = np.arange(12).reshape(3, 4)

    class UmapUnificado:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, vectores):
            np.testing.assert_array_equal(vectores, caracteristicas)
            return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=UmapUnificado))
    validar_metadata(metadata, set(), tmp_path, conjuntos=conjuntos)
    tabla = proyectar_caracteristicas_umap(
        caracteristicas, metadata, ConfiguracionUmap(n_neighbors=2), conjuntos=conjuntos
    )
    validar_metadata_resultado_umap(tabla, metadata)
    assert tabla["split"].tolist() == conjuntos
    origen, porcentaje, fracciones = preparar_datos_vistas_fracciones(
        tabla, conjuntos=conjuntos
    )
    assert origen.tolist() == ["completo", "tope", "base"]
    assert porcentaje.tolist() == [100, 25, 75]
    assert fracciones["split"].tolist() == ["val", "test"]
    # La protección de los notebooks exclusivos de test sigue activa.
    with pytest.raises(ValueError, match="conjuntos seleccionados"):
        preparar_datos_vistas_fracciones(tabla)
    with pytest.raises(ValueError, match="conjuntos seleccionados"):
        proyectar_caracteristicas_umap(
            caracteristicas, metadata, ConfiguracionUmap(n_neighbors=2)
        )


def test_vistas_unificadas_admiten_titulos_y_prefijo_propio(
    metadata_todos_los_conjuntos, tmp_path, capsys
):
    tabla = crear_tabla_umap(
        metadata_todos_los_conjuntos, np.array([[1, 2], [3, 4], [5, 6]])
    )
    origen, porcentaje, _ = preparar_datos_vistas_fracciones(tabla, conjuntos=None)
    figuras = []
    try:
        figuras.append(graficar_umap_test(tabla, titulo="UMAP unificado por clase"))
        figuras.append(
            graficar_umap_por_origen(
                tabla, origen, (0, 6), (0, 7), titulo="UMAP unificado por origen"
            )
        )
        figuras.append(
            graficar_umap_por_porcentaje(
                tabla,
                porcentaje,
                (0, 6),
                (0, 7),
                titulo="UMAP unificado por porcentaje",
            )
        )
        assert [figura.axes[0].get_title() for figura in figuras] == [
            "UMAP unificado por clase",
            "UMAP unificado por origen",
            "UMAP unificado por porcentaje",
        ]
        rutas = guardar_vistas_adicionales_umap(
            [("porcentaje", figuras[2])],
            tmp_path,
            Path("modelo.pth"),
            prefijo="umap_unificado",
        )
        assert rutas == (tmp_path / "umap_unificado_porcentaje_modelo.png",)
        assert rutas[0].is_file()
        imprimir_resumen_fracciones(origen, descripcion_conjunto="seleccionadas")
        assert "imágenes de test" not in capsys.readouterr().out
    finally:
        for figura in figuras:
            plt.close(figura)
