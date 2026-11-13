import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from vasijas.evaluacion import (
    DatasetImagenesSinClase,
    crear_loader_imagenes_sin_clase,
    crear_tabla_predicciones_con_metadata,
    predecir_imagenes_sin_clase,
    seleccionar_metadata_sin_clase,
    validar_predicciones_sin_clase,
)


class ModeloBrillo(torch.nn.Module):
    def forward(self, imagenes):
        brillo = imagenes.mean(dim=(1, 2, 3))
        return torch.stack([1 - brillo, brillo], dim=1)


class EvaluacionSinClaseTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame(
            {
                "Identificador": ["negro", "clasificado", "blanco", "gris"],
                "Clase Ana": [None, 3, "", "  "],
                "Sitio": ["A", "B", None, "D"],
            }
        )
        self.clases = ["clase01", "clase03"]

    def crear_imagen(self, raiz, nombre, color, carpeta="sin_clase"):
        ruta = raiz / carpeta / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(ruta)

    def test_seleccion_solo_nulos_y_vacios_sin_modificar_original(self):
        copia = self.metadata.copy(deep=True)
        seleccion = seleccionar_metadata_sin_clase(self.metadata)
        self.assertEqual(
            seleccion["Identificador"].tolist(), ["negro", "blanco", "gris"]
        )
        pd.testing.assert_frame_equal(self.metadata, copia)
        with self.assertRaisesRegex(ValueError, "Clase Ana"):
            seleccionar_metadata_sin_clase(self.metadata.drop(columns="Clase Ana"))

    def test_inferencia_cubre_ultimo_lote_y_conserva_identidad_y_metadata(self):
        seleccion = seleccionar_metadata_sin_clase(self.metadata).iloc[::-1]
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            self.crear_imagen(raiz, "blanco.png", "white")
            self.crear_imagen(raiz, "negro.png", "black", carpeta="otra_carpeta")
            self.crear_imagen(raiz, "gris.png", "gray")
            self.crear_imagen(raiz, "clasificado.png", "white")
            loader = crear_loader_imagenes_sin_clase(
                seleccion, raiz, ToTensor(), 2, torch.device("cpu")
            )
            predicciones = predecir_imagenes_sin_clase(
                ModeloBrillo(), loader, self.clases, torch.device("cpu")
            )
        validar_predicciones_sin_clase(predicciones, seleccion, self.clases)
        self.assertEqual(
            predicciones["Identificador"].tolist(), ["gris", "blanco", "negro"]
        )
        self.assertEqual(
            predicciones["clase_predicha"].tolist(), ["clase03", "clase03", "clase01"]
        )
        self.assertAlmostEqual(
            predicciones.loc[1, "confianza"], torch.sigmoid(torch.tensor(1.0)).item()
        )
        self.assertNotIn("acierto", predicciones)
        tabla = crear_tabla_predicciones_con_metadata(
            predicciones,
            self.metadata,
            ("ruta_relativa", "clase_predicha", "confianza"),
        )
        self.assertEqual(
            list(tabla),
            list(self.metadata) + ["ruta_relativa", "clase_predicha", "confianza"],
        )
        self.assertEqual(tabla["Clase Ana"].tolist()[0:2], ["  ", ""])
        self.assertTrue(pd.isna(tabla.iloc[2]["Clase Ana"]))

    def test_falla_si_falta_imagen_o_hay_identificador_ambiguo(self):
        seleccion = self.metadata.iloc[:1]
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            with self.assertRaisesRegex(FileNotFoundError, "negro"):
                DatasetImagenesSinClase(seleccion, raiz, ToTensor())
            self.crear_imagen(raiz, "negro.png", "black")
            self.crear_imagen(raiz, "negro.jpg", "black", carpeta="otra")
            with self.assertRaisesRegex(ValueError, "varias imágenes"):
                DatasetImagenesSinClase(seleccion, raiz, ToTensor())

    def test_validacion_rechaza_cobertura_clases_confianza_y_rutas_incorrectas(self):
        seleccion = self.metadata.iloc[:1]
        predicciones = pd.DataFrame(
            {
                "Identificador": ["negro"],
                "ruta_relativa": ["sin_clase/negro.png"],
                "clase_predicha": ["clase01"],
                "confianza": [0.9],
            }
        )
        casos = [
            predicciones.iloc[:0],
            pd.concat([predicciones, predicciones]),
            predicciones.assign(Identificador="otro"),
            predicciones.assign(ruta_relativa="sin_clase/otro.png"),
            predicciones.assign(clase_predicha="sin_clase"),
            predicciones.assign(confianza=1.1),
            predicciones.assign(confianza=float("nan")),
        ]
        for indice, caso in enumerate(casos):
            with self.subTest(caso=indice), self.assertRaises(ValueError):
                validar_predicciones_sin_clase(caso, seleccion, self.clases)

    def test_sin_perfiles_genera_tabla_vacia(self):
        seleccion = seleccionar_metadata_sin_clase(self.metadata.iloc[1:2])
        with tempfile.TemporaryDirectory() as temporal:
            loader = crear_loader_imagenes_sin_clase(
                seleccion, Path(temporal), ToTensor(), 2, torch.device("cpu")
            )
            predicciones = predecir_imagenes_sin_clase(
                ModeloBrillo(), loader, self.clases, torch.device("cpu")
            )
        validar_predicciones_sin_clase(predicciones, seleccion, self.clases)
        tabla = crear_tabla_predicciones_con_metadata(
            predicciones,
            self.metadata,
            ("ruta_relativa", "clase_predicha", "confianza"),
        )
        self.assertEqual(len(tabla), 0)


if __name__ == "__main__":
    unittest.main()
