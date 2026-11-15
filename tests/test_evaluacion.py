import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from vasijas.evaluacion import (
    cargar_modelo_resnet18_evaluacion,
    crear_loader_test_con_ruta,
    crear_tabla_predicciones_con_metadata,
    guardar_tabla_predicciones,
    indexar_metadata_por_ruta,
    predecir_test_con_rutas,
    validar_predicciones_test,
)


class ImagenesDePrueba(Dataset):
    def __init__(self, raiz):
        self.samples = [(str(raiz / f"clase/{i}.png"), i % 2) for i in range(3)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, indice):
        # Los tensores ya son logits: Identity permite verificar toda la inferencia.
        return torch.tensor([0.0, 1.0]), self.samples[indice][1]


class EvaluacionTests(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame(
            {
                "Identificador": [10, 20, 30],
                "ruta_relativa": [f"clase/{i}.png" for i in range(3)],
                "split": ["test"] * 3,
            }
        )

    def predecir(self, raiz, metadata=None):
        subset = Subset(ImagenesDePrueba(raiz), [2, 0, 1])
        loader = crear_loader_test_con_ruta(
            subset, raiz, self.metadata, 2, torch.device("cpu")
        )
        progreso = []
        modelo = torch.nn.Identity()
        predicciones = predecir_test_con_rutas(
            modelo,
            loader,
            indexar_metadata_por_ruta(self.metadata if metadata is None else metadata),
            ["clase0", "clase1"],
            torch.device("cpu"),
            informar_progreso=lambda procesadas, total: progreso.append(
                (procesadas, total)
            ),
            intervalo_progreso=1,
        )
        self.assertFalse(modelo.training)
        self.assertEqual(progreso, [(2, 3), (3, 3)])
        return predicciones

    def test_inferencia_conserva_identidad_con_orden_distinto_y_lote_incompleto(self):
        with tempfile.TemporaryDirectory() as temporal:
            predicciones = self.predecir(Path(temporal), self.metadata.iloc[::-1])
        self.assertEqual(predicciones["Identificador"].tolist(), [30, 10, 20])
        self.assertEqual(
            predicciones["clase_real"].tolist(), ["clase0", "clase0", "clase1"]
        )
        self.assertEqual(predicciones["acierto"].tolist(), [False, False, True])
        validar_predicciones_test(predicciones, self.metadata)

    def test_validacion_rechaza_cobertura_identidad_y_aciertos_incorrectos(self):
        with tempfile.TemporaryDirectory() as temporal:
            predicciones = self.predecir(Path(temporal))
        casos = {
            "faltante": predicciones.iloc[:2],
            "duplicada": pd.concat([predicciones.iloc[:2], predicciones.iloc[:1]]),
            "ruta ajena": predicciones.assign(
                ruta_relativa=["otra.png", "clase/0.png", "clase/1.png"]
            ),
            "identificador incorrecto": predicciones.assign(Identificador=[10, 30, 20]),
            "aciertos incorrectos": predicciones.assign(acierto=True),
            "nulos": predicciones.assign(clase_predicha=None),
            "columnas faltantes": predicciones.drop(columns="clase_real"),
        }
        for caso, tabla in casos.items():
            with self.subTest(caso=caso), self.assertRaises(ValueError):
                validar_predicciones_test(tabla, self.metadata)

    def test_loader_rechaza_tamano_incorrecto(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            with self.assertRaisesRegex(ValueError, "tamaño"):
                crear_loader_test_con_ruta(
                    Subset(ImagenesDePrueba(raiz), [0]),
                    raiz,
                    self.metadata,
                    2,
                    torch.device("cpu"),
                )

    def test_union_y_exportacion_conservan_metadata_nulos_y_orden(self):
        original = pd.DataFrame({"Identificador": [20, 10], "Sitio": [None, "Cañadón"]})
        predicciones = pd.DataFrame(
            {
                "Identificador": [10, 20],
                "clase_predicha": ["a", "b"],
                "acierto": [True, False],
            }
        )
        copia = original.copy(deep=True)
        tabla = crear_tabla_predicciones_con_metadata(predicciones, original)
        self.assertEqual(
            list(tabla), ["Identificador", "Sitio", "clase_predicha", "acierto"]
        )
        self.assertEqual(tabla["Identificador"].tolist(), [10, 20])
        self.assertEqual(tabla.loc[0, "Sitio"], "Cañadón")
        self.assertTrue(pd.isna(tabla.loc[1, "Sitio"]))
        pd.testing.assert_frame_equal(original, copia)
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "predicciones" / "test.csv"
            self.assertEqual(guardar_tabla_predicciones(tabla, ruta), ruta)
            self.assertTrue(ruta.read_bytes().startswith(b"\xef\xbb\xbf"))
            pd.testing.assert_frame_equal(pd.read_csv(ruta), tabla)

    def test_union_rechaza_metadata_faltante_o_identificadores_duplicados(self):
        original = pd.DataFrame({"Identificador": [10, 20]})
        for ids in ([10, 30], [10, 10], [10, None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                crear_tabla_predicciones_con_metadata(
                    pd.DataFrame(
                        {
                            "Identificador": ids,
                            "clase_predicha": ["a", "b"],
                            "acierto": [True, False],
                        }
                    ),
                    original,
                )

    def test_carga_modelo_valida_clases_y_no_descarga_pesos(self):
        class ModeloPequeno(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(2, 2)

        modelo = ModeloPequeno()
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "modelo.pth"
            torch.save(modelo.state_dict(), ruta)
            with patch(
                "vasijas.evaluacion.models.resnet18",
                side_effect=lambda **kwargs: ModeloPequeno(),
            ) as crear:
                cargado = cargar_modelo_resnet18_evaluacion(
                    ruta, ["a", "b"], torch.device("cpu")
                )
            crear.assert_called_once_with(weights=None)
            self.assertFalse(cargado.training)
            torch.testing.assert_close(cargado.fc.weight, modelo.fc.weight)
            with self.assertRaisesRegex(ValueError, "salidas"):
                cargar_modelo_resnet18_evaluacion(ruta, ["a"], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
