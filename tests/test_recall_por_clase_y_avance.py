import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from vasijas.evaluacion import graficar_recall_por_clase_y_avance


class RecallPorClaseYAvanceTests(unittest.TestCase):
    def setUp(self):
        self.tabla = pd.DataFrame(
            {
                "Clase Ana": [1, 1, 1, 2, 4, 3, 1, 2],
                "clase_predicha": [
                    "clase01",
                    "clase01",
                    "clase04",
                    "clase01",
                    "clase04",
                    "clase03",
                    "clase02",
                    "clase02",
                ],
                "borde_origen": ["tope"] * 5 + ["base", "completo", "completo"],
                "porcentaje_avance": [10] * 5 + [20, 100, None],
            }
        )

    def tearDown(self):
        plt.close("all")

    @patch("vasijas.evaluacion.plt.show")
    def test_recall_macro_no_ponderado_exclusion_y_grupos_ausentes(self, _mostrar):
        original = self.tabla.copy(deep=True)
        matriz = graficar_recall_por_clase_y_avance(self.tabla, "tope")
        self.assertAlmostEqual(matriz.loc["Clase 1", 10], 200 / 3)
        self.assertEqual(matriz.loc["Clase 2", 10], 0)
        self.assertAlmostEqual(matriz.loc["Media (macro)", 10], 100 / 3)
        self.assertNotIn("Clase 4", matriz.index)
        self.assertTrue(pd.isna(matriz.loc["Clase 3", 10]))
        self.assertEqual(plt.gcf().axes[0].images[0].get_clim(), (0, 100))
        pd.testing.assert_frame_equal(self.tabla, original)

    @patch("vasijas.evaluacion.plt.show")
    def test_seleccion_base_y_completos_con_clases_numericas(self, _mostrar):
        tabla = self.tabla.copy()
        tabla["clase_predicha"] = (
            tabla["clase_predicha"].str.removeprefix("clase").astype(int)
        )
        base = graficar_recall_por_clase_y_avance(tabla, "base")
        self.assertEqual(list(base.columns), [20])
        self.assertEqual(base.loc["Clase 3", 20], 100)
        completos = graficar_recall_por_clase_y_avance(tabla, "completo")
        self.assertEqual(list(completos.columns), [100])
        self.assertEqual(completos.loc["Media (macro)", 100], 50)

    def test_rechaza_origen_invalido_o_sin_ejemplos(self):
        with self.assertRaisesRegex(ValueError, "borde_origen"):
            graficar_recall_por_clase_y_avance(self.tabla, "otro")
        with self.assertRaisesRegex(ValueError, "No hay ejemplos"):
            graficar_recall_por_clase_y_avance(self.tabla.iloc[:0], "tope")


if __name__ == "__main__":
    unittest.main()
