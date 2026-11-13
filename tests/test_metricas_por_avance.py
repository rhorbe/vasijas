import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from vasijas.evaluacion import (
    graficar_exactitud_por_avance_y_origen,
    graficar_metrica_por_avance_y_origen,
)


class MetricasPorAvanceTests(unittest.TestCase):
    def setUp(self):
        self.tabla = pd.DataFrame(
            {
                "porcentaje_avance": [10] * 6 + [90] + [100] * 3,
                "borde_origen": ["tope"] * 4 + ["base"] * 2 + ["completo"] * 4,
                "Clase Ana": [1, 1, 1, 2, 1, 1, 1, 2, 2, 2],
                "clase_predicha": [1, 1, 1, 1, 1, 1, 2, 2, 1, 1],
            }
        )
        self.tabla["acierto"] = self.tabla["Clase Ana"].eq(self.tabla["clase_predicha"])

    def tearDown(self):
        plt.close("all")

    @patch("vasijas.evaluacion.plt.show")
    def test_f1_macro_pesa_clases_igualmente_y_recalcula_referencia(self, _mostrar):
        for tipo in ("lineas", "barras"):
            for etiquetas_texto in (False, True):
                with self.subTest(tipo=tipo, etiquetas_texto=etiquetas_texto):
                    tabla = self.tabla.copy(deep=True)
                    if etiquetas_texto:
                        tabla["clase_predicha"] = tabla["clase_predicha"].map(
                            lambda clase: f"clase{clase:02d}"
                        )
                    original = tabla.copy(deep=True)
                    resumen = graficar_metrica_por_avance_y_origen(
                        tabla, metrica="f1_macro", tipo_grafico=tipo
                    ).set_index(["porcentaje_avance", "borde_origen"])
                    # Tres aciertos sobre cuatro: accuracy=75 %, pero F1 macro=3/7.
                    self.assertAlmostEqual(
                        resumen.loc[(10, "tope"), "f1_macro_porcentual"], 100 * 3 / 7
                    )
                    # La clase 2, ausente en este grupo, sigue contando en el macro.
                    self.assertEqual(
                        resumen.loc[(10, "base"), "f1_macro_porcentual"], 50
                    )
                    # F1 conjunto=20 %; promediar los dos grupos daría otro valor.
                    referencia = plt.gcf().axes[0].lines[-1]
                    self.assertEqual(list(referencia.get_ydata()), [20, 20])
                    self.assertEqual(referencia.get_color(), "green")
                    self.assertEqual(referencia.get_linestyle(), "-")
                    pd.testing.assert_frame_equal(tabla, original)
                    plt.close("all")

    @patch("vasijas.evaluacion.plt.show")
    def test_exactitud_conserva_api_y_no_necesita_etiquetas_de_clase(self, _mostrar):
        tabla = self.tabla.drop(columns=["Clase Ana", "clase_predicha"])
        resumen = graficar_exactitud_por_avance_y_origen(tabla)
        self.assertEqual(
            list(resumen.columns),
            [
                "porcentaje_avance",
                "borde_origen",
                "aciertos",
                "imagenes",
                "exactitud_porcentual",
            ],
        )
        tope = resumen.loc[resumen["borde_origen"].eq("tope")].iloc[0]
        self.assertEqual(tope["exactitud_porcentual"], 75)
        self.assertEqual(list(plt.gcf().axes[0].lines[-1].get_ydata()), [25, 25])


if __name__ == "__main__":
    unittest.main()
