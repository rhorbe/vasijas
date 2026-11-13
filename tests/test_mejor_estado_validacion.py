import unittest
from unittest.mock import patch

import torch

from vasijas.entrenamiento_cnn import MetricasEpoch, entrenar_modelo


class MejorEstadoValidacionTests(unittest.TestCase):
    def ejecutar_entrenamiento(self, losses, paciencia, min_delta=0.001):
        modelo = torch.nn.Linear(1, 1, bias=False)
        optimizador = torch.optim.Adam(modelo.parameters())

        def entrenar_epoch(*args):
            epoch = args[6]
            with torch.no_grad():
                modelo.weight.fill_(epoch)
            return MetricasEpoch(epoch, 0.0, 0.0, losses[epoch - 1], 0.0, 0)

        with patch(
            "vasijas.entrenamiento_cnn._entrenar_y_validar_epoch",
            side_effect=entrenar_epoch,
        ):
            historial = entrenar_modelo(
                modelo=modelo,
                train_loader=None,
                val_loader=None,
                criterio=torch.nn.CrossEntropyLoss(),
                optimizador=optimizador,
                dispositivo=torch.device("cpu"),
                epochs=len(losses),
                early_stopping_patience=paciencia,
                early_stopping_min_delta=min_delta,
            )
        return modelo.weight.item(), len(historial.metricas)

    def test_sin_parada_completa_epochs_y_restaura_mejor(self):
        peso, epochs = self.ejecutar_entrenamiento([0.8, 0.4, 0.6, 0.7], None)
        self.assertEqual(epochs, 4)
        self.assertEqual(peso, 2.0)

    def test_con_parada_restaura_mejor(self):
        peso, epochs = self.ejecutar_entrenamiento([0.8, 0.4, 0.6, 0.7, 0.9], 2)
        self.assertEqual(epochs, 4)
        self.assertEqual(peso, 2.0)

    def test_min_delta_no_impide_guardar_reduccion_pequena(self):
        peso, epochs = self.ejecutar_entrenamiento([0.8, 0.799, 0.9], 1, 0.01)
        self.assertEqual(epochs, 2)
        self.assertEqual(peso, 2.0)

    def test_sin_loss_finita_falla_en_lugar_de_devolver_pesos_finales(self):
        with self.assertRaisesRegex(ValueError, "pérdida de validación finita"):
            self.ejecutar_entrenamiento([float("nan"), float("inf")], None)


if __name__ == "__main__":
    unittest.main()
