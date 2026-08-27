from pathlib import Path
import papermill as pm

notebook_entrada = Path(
    "notebooks/03.entrenamiento/1.Entrenamiento_resnet18.ipynb"
)

directorio_salida = Path("output/experiments")

directorio_salida.mkdir(
    parents=True,
    exist_ok=True
)

porcentajes = [10, 20, 30, 40, 50]


for porcentaje in porcentajes:

    notebook_salida = (
            directorio_salida
            / f"resnet18_tope_{porcentaje}.ipynb"
    )

    pm.execute_notebook(
        notebook_entrada,
        notebook_salida,
        parameters={
            "tipo_fragmento": "tope",
            "porcentaje_fragmento": porcentaje,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "seed": 42,
        },
    )