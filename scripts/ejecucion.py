from pathlib import Path

import papermill as pm

NOTEBOOK_ENTRADA = Path(
    "notebooks/03.entrenamiento/13.Entrenamiento_resnet18_completos.ipynb"
)
DIRECTORIO_SALIDA = Path("output/experiments")
DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)

PARAMETROS_BASE = {
    "DATASET_NAME": "unificado",
    "MODEL_NAME": "resnet18",
    "RUN_NAME": "default",
    "BATCH_SIZE": 16,
    "EPOCHS": 300,
    "EARLY_STOPPING_PATIENCE": 15,
    "EARLY_STOPPING_MIN_DELTA": 0.001,
    "LR": 1e-3,
    "UNFREEZE_EPOCH": None,
    "BACKBONE_LR_FACTOR": 0.1,
    "USE_CLASS_WEIGHTS": False,
    "FLIP_PROBABILITY": 0.5,
    "ROTATION_DEGREES": [-15, 15],
    "SCALING_FACTOR_INTERVAL": [0.8, 1.0],
    "IMG_SIZE": 224,
    "IMAGENET_MEAN": [0.485, 0.456, 0.406],
    "IMAGENET_STD": [0.229, 0.224, 0.225],
    "PIXEL_FILL": [255, 255, 255],
    "NUM_WORKERS": 8,
    "USE_AMP": True,
    "USE_CHANNELS_LAST": True,
}

MATRIZ_EXPERIMENTOS = [
    {
        "RUN_NAME": "prueba_experimento2",
        "BATCH_SIZE": 16,
        "EPOCHS": 10,
        "LR": 1e-3,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": False,
        "UNFREEZE_EPOCH": None,
    }
]
#
# MATRIZ_EXPERIMENTOS = [
#     {
#         "RUN_NAME": "bs16_lr1e-3_flip05_base",
#         "BATCH_SIZE": 16,
#         "LR": 1e-3,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": False,
#         "UNFREEZE_EPOCH": None,

#
#     },
#     {
#         "RUN_NAME": "bs32_lr1e-3_flip05_base",
#         "BATCH_SIZE": 32,
#         "LR": 1e-3,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": False,
#         "UNFREEZE_EPOCH": None,

#     },
#     {
#         "RUN_NAME": "bs16_lr1e-3_flip05_weighted",
#         "BATCH_SIZE": 16,
#         "LR": 1e-3,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": None,

#     },
#     {
#         "RUN_NAME": "bs32_lr1e-3_flip05_weighted",
#         "BATCH_SIZE": 32,
#         "LR": 1e-3,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": None,

#     },
#     {
#         "RUN_NAME": "bs16_lr1e-4_flip05_weighted_unfreeze",
#         "BATCH_SIZE": 16,
#         "LR": 1e-4,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": 10,
#     },
#     {
#         "RUN_NAME": "bs16_lr1e-4_flip08_weighted_unfreeze",
#         "BATCH_SIZE": 16,
#         "LR": 1e-4,
#         "FLIP_PROBABILITY": 0.8,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": 10,
#     },
#     {
#         "RUN_NAME": "bs16_lr3e-4_flip05_weighted_unfreeze",
#         "BATCH_SIZE": 16,
#         "LR": 3e-4,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": 10,
#     },
#     {
#         "RUN_NAME": "bs16_lr3e-4_flip08_weighted_unfreeze",
#         "BATCH_SIZE": 16,
#         "LR": 3e-4,
#         "FLIP_PROBABILITY": 0.8,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": 10,
#     },
#     {
#         "RUN_NAME": "bs16_lr1e-5_flip05_weighted_unfreeze",
#         "BATCH_SIZE": 16,
#         "LR": 1e-5,
#         "FLIP_PROBABILITY": 0.5,
#         "USE_CLASS_WEIGHTS": True,
#         "UNFREEZE_EPOCH": 10,
#     }
# ]


def ejecutar_experimento(num_experimento: int | None, experimento: dict) -> None:
    parametros = PARAMETROS_BASE | experimento

    run_name = parametros["RUN_NAME"]
    model_name = parametros["MODEL_NAME"]
    # Se usa para nombrar la salida; el notebook no consume este parámetro.
    dataset_name = parametros.pop("DATASET_NAME")

    notebook_salida = (
        DIRECTORIO_SALIDA
        / f"{num_experimento:02d}_exp_{model_name}_{dataset_name}_{run_name}.ipynb"
    )

    print(f"\nEjecutando: {run_name}")
    print(f"Notebook de salida: {notebook_salida}")

    pm.execute_notebook(
        input_path=NOTEBOOK_ENTRADA,
        output_path=notebook_salida,
        parameters=parametros,
        kernel_name="python3",
        progress_bar=True,
        log_output=True,
    )


def main() -> None:
    if not NOTEBOOK_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró el notebook plantilla: {NOTEBOOK_ENTRADA}"
        )

    errores = []

    for index, experimento in enumerate(MATRIZ_EXPERIMENTOS):
        num_experimento = index + 1
        try:
            ejecutar_experimento(num_experimento, experimento)
        except Exception as error:
            run_name = experimento["RUN_NAME"]
            errores.append((run_name, error))
            print(f"Falló el experimento {num_experimento} - {run_name}: {error}")

    if errores:
        nombres = ", ".join(nombre for nombre, _ in errores)
        raise RuntimeError(f"Fallaron {len(errores)} experimentos: {nombres}")

    print("\nTodos los experimentos finalizaron correctamente.")


if __name__ == "__main__":
    main()
