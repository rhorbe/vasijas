from pathlib import Path

import papermill as pm


NOTEBOOK_ENTRADA = Path("notebooks/03.entrenamiento/13.Entrenamiento_resnet18.ipynb")
DIRECTORIO_SALIDA = Path("output/experiments")

DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)


PARAMETROS_BASE = {
    "DATASET_NAME": "cuadrados",
    "MODEL_NAME": "resnet18",
    "BATCH_SIZE": 16,
    "EPOCHS": 30,
    "LR": 1e-3,
    "ENABLE_UNFREEZE": False,
    "UNFREEZE_EPOCH": 10,
    "BACKBONE_LR_FACTOR": 0.1,
    "USE_CLASS_WEIGHTS": False,
    "ENABLE_HORIZONTAL_FLIP": True,
    "FLIP_PROBABILITY": 0.5,
    "ENABLE_ROTATION": True,
    "ROTATION_DEGREES": [-15, 15],
    "ENABLE_SCALING": True,
    "SCALING_FACTOR_INTERVAL": [0.8, 1.0],
    "IMG_SIZE": 224,
    "PIXEL_FILL": [255, 255, 255],
    "SEED": 42,
    "TEST_VAL_FRACTION": 0.30,
    "TEST_FRACTION_OF_TEMP": 0.50,
    "NUM_WORKERS": 8,
    "USE_AMP": True,
    "USE_CHANNELS_LAST": True,
    "ENABLE_MODEL_COMPILE": False,
}


MATRIZ_EXPERIMENTOS = [
    {
        "RUN_NAME": "lr1e3_bs16_base",
        "BATCH_SIZE": 16,
        "EPOCHS": 1,
        "LR": 1e-3,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": False,
        "ENABLE_UNFREEZE": False,
    },
    {
        "RUN_NAME": "lr1e3_bs16_base",
        "BATCH_SIZE": 16,
        "EPOCHS": 30,
        "LR": 1e-3,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": False,
        "ENABLE_UNFREEZE": False,
    },
    {
        "RUN_NAME": "lr1e3_bs32_base",
        "BATCH_SIZE": 32,
        "EPOCHS": 30,
        "LR": 1e-3,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": False,
        "ENABLE_UNFREEZE": False,
    },
    {
        "RUN_NAME": "lr1e3_bs16_flip05_weighted",
        "BATCH_SIZE": 16,
        "EPOCHS": 30,
        "LR": 1e-3,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": False,
    },
    {
        "RUN_NAME": "lr1e3_bs32_flip05_weighted",
        "BATCH_SIZE": 32,
        "EPOCHS": 30,
        "LR": 1e-3,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": False,
    },
    {
        "RUN_NAME": "lr1e4_flip05_unfreeze",
        "BATCH_SIZE": 16,
        "EPOCHS": 300,
        "LR": 1e-4,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": True,
        "UNFREEZE_EPOCH": 10,
    },
    {
        "RUN_NAME": "lr1e4_flip08_unfreeze",
        "BATCH_SIZE": 16,
        "EPOCHS": 300,
        "LR": 1e-4,
        "FLIP_PROBABILITY": 0.8,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": True,
        "UNFREEZE_EPOCH": 10,
    },
    {
        "RUN_NAME": "lr3e4_flip05",
        "BATCH_SIZE": 16,
        "EPOCHS": 300,
        "LR": 3e-4,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": True,
        "UNFREEZE_EPOCH": 10,
    },
    {
        "RUN_NAME": "lr3e4_flip08",
        "BATCH_SIZE": 16,
        "EPOCHS": 300,
        "LR": 3e-4,
        "FLIP_PROBABILITY": 0.8,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": True,
        "UNFREEZE_EPOCH": 10,
    },
    {
        "RUN_NAME": "lr1e5_flip05",
        "BATCH_SIZE": 16,
        "EPOCHS": 300,
        "LR": 1e-5,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": True,
        "UNFREEZE_EPOCH": 10,
    },
    {
        "RUN_NAME": "dataset_completo",
        "DATASET_NAME": "unificado",
        "BATCH_SIZE": 16,
        "EPOCHS": 75,
        "LR": 1e-4,
        "FLIP_PROBABILITY": 0.5,
        "USE_CLASS_WEIGHTS": True,
        "ENABLE_UNFREEZE": True,
        "UNFREEZE_EPOCH": 10,
    },
]


def ejecutar_experimento(experimento: dict) -> None:
    parametros = PARAMETROS_BASE | experimento

    run_name = parametros["RUN_NAME"]
    model_name = parametros["MODEL_NAME"]

    notebook_salida = DIRECTORIO_SALIDA / f"experiment_{model_name}_{run_name}.ipynb"

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

    for experimento in MATRIZ_EXPERIMENTOS:
        try:
            ejecutar_experimento(experimento)
        except Exception as error:
            run_name = experimento["RUN_NAME"]
            errores.append((run_name, error))
            print(f"Falló el experimento {run_name}: {error}")

    if errores:
        nombres = ", ".join(nombre for nombre, _ in errores)
        raise RuntimeError(f"Fallaron {len(errores)} experimentos: {nombres}")

    print("\nTodos los experimentos finalizaron correctamente.")


if __name__ == "__main__":
    main()
