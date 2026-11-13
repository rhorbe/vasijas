# Vasijas

Proyecto de tesis para la clasificación de perfiles de vasijas mediante
aprendizaje profundo. Incluye exploración de datos, preparación de imágenes,
generación de perfiles parciales, entrenamiento por transferencia y evaluación.

El trabajo se organiza en notebooks Jupyter, con funciones reutilizables en
el paquete Python `vasijas`.

## Reportes de ejecución
Ver en: https://rhorbe.github.io/vasijas/

## Funcionalidades

- Exploración de imágenes y metadatos.
- Normalización de nombres y asociación de perfiles con su ubicación.
- Organización de imágenes según la etiqueta `Clase Ana`.
- Conversión a lienzos cuadrados, generación de mitades y fracciones,
  y centrado de los recortes.
- Preparación de datasets de perfiles completos y combinados con fracciones.
- Entrenamiento con ResNet18 preentrenada en ImageNet.
- Evaluación mediante métricas de clasificación, matrices de confusión
  y análisis por origen y porcentaje de perfil conservado.
- Visualización UMAP de las características extraídas por la red.
- Ejecución de experimentos parametrizados con Papermill.
- Generación de reportes HTML con Quarto.

## Requisitos e instalación

- Python 3.11 o superior.
- `uv` para gestionar el entorno y las dependencias.
- Quarto, instalado por separado, para generar los reportes.
- GPU compatible con CUDA opcional. El código selecciona CUDA si está
  disponible y utiliza CPU en caso contrario.

Desde la raíz del repositorio:

```bash
uv sync --extra dev
```

Este comando instala el proyecto y las herramientas de desarrollo:
pytest, Ruff y pre-commit.

Los modelos preentrenados pueden requerir descargar sus pesos en la
primera ejecución.

## Configuración

Copiar `.env.example` a `.env` y definir `PROJECT_ROOT` con la ruta
del proyecto. Por ejemplo, en Windows:

```dotenv
PROJECT_ROOT=C:/Users/rafae/Dev/vasijas
```

La variable es obligatoria para las funciones de configuración.
En `.env.example` las opciones están comentadas: hay que habilitar
la correspondiente o escribir una nueva.

El proyecto también contempla ejecución en Google Colab y montaje
de Google Drive.

## Datos de entrada

La carpeta `data/` está excluida del control de versiones. Los datos
deben obtenerse y colocarse por separado antes de ejecutar el flujo.

La estructura inicial esperada es:

```text
data/
└── 01.originales/
    ├── metadata.csv
    ├── ubicacion_1/
    │   └── perfil.png
    └── ubicacion_2/
        └── perfil.png
```

Entre las columnas utilizadas por la preparación se encuentran:

- `Identificador`: identificador asociado al nombre de la imagen.
- `Clase Ana`: etiqueta de clasificación.
- `Ubicacion`: incorporada durante la preparación a partir de la
  carpeta de procedencia.

Las etapas posteriores generan metadatos adicionales para conservar
la relación entre cada perfil completo y sus imágenes derivadas.

## Estructura del proyecto

```text
vasijas/
├── src/vasijas/                 # Funciones reutilizables
├── notebooks/
│   ├── 01.exploracion/          # Exploración inicial
│   ├── 02.preparacion/          # Preparación y partición de datos
│   ├── 03.entrenamiento/        # Entrenamiento de modelos
│   └── 04.evaluacion/           # Predicciones y visualizaciones UMAP
├── scripts/ejecucion.py         # Experimentos con Papermill
├── tests/                      # Pruebas automatizadas
├── data/                       # Datos originales y derivados
├── output/                     # Modelos y resultados
├── docs/                       # Sitio generado con Quarto
├── index.qmd                   # Índice de reportes
├── _quarto.yml                 # Configuración de Quarto
└── pyproject.toml              # Dependencias y configuración del paquete
```

## Flujo de trabajo

Iniciar Jupyter desde la raíz:

```bash
uv run jupyter notebook
```

Seleccionar el entorno del proyecto como kernel y ejecutar las celdas
de cada notebook en orden.

| Etapa | Notebooks | Contenido |
| --- | --- | --- |
| Exploración | 01 | Inspección de imágenes, metadatos y correspondencia entre ambos |
| Preparación | 02–05 | Normalización, ubicación, clasificación y conversión a cuadrados |
| Perfiles parciales | 06–09 | Mitades, fracciones, centrado y prueba de transformaciones |
| Datasets | 10–12 | División de perfiles completos, unión de datasets y división del unificado |
| Entrenamiento | 13 | Entrenamiento por transferencia |
| Evaluación | 14–16 | Predicciones y análisis UMAP |

Para trabajar únicamente con perfiles completos, preparar los datos
hasta el notebook 05 y generar su partición con el notebook 10.
Para el dataset unificado, completar también la generación de
fracciones y los notebooks 11 y 12.

### División de los datos

La pertenencia a `train`, `val` y `test` se guarda en `metadata.csv`.
Las imágenes mantienen su organización por clase; no se crean
carpetas independientes por partición.

En el dataset unificado, la división se realiza por
`id_perfil_completo`: cada perfil y todas sus fracciones reciben
el mismo conjunto. Esto evita compartir imágenes derivadas del mismo
perfil entre entrenamiento, validación y prueba.

Las proporciones se aplican a perfiles únicos; la cantidad relativa
de imágenes puede variar según las fracciones generadas.

El entrenamiento utiliza la partición persistida en los metadatos,
sin recalcularla.

### Entrenamiento

El notebook principal es:

```text
notebooks/03.entrenamiento/13.Entrenamiento_resnet18.ipynb
```

Permite configurar el dataset (`completos` o `unificado`), el nombre
de ejecución, las épocas, el tamaño de lote, la tasa de aprendizaje,
los aumentos de datos y la ponderación de clases.

Incluye parada anticipada y descongelación programada de `layer4`
para ResNet18. Al finalizar, restaura los pesos con mejor exactitud
de validación.

Los pesos se guardan con el patrón:

```text
output/models/model_{MODEL_NAME}_{DATASET_NAME}_{RUN_NAME}.pth
```

### Experimentos parametrizados

Revisar `PARAMETROS_BASE` y `MATRIZ_EXPERIMENTOS` en
`scripts/ejecucion.py` y ejecutar desde la raíz:

```bash
uv run python scripts/ejecucion.py
```

El script ejecuta el notebook de entrenamiento mediante Papermill
y guarda un notebook por experimento en `output/experiments/`.

Papermill utiliza el kernel `python3`. Para registrarlo con el
entorno del proyecto:

```bash
uv run python -m ipykernel install --user --name python3 --display-name "Python (vasijas)"
```

La configuración del script puede diferir de los valores del
notebook; revisar ambos antes de iniciar una ejecución.

## Resultados

| Directorio | Contenido |
| --- | --- |
| `output/models/` | Pesos de los modelos entrenados |
| `output/experiments/` | Notebooks ejecutados mediante Papermill |
| `output/predictions/` | Tablas de predicciones |
| `output/umap/` | Tablas y figuras de las proyecciones UMAP |
| `output/reports/` | Reportes de procesamiento |

Antes de evaluar, seleccionar los pesos correspondientes a la ejecución
y al dataset que se desea analizar.

## Reportes con Quarto

Quarto se instala como herramienta externa; no forma parte de las
dependencias Python.

Con el entorno del proyecto activado y el kernel `python3` configurado:

```bash
quarto render
```

El sitio se genera en `docs/`, según `_quarto.yml`.

Para previsualizarlo:

```bash
quarto preview
```

El índice reúne los notebooks de exploración, preparación,
entrenamiento, evaluación y experimentos.
