# Informe de configuración declarada y configuración efectiva del entrenamiento CNN

Fecha de revisión: **11 de octubre de 2026**.

## Alcance y dictamen

Se revisó el código actual de [entrenamiento_cnn.py](../../src/vasijas/entrenamiento_cnn.py), su integración con la [plantilla ResNet-18](../../notebooks/03.entrenamiento/13.Entrenamiento_resnet18.ipynb) y el [ejecutor Papermill](../../scripts/ejecucion.py). Se contrastaron los hallazgos con el [informe del 8 de octubre](../2026-10-08_entrenamiento/informe.md) y las salidas conservadas de dos pruebas recientes. No se ejecutó entrenamiento, no se cargaron pesos y no se modificó código.

**La integración actual corrige los principales desacoples anteriores:** ponderación conectada al criterio, descongelación controlada mediante `UNFREEZE_EPOCH=None`, LR del backbone calculado después de la inyección y transmisión de `NUM_WORKERS`, AMP y channels-last. Además, se restaura siempre el estado con menor pérdida de validación y se incorpora parada anticipada opcional.

**Persiste una discrepancia de identidad del dataset:** `DATASET_NAME="unificado"` solo interviene en el nombre del notebook de salida; la plantilla selecciona `PATHS.DATA_CUADRADOS_DIVIDIDO`. La configuración activa entrena sobre completos, aunque el archivo de salida diga “unificado”. También persisten limitaciones de reproducibilidad y trazabilidad del guardado.

“Efectivo” designa aquí el comportamiento deducido del flujo actual. Los logs se citan por separado como evidencia observada; no prueban ramas que esas ejecuciones no recorrieron ni fijan la versión del módulo importado entonces.

## Configuración declarada y configuración efectiva

La única entrada activa de `MATRIZ_EXPERIMENTOS` es `prueba_experimento2`. La matriz de nueve experimentos está comentada y no se ejecuta. El ejecutor combina `PARAMETROS_BASE | experimento`: los valores de la entrada prevalecen sobre los de la base y Papermill los inyecta después de la celda de parámetros.

| Aspecto | Plantilla directa | Ejecutor con matriz activa | Comportamiento efectivo actual |
|---|---|---|---|
| Modelo | `resnet18` | `resnet18` | Pesos ImageNet, parámetros originales congelados y nueva `fc` entrenable con una salida por clase. |
| Dataset | `PATHS.DATA_CUADRADOS_DIVIDIDO` | Etiqueta `unificado`; no inyecta `DATA_DIR` | Usa `data/10.cuadrados_divididos_en_train_val_y_test`, relativo a la raíz configurada. |
| Run | `default` | `prueba_experimento2` | Identifica notebook y archivo de pesos. |
| Batch | 16 | 16 | Se transmite a train, validación y test; train mezcla las muestras. |
| Épocas máximas | 30 | 10, sobrescribe las 300 de la base | El límite activo es 10; la parada puede reducirlo si se configura una paciencia alcanzable. |
| LR de cabeza | `1e-3` | `1e-3` | Adam recibe los parámetros entrenables a `1e-3`. |
| Ponderación | `False` | `False` | Entropía cruzada sin pesos. `True` sí activa el cálculo con frecuencias de train. |
| Descongelación | `None` | `None` | No añade `layer4` al optimizador. Un índice 10 la descongela en la época impresa 11. |
| Factor LR backbone | 0,1 | 0,1 | `backbone_lr = LR * BACKBONE_LR_FACTOR`, calculado en la celda de entrenamiento después de la inyección: `1e-4`, sin uso en la configuración activa. |
| Resolución | 224 × 224 | 224 × 224 | Resize bilineal en las tres particiones. |
| Flip horizontal | 0,5 | 0,5 | Aleatorio, solo en train. |
| Rotación | 15 | `[-15, 15]` | Ambos expresan un intervalo de −15° a +15°; solo train. |
| Escala | `(0.7, 1.0)` | `[0.8, 1.0]` | El ejecutor cambia el intervalo efectivo de train a 0,8–1,0. |
| Relleno | Blanco `(255,255,255)` | Blanco `[255,255,255]` | Se aplica a rotación y transformación afín. |
| Normalización | Media `[0.485,0.456,0.406]`, desvío `[0.229,0.224,0.225]` | Mismos valores | Conversión a tensor y normalización ImageNet en las tres particiones. |
| Workers | 8 | 8 | La plantilla transmite `num_workers=8` explícitamente. |
| AMP / channels-last | `True` / `True` | `True` / `True` | Se transmiten a entrenamiento y test; ambos solo se habilitan en CUDA. En CPU quedan desactivados. |
| Paciencia | 15 | 15 | Control de pérdida de validación; con máximo 10 épocas no puede acumular 15 épocas sin mejora. |
| Delta de parada | 0,001 | 0,001 | Solo controla qué mejora reinicia la paciencia; no filtra la selección de mejores pesos. |
| Estado final | Mejor pérdida de validación | Mejor pérdida de validación | Se restaura incluso con `early_stopping_patience=None`; no se selecciona por accuracy ni F1. |

La escala por defecto de `ConfiguracionTransformacionesImagen` también es 0,7–1,0. El módulo no decide por sí solo batch, número de épocas, LR ni dataset: esos valores los entrega quien lo invoca. Sus funciones de entrenamiento y evaluación tienen AMP/channels-last en `False` por defecto; la plantilla los activa explícitamente.

## Datos y partición

La metadata actual de completos contiene **1.071 registros**, distribuidos en **749 train, 161 validación y 161 test**, con diez clases: 01, 02, 03, 05, 06, 07, 08, 09, 10 y 11. Coincide en tamaños y clases con las dos pruebas recientes. Esto actualiza la observación del informe anterior sobre una metadata de completos distinta; no demuestra identidad de imágenes con las series históricas.

La ruta usada por la plantilla carga un split persistido. `obtener_indices_split_desde_metadata` exige `ruta_relativa`, `clase_carpeta` y `split`; valida nulos, splits inválidos, rutas duplicadas, correspondencia de clase, imágenes sin metadata, referencias sin imagen y particiones vacías. No recalcula proporciones ni usa una semilla de split. Las clases y sus índices proceden de `ImageFolder`, no de una lista explícita de clases seleccionadas.

Estas comprobaciones **no verifican agrupación por `id_perfil_completo`**. Para un dataset con recortes, la ausencia de fuga entre perfiles debe auditarse adicionalmente. La ruta alternativa `crear_dataloaders_imagen` genera un split estratificado por imagen, con semilla 42 y proporciones nominales 70/15/15; no agrupa por perfil y no es la ruta invocada por esta plantilla.

En la ruta con metadata, si no se entrega `num_workers`, se usa `min(8, os.cpu_count() or 1)`. `pin_memory` depende de la disponibilidad global de CUDA; `persistent_workers` se activa con workers positivos y `prefetch_factor=4`. Con `num_workers=0` se desactiva la carga en procesos. La ruta alternativa sin metadata construye loaders sin estas opciones explícitas.

## Ponderación, congelación y tasas de aprendizaje

**Ponderación corregida en código.** `USE_CLASS_WEIGHTS` llega a `crear_criterio_entropia_cruzada`. Con `True`, el peso de clase es `N_train / (num_classes * max(conteo_clase, 1))`, usando únicamente las etiquetas de los índices de train. Los pesos se colocan en el dispositivo y se pasan a `nn.CrossEntropyLoss`. La implementación presupone un `TransformSubset` sobre un dataset con `targets`. Las pruebas recientes usan `False`, por lo que no confirman mediante ejecución la rama ponderada.

**Descongelación opcional corregida.** `None` desactiva la llamada. Para ResNet-18, el índice desde cero selecciona la época de activación de `layer4`; se añade un grupo con nombre `layer4` y LR propio, evitando duplicarlo. No se descongela todo el backbone. VGG16 puede construirse, pero esta rutina de descongelación solo actúa para `model_name="resnet18"`.

**LR derivado corregido.** El cálculo ocurre después de la inyección. Para las configuraciones comentadas con LR `1e-4`, `3e-4` y `1e-5`, el backbone recibiría respectivamente `1e-5`, `3e-5` y `1e-6`, si se activaran y alcanzaran la época indicada. No hay scheduler en el flujo revisado.

**Congelar parámetros no congela todos los estados.** `congelar_parametros` cambia `requires_grad=False`, pero cada época llama a `modelo.train()`. En ResNet-18, las estadísticas móviles de BatchNorm pueden seguir actualizándose aun con el backbone sin gradientes. Por tanto, “sin descongelación” significa que no se optimizan sus parámetros originales; no implica que todos sus buffers permanezcan constantes. Esto importa al diseñar un control de backbone completamente fijo.

## Selección del estado y parada anticipada

`_MejorEstadoValidacion` conserva una copia de `state_dict` cada vez que encuentra una pérdida de validación finita estrictamente menor. Al terminar, la carga en el modelo; si nunca hubo una pérdida finita, lanza un error. La selección contempla todas las épocas, incluidas las anteriores a descongelar; puede restaurar un estado anterior a esa transición.

El control de parada usa un criterio diferente: mejora estrictamente mayor que `early_stopping_min_delta` respecto de su referencia. Con descongelación programada, no cuenta paciencia antes de esa época y reinicia la referencia al alcanzarla. Sin descongelación cuenta desde el comienzo. Si la descongelación queda fuera del máximo de épocas, la paciencia no comienza a contar durante el bucle. Los empates de pérdida no sustituyen el mejor estado.

El historial devuelve las épocas efectivamente recorridas. Su último punto puede corresponder a pesos distintos de los restaurados para test. Se imprime la época seleccionada, pero no se devuelve como campo del historial ni se guarda en el archivo de pesos. Tampoco se restaura el estado del optimizador; el flujo revisado evalúa y guarda, sin reanudar entrenamiento.

La pérdida registrada es un **promedio de pérdidas de lote**, no un promedio ponderado por cantidad de imágenes. Un último lote pequeño pesa lo mismo que uno completo. Con ponderación de clases, cada pérdida de lote también depende de la normalización del criterio. Esta es la magnitud concreta usada para elegir el estado y controlar la parada; la accuracy sí se calcula como aciertos sobre el total de muestras.

## Evidencia de las pruebas recientes

| Artefacto | Datos observados | Descongelación | Selección observada | Test Acc |
|---|---|---|---|---:|
| [prueba_experimento](../../output/experiments/01_exp_resnet18_unificado_prueba_experimento.ipynb) | CPU, completos, 749/161/161 | Índice 4; log de activación en época 5, LR backbone `1e-4` y cabeza `1e-3` | Sin mensaje de restauración en los streams revisados | 0,8696 |
| [prueba_experimento2](../../output/experiments/01_exp_resnet18_unificado_prueba_experimento2.ipynb) | CPU, completos, 749/161/161 | `None`; sin mensaje de activación | Log: restauración de época 10, `val_loss=0.8136` | 0,7329 |

Los nombres contienen “unificado”, mientras que los logs imprimen la metadata de completos. La segunda prueba confirma una restauración explícita y completa diez épocas. Como el estado elegido coincide con la última época, no demuestra por sí sola una recuperación de pesos de una época anterior. Tampoco confirma parada anticipada, ponderación o ejecución AMP en CUDA.

Estos resultados no constituyen una comparación controlada del efecto de descongelar: falta una versión fijada del módulo para cada ejecución y no hay semillas de entrenamiento controladas. Las celdas inyectadas de estos artefactos conservan opciones antiguas que ya no aparecen en la base actual del ejecutor; los notebooks ejecutados son evidencia histórica, no una copia exacta de la configuración activa de hoy.

## Reproducibilidad y guardado

No se observa fijación explícita de semillas de Python, NumPy o Torch en el flujo actual. El orden de train, la inicialización de la cabeza y la aumentación son aleatorios. En CUDA se establece `torch.backends.cudnn.benchmark=True`; no se configura ejecución determinista. La semilla 42 de la ruta alternativa de split solo controla la partición, no todo el entrenamiento.

`guardar_pesos_modelo` persiste únicamente `state_dict`, bajo `model_{MODEL_NAME}_{RUN_NAME}.pth`. La restauración hace que ese archivo contenga el estado seleccionado por validación, pero no incluye dataset, clases, configuración, split, época elegida, optimizador ni versión de código. Repetir modelo y run en la misma carpeta sobrescribe el destino; el ejecutor también reutiliza el nombre de notebook para el mismo índice/modelo/dataset/run.

## Diferencias respecto del informe anterior

| Hallazgo del 8 de octubre | Situación actual |
|---|---|
| `USE_CLASS_WEIGHTS` no se consultaba | Conectado desde la plantilla y aplicado por el criterio cuando es `True`. |
| Descongelación aun con flag desactivado | Control único mediante `UNFREEZE_EPOCH`; `None` la desactiva. |
| LR backbone calculado antes de Papermill | Calculado en la celda de entrenamiento después de la inyección. |
| Workers y opciones CUDA no transmitidos | Transmitidos explícitamente; opciones CUDA condicionadas al dispositivo. |
| Se evaluaba y guardaba la última época | Restauración obligatoria del mínimo de pérdida de validación y parada opcional. |
| Dataset nominal sin seleccionar las entradas | Persiste: el ejecutor usa `DATASET_NAME` para nombrar la salida. |
| Semillas y archivo de pesos sin trazabilidad completa | Persiste. |

Las correcciones actuales no cambian lo que ocurrió en las 27 ejecuciones históricas ni validan retrospectivamente sus etiquetas “weighted” o “unfreeze”.

## Acciones propuestas

1. Alinear la identidad del experimento con los datos: inyectar `DATA_DIR` y registrar su ruta efectiva, o etiquetar estas ejecuciones como completos.
2. Guardar configuración efectiva, clases, hash de metadata, versión de código y época seleccionada junto con los pesos; usar un identificador de ejecución que evite sobrescrituras.
3. Fijar y registrar semillas de entrenamiento; documentar las opciones de determinismo y el dispositivo efectivo.
4. Decidir si el control congelado debe fijar también BatchNorm y, para futuros recortes, verificar separación por perfil entre particiones.
5. Verificar en una ejecución controlada las ramas de ponderación, descongelación, parada y restauración de una época anterior antes de repetir la matriz completa.

## Trazabilidad de la revisión

Las celdas relevantes son `50c249aae28f1666` (parámetros), `80938cae994634fd` (datos y loaders), `ac5cff20ce78fde` (criterio y optimizador), `ec18bc38abd0749a` (LR derivado y entrenamiento), `87a06fdb2ea497ab` (test) y `8210ac633d5231cb` (guardado).

SHA-256 de las fuentes leídas, calculados durante esta revisión:

| Fuente | SHA-256 |
|---|---|
| `src/vasijas/entrenamiento_cnn.py` | `D8396C0DA8210B1D11B15418D479F1744C29D8520334FC48811179D080016381` |
| `scripts/ejecucion.py` | `5D04CC98B70AB039BBBC0657A37B1DE9DECAAFC1653E5BDBD5A715815EAA4166` |
| Plantilla `13.Entrenamiento_resnet18.ipynb` | `341425C12A81E3DDE2F1981F509A123ED19B3027FBD2B7CD7988550628292442` |

También se revisaron [split_dataset.py](../../src/vasijas/split_dataset.py), [paths.py](../../src/vasijas/paths.py) y la [metadata actual de completos](../../data/10.cuadrados_divididos_en_train_val_y_test/metadata.csv). La verificación fue documental y estática; no certifica nuevos resultados de entrenamiento.
