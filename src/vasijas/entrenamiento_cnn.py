from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from torchvision import transforms as tfs
from torchvision.models import ResNet18_Weights, VGG16_Weights

from vasijas.split_dataset import (
    ConfiguracionSplitDataset,
    generar_indices_split_estratificado,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# TODO 1.1
@dataclass(frozen=True)
class ConfiguracionTransformacionesImagen:
    tamano_imagen: int = 224
    probabilidad_flip: float = 0.5
    intervalo_escalado: tuple[float, float] = (0.7, 1.0)
    grados_rotacion: int = 15
    relleno_pixeles: tuple[int, int, int] = (255, 255, 255)
    media_normalizacion: Sequence[float] = field(default_factory=lambda: IMAGENET_MEAN)
    desvio_normalizacion: Sequence[float] = field(default_factory=lambda: IMAGENET_STD)


@dataclass(frozen=True)
class DatasetsImagen:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list[str]


@dataclass(frozen=True)
class DataLoadersImagen:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class ResultadoEvaluacion:
    loss: float
    accuracy: float
    predicciones: list[int]
    etiquetas: list[int]


@dataclass(frozen=True)
class MetricasEpoch:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    duracion: Any


@dataclass(frozen=True)
class HistorialEntrenamiento:
    metricas: list[MetricasEpoch]

    @property
    def train_losses(self) -> list[float]:
        return [metrica.train_loss for metrica in self.metricas]

    @property
    def val_losses(self) -> list[float]:
        return [metrica.val_loss for metrica in self.metricas]

    @property
    def train_accuracies(self) -> list[float]:
        return [metrica.train_accuracy for metrica in self.metricas]

    @property
    def val_accuracies(self) -> list[float]:
        return [metrica.val_accuracy for metrica in self.metricas]


@dataclass(frozen=True)
class ResultadoClasificacion:
    reporte: str
    matriz_confusion: np.ndarray


class TransformSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform: Callable | None = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, indice: int) -> tuple[Any, int]:
        imagen, etiqueta = self.dataset[self.indices[indice]]
        if self.transform is not None:
            imagen = self.transform(imagen)
        return imagen, etiqueta


def seleccionar_dispositivo_procesamiento() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describir_dispositivo(dispositivo: torch.device) -> str:
    if dispositivo.type == "cuda":
        return f"Dispositivo utilizado para procesamiento: CUDA - {torch.cuda.get_device_name(0)}"
    return "Dispositivo utilizado para procesamiento: CPU"


def describir_metricas_epoch(
    metrica: MetricasEpoch,
    total_epochs: int,
) -> str:
    """Construye el texto mostrado al finalizar una época de entrenamiento."""
    return "\n".join(
        [
            f"Epoch {metrica.epoch}/{total_epochs}",
            (
                f"Train Loss: {metrica.train_loss:.4f} | "
                f"Train Acc: {metrica.train_accuracy:.4f}"
            ),
            f"Val Loss: {metrica.val_loss:.4f} | Val Acc: {metrica.val_accuracy:.4f}",
            f"Tiempo de epoch: {metrica.duracion}",
        ]
    )


def graficar_lote_imagenes(
    loader: DataLoader,
    nombres_clases: Sequence[str],
    cantidad: int = 8,
    *,
    media_normalizacion: Sequence[float] = IMAGENET_MEAN,
    desvio_normalizacion: Sequence[float] = IMAGENET_STD,
) -> Figure:
    """Grafica un lote deshaciendo la normalización solo para visualizar.

    La media y el desvío deben coincidir con los usados por el loader.
    """
    if cantidad <= 0:
        raise ValueError("cantidad debe ser mayor a cero")

    imagenes, etiquetas = next(iter(loader))
    cantidad_mostrada = min(cantidad, len(imagenes))
    figura, ejes = plt.subplots(
        1, cantidad_mostrada, figsize=(2 * cantidad_mostrada, 3)
    )
    ejes_array = np.atleast_1d(ejes)

    for imagen, etiqueta, eje in zip(
        imagenes[:cantidad_mostrada],
        etiquetas[:cantidad_mostrada],
        ejes_array,
        strict=True,
    ):
        imagen_array = imagen.detach().cpu().permute(1, 2, 0).numpy()
        imagen_array = imagen_array * np.asarray(desvio_normalizacion) + np.asarray(
            media_normalizacion
        )
        imagen_array = np.clip(imagen_array, 0, 1)
        eje.imshow(imagen_array.squeeze(), cmap="gray")
        eje.set_title(nombres_clases[int(etiqueta)])
        eje.axis("off")

    figura.tight_layout()
    return figura


def crear_transformaciones_imagen(
    configuracion: ConfiguracionTransformacionesImagen,
) -> tuple[tfs.Compose, tfs.Compose, tfs.Compose]:
    """Crea las transformaciones para entrenamiento, validación y prueba.

    La transformación de entrenamiento aplica redimensionado, aumento de datos
    mediante volteo horizontal, rotación y escalado aleatorios, conversión a
    tensor y normalización.

    Args:
        configuracion: Parámetros de redimensionado, aumento de datos y
            normalización que se aplicarán a las imágenes.

    Returns:
        Una tupla con las transformaciones de entrenamiento, validación y
        prueba, respectivamente.
    """
    transformacion_validacion = tfs.Compose(
        [
            tfs.Resize(
                (configuracion.tamano_imagen, configuracion.tamano_imagen),
                interpolation=tfs.InterpolationMode.BILINEAR,
            ),
            tfs.ToTensor(),
            tfs.Normalize(
                mean=configuracion.media_normalizacion,
                std=configuracion.desvio_normalizacion,
            ),
        ]
    )

    transformacion_entrenamiento = tfs.Compose(
        [
            tfs.Resize(
                (configuracion.tamano_imagen, configuracion.tamano_imagen),
                interpolation=tfs.InterpolationMode.BILINEAR,
            ),
            tfs.RandomHorizontalFlip(p=configuracion.probabilidad_flip),
            tfs.RandomRotation(
                degrees=configuracion.grados_rotacion,
                fill=configuracion.relleno_pixeles,
            ),
            tfs.RandomAffine(
                degrees=0,
                scale=configuracion.intervalo_escalado,
                fill=configuracion.relleno_pixeles,
            ),
            tfs.ToTensor(),
            tfs.Normalize(
                mean=configuracion.media_normalizacion,
                std=configuracion.desvio_normalizacion,
            ),
        ]
    )

    return (
        transformacion_entrenamiento,
        transformacion_validacion,
        transformacion_validacion,
    )


def crear_datasets_imagen(
    data_dir: Path,
    transformacion_entrenamiento: Callable,
    transformacion_validacion: Callable,
    transformacion_test: Callable,
    configuracion_split: ConfiguracionSplitDataset,
) -> DatasetsImagen:
    """Crea los datasets de entrenamiento, validación y prueba.

    Carga las imágenes desde un directorio compatible con
    :class:`torchvision.datasets.ImageFolder`, genera una división
    estratificada y asigna a cada subconjunto su transformación correspondiente.
    Los tres subconjuntos comparten el mismo dataset base, pero contienen índices
    diferentes.

    Args:
        data_dir: Directorio raíz con una subcarpeta por clase.
        transformacion_entrenamiento: Transformación aplicada a las imágenes del
            subconjunto de entrenamiento.
        transformacion_validacion: Transformación aplicada a las imágenes del
            subconjunto de validación.
        transformacion_test: Transformación aplicada a las imágenes del
            subconjunto de prueba.
        configuracion_split: Parámetros usados para generar la división
            estratificada del dataset.

    Returns:
        Los datasets de entrenamiento, validación y prueba junto con los nombres
        de las clases, en el orden asignado por ``ImageFolder``.
    """
    dataset_completo = datasets.ImageFolder(data_dir)
    indices_split = generar_indices_split_estratificado(
        etiquetas=dataset_completo.targets,
        configuracion_split=configuracion_split,
    )

    return DatasetsImagen(
        train=TransformSubset(
            dataset_completo,
            indices_split.train,
            transformacion_entrenamiento,
        ),
        val=TransformSubset(
            dataset_completo,
            indices_split.val,
            transformacion_validacion,
        ),
        test=TransformSubset(
            dataset_completo,
            indices_split.test,
            transformacion_test,
        ),
        class_names=dataset_completo.classes,
    )


def obtener_indices_split_desde_metadata(
    dataset: datasets.ImageFolder,
    data_dir: Path,
    metadata_split: pd.DataFrame,
) -> dict[str, list[int]]:
    """Relaciona las muestras de ``ImageFolder`` con un split persistido.

    La relación se realiza mediante ``ruta_relativa`` para no depender del
    orden en que ``ImageFolder`` descubre los archivos.
    """
    columnas_requeridas = {"ruta_relativa", "clase_carpeta", "split"}
    columnas_faltantes = columnas_requeridas - set(metadata_split.columns)
    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en la metadata: {sorted(columnas_faltantes)}"
        )

    columnas_con_nulos = [
        columna
        for columna in columnas_requeridas
        if metadata_split[columna].isna().any()
    ]
    if columnas_con_nulos:
        raise ValueError(
            f"La metadata contiene valores nulos en: {sorted(columnas_con_nulos)}"
        )

    splits_validos = {"train", "val", "test"}
    splits_encontrados = set(metadata_split["split"].dropna().astype(str))
    splits_invalidos = splits_encontrados - splits_validos
    if splits_invalidos:
        raise ValueError(
            f"La metadata contiene splits inválidos: {sorted(splits_invalidos)}"
        )

    rutas_metadata = (
        metadata_split["ruta_relativa"]
        .astype(str)
        .map(lambda ruta: Path(ruta).as_posix())
    )
    rutas_duplicadas = rutas_metadata[rutas_metadata.duplicated()].unique().tolist()
    if rutas_duplicadas:
        raise ValueError(
            f"La metadata contiene rutas duplicadas: {rutas_duplicadas[:10]}"
        )

    metadata_por_ruta = metadata_split.assign(
        ruta_relativa_normalizada=rutas_metadata
    ).set_index("ruta_relativa_normalizada")
    data_dir_resuelto = data_dir.resolve()
    indices_por_split: dict[str, list[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    rutas_dataset: set[str] = set()

    for indice, (ruta_imagen, etiqueta) in enumerate(dataset.samples):
        ruta_relativa = (
            Path(ruta_imagen).resolve().relative_to(data_dir_resuelto).as_posix()
        )
        rutas_dataset.add(ruta_relativa)
        if ruta_relativa not in metadata_por_ruta.index:
            raise ValueError(
                f"La imagen {ruta_relativa!r} no tiene un split asignado en la metadata"
            )

        registro = metadata_por_ruta.loc[ruta_relativa]
        clase_dataset = dataset.classes[etiqueta]
        if str(registro["clase_carpeta"]) != clase_dataset:
            raise ValueError(
                f"La clase de {ruta_relativa!r} no coincide: "
                f"metadata={registro['clase_carpeta']!r}, dataset={clase_dataset!r}"
            )
        indices_por_split[str(registro["split"])].append(indice)

    rutas_sin_imagen = sorted(set(metadata_por_ruta.index) - rutas_dataset)
    if rutas_sin_imagen:
        raise ValueError(
            f"La metadata referencia imágenes inexistentes: {rutas_sin_imagen[:10]}"
        )

    splits_vacios = [
        split for split, indices in indices_por_split.items() if not indices
    ]
    if splits_vacios:
        raise ValueError(f"La metadata contiene splits vacíos: {splits_vacios}")

    return indices_por_split


def crear_datasets_imagen_desde_metadata(
    data_dir: Path,
    metadata_split: pd.DataFrame,
    transformacion_entrenamiento: Callable,
    transformacion_validacion: Callable,
    transformacion_test: Callable,
) -> DatasetsImagen:
    """Crea datasets usando la división persistida en la metadata."""
    dataset_completo = datasets.ImageFolder(data_dir)
    indices = obtener_indices_split_desde_metadata(
        dataset=dataset_completo,
        data_dir=data_dir,
        metadata_split=metadata_split,
    )
    return DatasetsImagen(
        train=TransformSubset(
            dataset_completo, indices["train"], transformacion_entrenamiento
        ),
        val=TransformSubset(
            dataset_completo, indices["val"], transformacion_validacion
        ),
        test=TransformSubset(dataset_completo, indices["test"], transformacion_test),
        class_names=dataset_completo.classes,
    )


# TODO 3.1
def crear_dataloaders_imagen(
    data_dir: Path,
    batch_size: int,
    configuracion_transformaciones: ConfiguracionTransformacionesImagen,
    configuracion_split: ConfiguracionSplitDataset,
) -> DataLoadersImagen:
    """Crea los ``DataLoader`` para entrenamiento, validación y prueba.

    Las imágenes se cargan desde un directorio compatible con
    :class:`torchvision.datasets.ImageFolder`, se dividen de forma estratificada
    según ``configuracion_split`` y reciben las transformaciones definidas en
    ``configuracion_transformaciones``. El conjunto de entrenamiento se mezcla
    en cada época; los conjuntos de validación y prueba conservan su orden.

    Args:
        data_dir: Directorio raíz con una subcarpeta por clase.
        batch_size: Cantidad de imágenes incluidas en cada lote.
        configuracion_transformaciones: Parámetros de aumento, redimensionado y
            normalización de las imágenes.
        configuracion_split: Parámetros usados para dividir el dataset en los
            subconjuntos de entrenamiento, validación y prueba.

    Returns:
        Los tres ``DataLoader`` junto con los nombres de las clases, en el orden
        asignado por ``ImageFolder``.
    """
    # TODO 3.2
    train_transform, val_transform, test_transform = crear_transformaciones_imagen(
        configuracion_transformaciones
    )

    # TODO 3.3
    datasets_imagen = crear_datasets_imagen(
        data_dir=data_dir,
        transformacion_entrenamiento=train_transform,
        transformacion_validacion=val_transform,
        transformacion_test=test_transform,
        configuracion_split=configuracion_split,
    )

    return DataLoadersImagen(
        train=DataLoader(datasets_imagen.train, batch_size=batch_size, shuffle=True),
        val=DataLoader(datasets_imagen.val, batch_size=batch_size, shuffle=False),
        test=DataLoader(datasets_imagen.test, batch_size=batch_size, shuffle=False),
        class_names=datasets_imagen.class_names,
    )


def crear_dataloaders_imagen_desde_metadata(
    data_dir: Path,
    metadata_split: pd.DataFrame,
    batch_size: int,
    configuracion_transformaciones: ConfiguracionTransformacionesImagen,
) -> DataLoadersImagen:
    """Crea dataloaders respetando una división persistida en metadata.

    Las transformaciones aleatorias se aplican solamente al subconjunto
    ``train``. La pertenencia de cada imagen a ``train``, ``val`` o ``test`` se
    toma de la columna ``split`` y no se vuelve a calcular.
    """
    train_transform, val_transform, test_transform = crear_transformaciones_imagen(
        configuracion_transformaciones
    )
    datasets_imagen = crear_datasets_imagen_desde_metadata(
        data_dir=data_dir,
        metadata_split=metadata_split,
        transformacion_entrenamiento=train_transform,
        transformacion_validacion=val_transform,
        transformacion_test=test_transform,
    )
    return DataLoadersImagen(
        train=DataLoader(datasets_imagen.train, batch_size=batch_size, shuffle=True),
        val=DataLoader(datasets_imagen.val, batch_size=batch_size, shuffle=False),
        test=DataLoader(datasets_imagen.test, batch_size=batch_size, shuffle=False),
        class_names=datasets_imagen.class_names,
    )


def crear_modelo_transfer_learning(
    nombre_modelo: str,
    num_classes: int,
    dispositivo: torch.device,
) -> nn.Module:
    """Crea un modelo preentrenado adaptado para clasificación de imágenes.

    Construye una arquitectura ``resnet18`` o ``vgg16`` con pesos preentrenados
    en ImageNet, congela sus parámetros originales y reemplaza la última capa
    clasificadora por una capa entrenable con una salida por clase. Finalmente,
    mueve el modelo al dispositivo indicado.

    Args:
        nombre_modelo: Nombre de la arquitectura; debe ser ``"resnet18"`` o
            ``"vgg16"``.
        num_classes: Cantidad de clases que deberá predecir el modelo.
        dispositivo: Dispositivo al que se moverá el modelo creado.

    Returns:
        El modelo adaptado para la cantidad de clases solicitada.

    Raises:
        ValueError: Si ``nombre_modelo`` no corresponde a una arquitectura
            soportada.
    """
    if nombre_modelo == "resnet18":
        modelo = models.resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
        congelar_parametros(modelo)
        modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
        return modelo.to(dispositivo)

    if nombre_modelo == "vgg16":
        modelo = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        congelar_parametros(modelo)
        modelo.classifier[6] = nn.Linear(modelo.classifier[6].in_features, num_classes)
        return modelo.to(dispositivo)

    raise ValueError(f"Modelo no soportado: {nombre_modelo}")


def congelar_parametros(modelo: nn.Module) -> None:
    for parametro in modelo.parameters():
        parametro.requires_grad = False


def crear_optimizador_adam(
    modelo: nn.Module, learning_rate: float
) -> torch.optim.Optimizer:
    parametros_entrenables = (
        parametro for parametro in modelo.parameters() if parametro.requires_grad
    )
    return torch.optim.Adam(parametros_entrenables, lr=learning_rate)


def crear_criterio_entropia_cruzada() -> nn.CrossEntropyLoss:
    """Crea el criterio de clasificación multiclase usado por los modelos."""
    return nn.CrossEntropyLoss()


def entrenar_una_epoch(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    optimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
) -> tuple[float, float]:
    modelo.train()
    total_loss = 0.0
    correctas = 0

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        optimizador.zero_grad()
        outputs = modelo(imagenes)
        loss = criterio(outputs, etiquetas)
        loss.backward()
        optimizador.step()

        total_loss += loss.item()
        _, predicciones = torch.max(outputs, 1)
        correctas += (predicciones == etiquetas).sum().item()

    accuracy = correctas / len(loader.dataset)
    return total_loss / len(loader), accuracy


def validar_modelo(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
) -> ResultadoEvaluacion:
    modelo.eval()
    total_loss = 0.0
    correctas = 0
    predicciones_totales: list[int] = []
    etiquetas_totales: list[int] = []

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            outputs = modelo(imagenes)
            loss = criterio(outputs, etiquetas)

            total_loss += loss.item()
            _, predicciones = torch.max(outputs, 1)
            correctas += (predicciones == etiquetas).sum().item()

            predicciones_totales.extend(predicciones.cpu().numpy().tolist())
            etiquetas_totales.extend(etiquetas.cpu().numpy().tolist())

    accuracy = correctas / len(loader.dataset)
    return ResultadoEvaluacion(
        loss=total_loss / len(loader),
        accuracy=accuracy,
        predicciones=predicciones_totales,
        etiquetas=etiquetas_totales,
    )


def entrenar_modelo(
    modelo: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterio: nn.Module,
    optimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
    epochs: int,
    al_finalizar_epoch: Callable[[MetricasEpoch], None] | None = None,
) -> HistorialEntrenamiento:
    """Entrena y valida un modelo durante una cantidad determinada de épocas.

    En cada época ejecuta una pasada completa sobre el conjunto de entrenamiento,
    evalúa el modelo con el conjunto de validación y registra las pérdidas,
    exactitudes y duración obtenidas. Si se proporciona un callback, se invoca
    al finalizar cada época con sus métricas.

    Args:
        modelo: Modelo que se entrenará y validará.
        train_loader: Lotes de imágenes y etiquetas usados para entrenamiento.
        val_loader: Lotes de imágenes y etiquetas usados para validación.
        criterio: Función de pérdida empleada en ambas etapas.
        optimizador: Optimizador encargado de actualizar los parámetros del
            modelo.
        dispositivo: Dispositivo donde se ejecutarán el modelo y los tensores.
        epochs: Cantidad de épocas completas que se ejecutarán.
        al_finalizar_epoch: Función opcional que recibe las métricas de cada
            época inmediatamente después de calcularlas.

    Returns:
        El historial con las métricas de entrenamiento y validación de todas las
        épocas ejecutadas.
    """
    metricas: list[MetricasEpoch] = []

    for epoch in range(1, epochs + 1):
        momento_inicio = datetime.now()
        train_loss, train_accuracy = entrenar_una_epoch(
            modelo=modelo,
            loader=train_loader,
            criterio=criterio,
            optimizador=optimizador,
            dispositivo=dispositivo,
        )
        validacion = validar_modelo(
            modelo=modelo,
            loader=val_loader,
            criterio=criterio,
            dispositivo=dispositivo,
        )
        metrica = MetricasEpoch(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=validacion.loss,
            val_accuracy=validacion.accuracy,
            duracion=datetime.now() - momento_inicio,
        )
        metricas.append(metrica)

        if al_finalizar_epoch is not None:
            al_finalizar_epoch(metrica)

    return HistorialEntrenamiento(metricas=metricas)


def graficar_historial_entrenamiento(
    historial: HistorialEntrenamiento,
) -> tuple[Figure, Figure]:
    """Grafica las curvas de pérdida y exactitud de entrenamiento y validación."""
    figura_loss, eje_loss = plt.subplots()
    eje_loss.plot(historial.train_losses, label="Train Loss")
    eje_loss.plot(historial.val_losses, label="Val Loss")
    eje_loss.legend()
    eje_loss.set_title("Loss")
    figura_loss.tight_layout()

    figura_accuracy, eje_accuracy = plt.subplots()
    eje_accuracy.plot(historial.train_accuracies, label="Train Acc")
    eje_accuracy.plot(historial.val_accuracies, label="Val Acc")
    eje_accuracy.legend()
    eje_accuracy.set_title("Accuracy")
    figura_accuracy.tight_layout()

    return figura_loss, figura_accuracy


def calcular_resultado_clasificacion(
    resultado: ResultadoEvaluacion,
    nombres_clases: Sequence[str],
) -> ResultadoClasificacion:
    """Calcula el reporte y la matriz de confusión de una evaluación."""
    ids_clases = list(range(len(nombres_clases)))
    reporte = classification_report(
        resultado.etiquetas,
        resultado.predicciones,
        labels=ids_clases,
        target_names=list(nombres_clases),
        zero_division=0,
    )
    matriz = confusion_matrix(
        resultado.etiquetas,
        resultado.predicciones,
        labels=ids_clases,
        normalize="true",
    )
    return ResultadoClasificacion(reporte=reporte, matriz_confusion=matriz)


def graficar_matriz_confusion(
    matriz: np.ndarray,
    nombres_clases: Sequence[str],
) -> Figure:
    """Grafica una matriz de confusión normalizada con etiquetas de clase."""
    figura, eje = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matriz,
        annot=True,
        fmt=".2f",
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        ax=eje,
    )
    eje.set_title("Confusion Matrix")
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Clase real")
    figura.tight_layout()
    return figura


def guardar_pesos_modelo(modelo: nn.Module, ruta_destino: Path) -> Path:
    """Guarda el estado entrenado del modelo y devuelve su ruta."""
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), ruta_destino)
    return ruta_destino
