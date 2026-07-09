from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from torchvision import transforms as tfs
from torchvision.models import ResNet18_Weights, VGG16_Weights


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
class ConfiguracionSplitDataset:
    proporcion_temporal: float = 0.30
    proporcion_test_desde_temporal: float = 0.50
    semilla: int = 42


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


def seleccionar_dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describir_dispositivo(dispositivo: torch.device) -> str:
    if dispositivo.type == "cuda":
        return f"Dispositivo seleccionado para procesamiento: CUDA - {torch.cuda.get_device_name(0)}"
    return "Dispositivo seleccionado para procesamiento: CPU"


def crear_transformaciones_imagen(
    configuracion: ConfiguracionTransformacionesImagen,
) -> tuple[tfs.Compose, tfs.Compose, tfs.Compose]:
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
    dataset_completo = datasets.ImageFolder(data_dir)
    indices = np.arange(len(dataset_completo))
    etiquetas = np.array(dataset_completo.targets)

    train_idx, temp_idx, _, etiquetas_temporales = train_test_split(
        indices,
        etiquetas,
        test_size=configuracion_split.proporcion_temporal,
        stratify=etiquetas,
        random_state=configuracion_split.semilla,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=configuracion_split.proporcion_test_desde_temporal,
        stratify=etiquetas_temporales,
        random_state=configuracion_split.semilla,
    )

    return DatasetsImagen(
        train=TransformSubset(
            dataset_completo, train_idx, transformacion_entrenamiento
        ),
        val=TransformSubset(dataset_completo, val_idx, transformacion_validacion),
        test=TransformSubset(dataset_completo, test_idx, transformacion_test),
        class_names=dataset_completo.classes,
    )


def crear_dataloaders_imagen(
    data_dir: Path,
    batch_size: int,
    configuracion_transformaciones: ConfiguracionTransformacionesImagen,
    configuracion_split: ConfiguracionSplitDataset,
) -> DataLoadersImagen:
    train_transform, val_transform, test_transform = crear_transformaciones_imagen(
        configuracion_transformaciones
    )
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


def crear_modelo_transfer_learning(
    nombre_modelo: str,
    num_classes: int,
    dispositivo: torch.device,
) -> nn.Module:
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
