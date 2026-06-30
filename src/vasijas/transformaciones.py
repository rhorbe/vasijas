from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms as tfs


@dataclass(frozen=True)
class ConfiguracionTransformaciones:
    img_size: int = 128
    flip_probability: float = 0.7
    scaling_factor_interval: tuple[float, float] = (0.7, 1.0)
    rotation_degrees: tuple[int, int] = (0, 360)
    pixel_fill: tuple[int, int, int] = (255, 255, 255)


def crear_transformaciones(
    config: ConfiguracionTransformaciones,
) -> list:
    return [
        tfs.RandomHorizontalFlip(p=config.flip_probability),
        tfs.RandomAffine(
            degrees=0,
            scale=config.scaling_factor_interval,
            fill=config.pixel_fill,
        ),
        tfs.RandomRotation(
            degrees=config.rotation_degrees,
            fill=config.pixel_fill,
        ),
        tfs.Resize((config.img_size, config.img_size)),
        tfs.ToTensor(),
    ]


def componer_transformaciones(
    transformaciones: Iterable,
) -> tfs.Compose:
    return tfs.Compose(list(transformaciones))


def crear_dataset_imagenes(
    ruta_datos: Path,
    config: ConfiguracionTransformaciones,
) -> datasets.ImageFolder:
    return datasets.ImageFolder(
        root=ruta_datos,
        transform=componer_transformaciones(crear_transformaciones(config)),
    )


def crear_dataloader(
    dataset: datasets.ImageFolder,
    batch_size: int = 4,
    shuffle: bool = True,
) -> data.DataLoader:
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def imagen_a_array_visualizable(imagen) -> np.ndarray:
    if torch.is_tensor(imagen):
        imagen = imagen.detach().cpu()
        if imagen.ndim == 3:
            imagen = imagen.permute(1, 2, 0)
        return imagen.numpy()

    return np.asarray(imagen)


def graficar_muestras_dataset(
    dataset: datasets.ImageFolder,
    cantidad: int = 20,
    offset: int = 10,
    columnas: int = 5,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    if cantidad <= 0:
        raise ValueError("cantidad debe ser mayor a cero")
    if columnas <= 0:
        raise ValueError("columnas debe ser mayor a cero")
    if offset < 0:
        raise ValueError("offset no puede ser negativo")

    muestras_disponibles = max(0, len(dataset) - offset)
    cantidad_muestras = min(cantidad, muestras_disponibles)
    filas = max(1, int(np.ceil(cantidad_muestras / columnas)))
    figura = plt.figure(figsize=figsize)

    for indice in range(cantidad_muestras):
        imagen, _ = dataset[indice + offset]
        eje = plt.subplot(filas, columnas, indice + 1)
        plt.tight_layout()
        plt.imshow(imagen_a_array_visualizable(imagen), cmap=plt.cm.gray_r)
        eje.set_title(f"Sample {indice}")
        eje.axis("off")

    return figura
