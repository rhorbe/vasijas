from pathlib import Path
from typing import Literal

import pandas as pd
from torch.utils.data import Dataset


def validar_identificadores_metadata(metadata_unificado: pd.DataFrame) -> None:
    """Verifica que la metadata original tenga identificadores no nulos y únicos."""
    if "Identificador" not in metadata_unificado:
        raise ValueError("La metadata original no incluye la columna Identificador.")
    if metadata_unificado["Identificador"].isna().any():
        raise ValueError("La metadata original contiene identificadores nulos.")
    if metadata_unificado["Identificador"].duplicated().any():
        raise ValueError("La metadata original contiene identificadores duplicados.")


def validar_identificadores_split(metadata_split: pd.DataFrame) -> None:
    """Verifica que cada imagen del split tenga un identificador no nulo."""
    if "Identificador" not in metadata_split or metadata_split["Identificador"].isna().any():
        raise ValueError("La metadata debe incluir un Identificador para cada imagen.")


def contar_imagenes_segun_conjunto(
    metadata_split: pd.DataFrame, conjunto: Literal["train", "test", "val"]
) -> int:
    """Cuenta las imágenes del conjunto indicado; rechaza conjuntos no admitidos."""
    if conjunto not in ("train", "test", "val"):
        raise ValueError('El conjunto debe ser "train", "test" o "val".')
    return int(metadata_split["split"].eq(conjunto).sum())


def indexar_metadata_por_ruta(metadata_split: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la metadata indexada por rutas POSIX sin modificar el original."""
    return metadata_split.assign(
        ruta_relativa=metadata_split["ruta_relativa"].map(lambda ruta: Path(ruta).as_posix())
    ).set_index("ruta_relativa")


def obtener_identificadores_test(metadata_split: pd.DataFrame) -> pd.Series:
    """Devuelve los identificadores del split de test conservando su índice y orden."""
    return metadata_split.loc[metadata_split["split"].eq("test"), "Identificador"]


def validar_identificadores_test(
    identificadores_test: pd.Series, metadata_unificado: pd.DataFrame
) -> None:
    """Verifica que los identificadores de test sean únicos y tengan metadata original."""
    if identificadores_test.duplicated().any():
        raise ValueError("Hay identificadores de prueba duplicados en el split.")
    faltantes = set(identificadores_test) - set(metadata_unificado["Identificador"])
    if faltantes:
        raise ValueError(
            f"Falta metadata original para estas imágenes: {sorted(faltantes)[:10]}"
        )


class DatasetTestConRuta(Dataset):
    """Devuelve la ruta junto a la imagen para conservar su identidad por lote."""

    def __init__(self, subset, data_dir):
        self.subset = subset
        self.data_dir = data_dir.resolve()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, indice):
        imagen, etiqueta = self.subset[indice]
        ruta, _ = self.subset.dataset.samples[self.subset.indices[indice]]
        ruta_relativa = Path(ruta).resolve().relative_to(self.data_dir).as_posix()
        return imagen, etiqueta, ruta_relativa
