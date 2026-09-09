from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets

from vasijas.metadata import COLUMNA_IDENTIFICADOR

COLUMNAS_SPLIT_METADATA = ["archivo", "ruta_relativa", "clase_carpeta", "split"]


@dataclass(frozen=True)
class ConfiguracionSplitDataset:
    proporcion_temporal: float = 0.30
    proporcion_test_desde_temporal: float = 0.50
    semilla: int = 42


@dataclass(frozen=True)
class IndicesSplitDataset:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def por_indice(self) -> dict[int, str]:
        split_por_indice = {
            **{int(indice): "train" for indice in self.train},
            **{int(indice): "val" for indice in self.val},
            **{int(indice): "test" for indice in self.test},
        }
        return split_por_indice

    @property
    def conteos(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class ResultadoSplitPerfilesCompletos:
    metadata: pd.DataFrame
    registros_split: pd.DataFrame
    clases_procesadas: list[str]
    conteos_split: dict[str, int]
    ids_sin_metadata: list[str]

    @property
    def total_imagenes(self) -> int:
        return len(self.registros_split)


def normalizar_clases_seleccionadas(
    clases: Sequence[str | int] | None,
) -> set[str] | None:
    if clases is None:
        return None
    return {str(clase) for clase in clases}


def validar_clases_seleccionadas(
    clases_disponibles: Sequence[str],
    clases_a_dividir: Sequence[str | int] | None,
) -> set[str]:
    clases_normalizadas = normalizar_clases_seleccionadas(clases_a_dividir)
    if clases_normalizadas is None:
        return set(clases_disponibles)

    clases_inexistentes = sorted(clases_normalizadas - set(clases_disponibles))
    if clases_inexistentes:
        raise ValueError(
            "Las siguientes clases no existen en DATA_ORIGEN: "
            f"{clases_inexistentes}. Clases disponibles: {list(clases_disponibles)}"
        )

    return clases_normalizadas


def filtrar_muestras_por_clase(
    dataset: datasets.ImageFolder,
    clases_a_dividir: Sequence[str | int] | None,
) -> tuple[list[tuple[str, int]], list[str]]:
    clases_a_procesar = validar_clases_seleccionadas(
        clases_disponibles=dataset.classes,
        clases_a_dividir=clases_a_dividir,
    )
    indices_clases = {dataset.class_to_idx[clase] for clase in clases_a_procesar}
    muestras_filtradas = [
        (ruta, etiqueta)
        for ruta, etiqueta in dataset.samples
        if etiqueta in indices_clases
    ]

    if not muestras_filtradas:
        raise ValueError("No hay imagenes para las clases seleccionadas")

    clases_procesadas = [
        clase for clase in dataset.classes if clase in clases_a_procesar
    ]
    return muestras_filtradas, clases_procesadas


def generar_indices_split_estratificado(
    etiquetas: Sequence[int],
    configuracion_split: ConfiguracionSplitDataset,
) -> IndicesSplitDataset:
    indices = np.arange(len(etiquetas))
    etiquetas_array = np.array(etiquetas)

    train_idx, temp_idx, _, etiquetas_temporales = train_test_split(
        indices,
        etiquetas_array,
        test_size=configuracion_split.proporcion_temporal,
        stratify=etiquetas_array,
        random_state=configuracion_split.semilla,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=configuracion_split.proporcion_test_desde_temporal,
        stratify=etiquetas_temporales,
        random_state=configuracion_split.semilla,
    )

    split = IndicesSplitDataset(train=train_idx, val=val_idx, test=test_idx)
    if len(split.por_indice) != len(etiquetas):
        raise ValueError("Hay indices duplicados o faltantes en el split")

    return split


def construir_dataframe_split_imagenes(
    data_origen: Path,
    muestras: Sequence[tuple[str, int]],
    clases: Sequence[str],
    split_por_indice: dict[int, str],
) -> pd.DataFrame:
    registros_split: list[dict[str, Any]] = []
    data_origen = data_origen.resolve()

    for indice, (ruta_imagen, etiqueta) in enumerate(muestras):
        ruta_absoluta = Path(ruta_imagen).resolve()
        ruta_relativa = ruta_absoluta.relative_to(data_origen)
        registros_split.append(
            {
                COLUMNA_IDENTIFICADOR: ruta_absoluta.stem,
                "archivo": ruta_absoluta.name,
                "ruta_relativa": ruta_relativa.as_posix(),
                "clase_carpeta": clases[etiqueta],
                "split": split_por_indice[indice],
            }
        )

    return pd.DataFrame(registros_split)


def construir_split_perfiles_completos(
    data_origen: Path,
    configuracion_split: ConfiguracionSplitDataset,
    clases_a_dividir: Sequence[str | int] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    dataset = datasets.ImageFolder(data_origen)
    muestras_filtradas, clases_procesadas = filtrar_muestras_por_clase(
        dataset=dataset,
        clases_a_dividir=clases_a_dividir,
    )
    etiquetas = [etiqueta for _, etiqueta in muestras_filtradas]
    indices_split = generar_indices_split_estratificado(
        etiquetas=etiquetas,
        configuracion_split=configuracion_split,
    )
    df_split = construir_dataframe_split_imagenes(
        data_origen=data_origen,
        muestras=muestras_filtradas,
        clases=dataset.classes,
        split_por_indice=indices_split.por_indice,
    )

    return df_split, clases_procesadas, indices_split.conteos


def construir_metadata_con_split(
    metadata_origen: pd.DataFrame,
    df_split: pd.DataFrame,
    columna_identificador: str = COLUMNA_IDENTIFICADOR,
) -> tuple[pd.DataFrame, list[str]]:
    df_metadata = metadata_origen.copy()
    if "split" in df_metadata.columns:
        df_metadata = df_metadata.drop(columns=["split"])

    ids_imagenes = set(df_split[columna_identificador])
    ids_metadata_origen = set(df_metadata[columna_identificador])
    ids_sin_metadata = sorted(ids_imagenes - ids_metadata_origen)

    df_metadata = df_metadata[
        df_metadata[columna_identificador].isin(ids_imagenes)
    ].copy()
    df_metadata_split = df_metadata.merge(
        df_split[[columna_identificador, *COLUMNAS_SPLIT_METADATA]],
        on=columna_identificador,
        how="left",
        validate="one_to_one",
    )

    faltantes_en_imagenes = df_metadata_split["split"].isna().sum()
    if faltantes_en_imagenes:
        ids_faltantes = df_metadata_split.loc[
            df_metadata_split["split"].isna(), columna_identificador
        ].tolist()
        raise ValueError(
            "Hay registros de metadata sin imagen procesada en DATA_CUADRADOS: "
            f"{ids_faltantes[:10]}"
        )

    return df_metadata_split, ids_sin_metadata


def preparar_metadata_split_perfiles_completos(
    data_origen: Path,
    metadata_origen: pd.DataFrame,
    configuracion_split: ConfiguracionSplitDataset,
    clases_a_dividir: Sequence[str | int] | None = None,
) -> ResultadoSplitPerfilesCompletos:
    df_split, clases_procesadas, conteos_split = construir_split_perfiles_completos(
        data_origen=data_origen,
        configuracion_split=configuracion_split,
        clases_a_dividir=clases_a_dividir,
    )
    metadata_split, ids_sin_metadata = construir_metadata_con_split(
        metadata_origen=metadata_origen,
        df_split=df_split,
    )

    return ResultadoSplitPerfilesCompletos(
        metadata=metadata_split,
        registros_split=df_split,
        clases_procesadas=clases_procesadas,
        conteos_split=conteos_split,
        ids_sin_metadata=ids_sin_metadata,
    )


def copiar_imagenes_split(
    data_origen: Path,
    data_destino: Path,
    registros_split: pd.DataFrame,
) -> int:
    """Copia las imágenes procesadas conservando las carpetas y rutas relativas.

    La partición train/val/test se conserva en la metadata, sin agregar carpetas
    por split. Los archivos existentes con la misma ruta se sobrescriben.
    """
    data_origen = Path(data_origen).resolve()
    data_destino = Path(data_destino).resolve()
    if data_origen == data_destino:
        raise ValueError("El destino debe ser distinto del origen")

    for ruta in registros_split["ruta_relativa"]:
        ruta_relativa = Path(ruta)
        origen = data_origen / ruta_relativa
        destino = data_destino / ruta_relativa
        destino.parent.mkdir(parents=True, exist_ok=True)
        copy2(origen, destino)

    return len(registros_split)


def calcular_proporciones_split(
    configuracion: ConfiguracionSplitDataset,
) -> dict[str, float]:
    proporcion_validacion = configuracion.proporcion_temporal * (
        1 - configuracion.proporcion_test_desde_temporal
    )
    proporcion_test = (
        configuracion.proporcion_temporal * configuracion.proporcion_test_desde_temporal
    )

    return {
        "train": 1 - configuracion.proporcion_temporal,
        "val": proporcion_validacion,
        "test": proporcion_test,
    }


def describir_configuracion_split(configuracion: ConfiguracionSplitDataset) -> str:
    proporciones = calcular_proporciones_split(configuracion)
    return (
        "Split configurado -> "
        f"train: {proporciones['train']:.0%}, "
        f"val: {proporciones['val']:.0%}, "
        f"test: {proporciones['test']:.0%}, "
        f"semilla: {configuracion.semilla}"
    )


def advertir_imagenes_sin_metadata(ids_sin_metadata: list[str]) -> None:
    if not ids_sin_metadata:
        return

    print(
        "Advertencia: hay imagenes sin registro en metadata. "
        f"Cantidad: {len(ids_sin_metadata)}. "
        f"Primeros casos: {ids_sin_metadata[:10]}"
    )


def mostrar_resumen_split(resultado) -> None:
    conteos = resultado.conteos_split
    print("Clases procesadas:", resultado.clases_procesadas)
    print("Imagenes procesadas:", resultado.total_imagenes)
    print(
        f"Split -> train: {conteos['train']}, "
        f"val: {conteos['val']}, "
        f"test: {conteos['test']}"
    )


def construir_resumen_split(metadata_split: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        metadata_split["clase_carpeta"],
        metadata_split["split"],
        margins=True,
    )
