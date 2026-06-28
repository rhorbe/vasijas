import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetFuente:
    ruta: Path
    etiqueta: str


@dataclass(frozen=True)
class RegistroDatasetUnificado:
    archivo_generado: str
    archivo_origen: str
    origen: str
    clase_carpeta: str


@dataclass(frozen=True)
class ResultadoUnionDatasets:
    archivos_copiados: int
    registros: list[RegistroDatasetUnificado]
    metadata_destino: Path

    @property
    def dataframe_registros(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "archivo_generado": registro.archivo_generado,
                    "archivo_origen": registro.archivo_origen,
                    "origen": registro.origen,
                    "clase_carpeta": registro.clase_carpeta,
                }
                for registro in self.registros
            ]
        )

    @property
    def mensaje_resumen(self) -> str:
        return (
            f"Copiados {self.archivos_copiados} archivos | "
            f"Metadata generada: {self.metadata_destino}"
        )


def copiar_archivo_preservando_relativa(
    archivo_origen: Path,
    ruta_origen_base: Path,
    ruta_destino_base: Path,
) -> Path:
    ruta_relativa = archivo_origen.relative_to(ruta_origen_base)
    ruta_destino = ruta_destino_base / ruta_relativa
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archivo_origen, ruta_destino)
    return ruta_destino


def copiar_png_dataset(
    dataset: DatasetFuente,
    ruta_destino: Path,
    patron: str = "*.png",
) -> list[RegistroDatasetUnificado]:
    registros: list[RegistroDatasetUnificado] = []

    for archivo_origen in sorted(dataset.ruta.rglob(patron)):
        if not archivo_origen.is_file():
            continue

        archivo_destino = copiar_archivo_preservando_relativa(
            archivo_origen=archivo_origen,
            ruta_origen_base=dataset.ruta,
            ruta_destino_base=ruta_destino,
        )
        registros.append(
            RegistroDatasetUnificado(
                archivo_generado=archivo_destino.name,
                archivo_origen=archivo_origen.name,
                origen=dataset.etiqueta,
                clase_carpeta=archivo_destino.parent.name,
            )
        )

    return registros


def unir_datasets_png(
    datasets_fuente: Iterable[DatasetFuente],
    ruta_destino: Path,
    nombre_metadata: str = "metadata.csv",
    patron: str = "*.png",
) -> ResultadoUnionDatasets:
    ruta_destino.mkdir(parents=True, exist_ok=True)
    registros: list[RegistroDatasetUnificado] = []

    for dataset in datasets_fuente:
        registros.extend(
            copiar_png_dataset(
                dataset=dataset,
                ruta_destino=ruta_destino,
                patron=patron,
            )
        )

    resultado = ResultadoUnionDatasets(
        archivos_copiados=len(registros),
        registros=registros,
        metadata_destino=ruta_destino / nombre_metadata,
    )
    resultado.dataframe_registros.to_csv(resultado.metadata_destino, index=False)

    return resultado
