from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


def graficar_error_por_grupo(
    tabla_predicciones: pd.DataFrame,
    columna: str,
    etiqueta: str,
    tipo_grafico: Literal["lineas", "barras"] = "lineas",
) -> pd.DataFrame:
    """Grafica el porcentaje de error por grupo y devuelve sus recuentos.

    Incluye los grupos sin dato y conserva intacta la tabla de predicciones.
    ``tipo_grafico`` permite elegir entre "lineas" (línea con puntos,
    predeterminado) y "barras". Las etiquetas indican el porcentaje de errores
    y la cantidad de imágenes. Rechaza tipos de gráfico no admitidos.
    """
    if tipo_grafico not in ("lineas", "barras"):
        raise ValueError('tipo_grafico debe ser "lineas" o "barras".')

    datos = tabla_predicciones[[columna, "acierto"]].copy()
    datos["error"] = ~datos["acierto"]
    resumen = (
        datos.groupby(columna, dropna=False, observed=True, sort=True)["error"]
        .agg(errores="sum", imagenes="size")
        .reset_index()
    )
    resumen["porcentaje_error"] = 100 * resumen["errores"] / resumen["imagenes"]
    etiquetas = [
        "Sin dato" if pd.isna(valor) else str(valor) for valor in resumen[columna]
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(resumen) * 0.65), 5))
    if tipo_grafico == "lineas":
        ax.plot(
            etiquetas,
            resumen["porcentaje_error"],
            marker="o",
            linestyle="-",
            color="#307ae3",
        )
    else:
        ax.bar(etiquetas, resumen["porcentaje_error"], color="#307ae3")
    for posicion, (porcentaje, cantidad) in enumerate(
        zip(resumen["porcentaje_error"], resumen["imagenes"])
    ):
        ax.annotate(
            f"{porcentaje:.1f}%\n(n={cantidad})",
            xy=(posicion, porcentaje),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set(
        title=f"{etiqueta}",
        xlabel=etiqueta,
        ylabel="Predicciones incorrectas",
    )
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.tick_params(axis="x", rotation=45)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()
    return resumen


def graficar_error_por_avance_y_origen(
    tabla_predicciones: pd.DataFrame,
    tipo_grafico: Literal["lineas", "barras"] = "lineas",
) -> pd.DataFrame:
    """Compara el error por avance y origen con los perfiles completos.

    Calcula errores / imágenes para cada par (avance, origen). Los perfiles
    completos se representan como una referencia horizontal con su error real.
    Devuelve los recuentos y porcentajes, sin modificar la tabla original.
    """
    if tipo_grafico not in ("lineas", "barras"):
        raise ValueError('tipo_grafico debe ser "lineas" o "barras".')

    datos = tabla_predicciones[["porcentaje_avance", "borde_origen", "acierto"]].copy()
    datos["error"] = ~datos["acierto"]
    resumen = (
        datos.groupby(
            ["porcentaje_avance", "borde_origen"],
            dropna=False,
            observed=True,
            sort=True,
        )["error"]
        .agg(errores="sum", imagenes="size")
        .reset_index()
    )
    resumen["porcentaje_error"] = 100 * resumen["errores"] / resumen["imagenes"]
    completos = resumen.loc[resumen["borde_origen"].eq("completo")]
    cortes = resumen.loc[
        ~resumen["borde_origen"].eq("completo") & resumen["porcentaje_avance"].notna()
    ].copy()
    cortes["borde_origen"] = cortes["borde_origen"].fillna("Sin dato")
    origenes = sorted(
        cortes["borde_origen"].unique(),
        key=lambda origen: (origen != "tope", origen),
    )
    avances = sorted(cortes["porcentaje_avance"].unique())
    colores = {"tope": "#1684b5", "base": "#ee7a19"}
    marcadores = {"tope": "o", "base": "s"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for indice, origen in enumerate(origenes):
        serie = cortes.loc[cortes["borde_origen"].eq(origen)]
        if tipo_grafico == "lineas":
            ax.plot(
                serie["porcentaje_avance"],
                serie["porcentaje_error"],
                label=origen.capitalize(),
                color=colores.get(origen),
                marker=marcadores.get(origen, "o"),
            )
        else:
            # Barras contiguas en cada avance; los grupos ausentes no se rellenan.
            paso = min((b - a for a, b in zip(avances, avances[1:])), default=10)
            ancho = paso * 0.8 / len(origenes)
            desplazamiento = (indice - (len(origenes) - 1) / 2) * ancho
            ax.bar(
                serie["porcentaje_avance"] + desplazamiento,
                serie["porcentaje_error"],
                width=ancho,
                label=origen.capitalize(),
                color=colores.get(origen),
            )
    if not completos.empty:
        error_completos = 100 * completos["errores"].sum() / completos["imagenes"].sum()
        ax.axhline(
            error_completos,
            color="#7c8797",
            linestyle="--",
            label=f"Perfiles completos: {error_completos:.1f} %",
        )
    ax.set(
        title="Error según avance del corte y borde de origen",
        xlabel="Avance del corte (%)",
        ylabel="Error (%)",
        ylim=(0, 100),
    )
    if avances:
        ax.set_xticks(avances)
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
    return resumen


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
    if (
        "Identificador" not in metadata_split
        or metadata_split["Identificador"].isna().any()
    ):
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
        ruta_relativa=metadata_split["ruta_relativa"].map(
            lambda ruta: Path(ruta).as_posix()
        )
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


def crear_loader_test_con_ruta(
    subset: Dataset,
    data_dir: Path,
    metadata_split: pd.DataFrame,
    batch_size: int,
    dispositivo: torch.device,
) -> DataLoader:
    """Carga todo el test persistido, conservando la ruta y el último lote."""
    cantidad_test = contar_imagenes_segun_conjunto(metadata_split, "test")
    if len(subset) != cantidad_test:
        raise ValueError(
            "El dataset de prueba no coincide con el tamaño del split test."
        )
    return DataLoader(
        DatasetTestConRuta(subset, data_dir),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=dispositivo.type == "cuda",
    )


def cargar_modelo_resnet18_evaluacion(
    ruta_modelo: Path,
    nombres_clases: Sequence[str],
    dispositivo: torch.device,
) -> nn.Module:
    """Reconstruye ResNet18 desde un state_dict local, sin descargar pesos.

    Las clases deben conservar el orden usado durante el entrenamiento. El
    state_dict permite comprobar su cantidad, pero no contiene sus nombres.
    """
    state_dict = torch.load(ruta_modelo, map_location="cpu", weights_only=True)
    num_classes = state_dict["fc.weight"].shape[0]
    if num_classes != len(nombres_clases):
        raise ValueError(
            f"El modelo tiene {num_classes} salidas y el dataset "
            f"{len(nombres_clases)} clases."
        )
    modelo = models.resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)
    modelo.load_state_dict(state_dict, strict=True)
    modelo = modelo.to(dispositivo)
    modelo.eval()
    return modelo


def predecir_test_con_rutas(
    modelo: nn.Module,
    test_loader: DataLoader,
    metadata_por_ruta: pd.DataFrame,
    nombres_clases: Sequence[str],
    dispositivo: torch.device,
    informar_progreso: Callable[[int, int], None] | None = None,
    intervalo_progreso: int = 25,
) -> pd.DataFrame:
    """Predice por lotes y relaciona cada resultado con su ruta e identificador.

    El callback opcional recibe las cantidades procesada y total. La función
    no imprime ni exporta, para que el notebook decida cómo mostrar el avance.
    """
    if intervalo_progreso < 1:
        raise ValueError("El intervalo de progreso debe ser mayor que cero.")
    filas = []
    modelo.eval()
    with torch.inference_mode():
        for numero_lote, (imagenes, etiquetas, rutas) in enumerate(
            test_loader, start=1
        ):
            salidas = modelo(
                imagenes.to(dispositivo, non_blocking=dispositivo.type == "cuda")
            )
            predicciones = salidas.argmax(dim=1).cpu().tolist()
            for ruta, real, predicha in zip(
                rutas, etiquetas.tolist(), predicciones, strict=True
            ):
                filas.append(
                    {
                        "Identificador": metadata_por_ruta.at[ruta, "Identificador"],
                        "ruta_relativa": ruta,
                        "clase_real": nombres_clases[real],
                        "clase_predicha": nombres_clases[predicha],
                        "acierto": real == predicha,
                    }
                )
            if informar_progreso is not None and (
                numero_lote % intervalo_progreso == 0 or numero_lote == len(test_loader)
            ):
                informar_progreso(len(filas), len(test_loader.dataset))
    return pd.DataFrame(
        filas,
        columns=[
            "Identificador",
            "ruta_relativa",
            "clase_real",
            "clase_predicha",
            "acierto",
        ],
    )


def validar_predicciones_test(
    predicciones: pd.DataFrame, metadata_split: pd.DataFrame
) -> None:
    """Exige una predicción completa por imagen de test y aciertos coherentes."""
    columnas = {
        "Identificador",
        "ruta_relativa",
        "clase_real",
        "clase_predicha",
        "acierto",
    }
    if not columnas.issubset(predicciones.columns):
        raise ValueError("Faltan columnas requeridas en las predicciones.")
    if predicciones[list(columnas)].isna().any().any():
        raise ValueError("Las predicciones contienen valores nulos.")
    metadata_test = metadata_split.loc[metadata_split["split"].eq("test")]
    rutas_test = set(
        metadata_test["ruta_relativa"].map(lambda ruta: Path(ruta).as_posix())
    )
    if (
        len(predicciones) != len(metadata_test)
        or not predicciones["ruta_relativa"].is_unique
        or set(predicciones["ruta_relativa"]) != rutas_test
    ):
        raise ValueError("Debe existir exactamente una predicción por imagen de test.")
    identificadores_esperados = predicciones["ruta_relativa"].map(
        indexar_metadata_por_ruta(metadata_test)["Identificador"]
    )
    if not predicciones["Identificador"].equals(identificadores_esperados):
        raise ValueError(
            "Los identificadores de las predicciones no coinciden con sus rutas."
        )
    if not predicciones["acierto"].equals(
        predicciones["clase_real"].eq(predicciones["clase_predicha"])
    ):
        raise ValueError("Los aciertos no coinciden con las clases reales y predichas.")


def crear_tabla_predicciones_con_metadata(
    predicciones: pd.DataFrame,
    metadata_original: pd.DataFrame,
    columnas_prediccion: Sequence[str] = ("clase_predicha", "acierto"),
) -> pd.DataFrame:
    """Une por identificador, preservando columnas, nulos y orden de predicciones."""
    validar_identificadores_metadata(metadata_original)
    validar_identificadores_split(predicciones)
    validar_identificadores_test(predicciones["Identificador"], metadata_original)
    if predicciones.empty:
        tabla = metadata_original.iloc[:0].copy().reset_index(drop=True)
        for columna in columnas_prediccion:
            tabla[columna] = predicciones[columna].reset_index(drop=True)
        return tabla
    tabla = predicciones[["Identificador", *columnas_prediccion]].merge(
        metadata_original,
        on="Identificador",
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not tabla["_merge"].eq("both").all():
        raise ValueError("Hay predicciones sin metadata original.")
    tabla = tabla[list(metadata_original.columns) + list(columnas_prediccion)]
    pd.testing.assert_frame_equal(
        tabla[list(metadata_original.columns)].reset_index(drop=True),
        metadata_original.set_index("Identificador", drop=False)
        .loc[tabla["Identificador"]]
        .reset_index(drop=True),
    )
    return tabla


def convertir_clases_a_enteros(tabla_predicciones: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con Clase Ana y clase_predicha convertidas a enteros.

    Elimina el texto ``clase`` de las etiquetas predichas antes de convertirlas.
    """
    tabla_predicciones_enteros = tabla_predicciones.copy()
    tabla_predicciones_enteros["Clase Ana"] = tabla_predicciones_enteros[
        "Clase Ana"
    ].apply(int)
    tabla_predicciones_enteros["clase_predicha"] = tabla_predicciones_enteros[
        "clase_predicha"
    ].apply(lambda x: int(x.replace("clase", "")))
    return tabla_predicciones_enteros


def seleccionar_predicciones_por_borde_origen(
    tabla_predicciones: pd.DataFrame,
    borde_origen: Literal["base", "tope", "completo"],
) -> pd.DataFrame:
    """Devuelve una copia de las predicciones del borde de origen indicado."""
    if borde_origen not in ("base", "tope", "completo"):
        raise ValueError('borde_origen debe ser "base", "tope" o "completo".')
    return tabla_predicciones.loc[
        tabla_predicciones["borde_origen"].eq(borde_origen)
    ].copy()


def obtener_clases_reales_y_predichas(
    tabla_predicciones: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (y_true, y_pred) como arrays, conservando el orden de las filas."""
    return (
        tabla_predicciones["Clase Ana"].to_numpy(),
        tabla_predicciones["clase_predicha"].to_numpy(),
    )


def guardar_tabla_predicciones(tabla: pd.DataFrame, ruta_destino: Path) -> Path:
    """Exporta la tabla completa en CSV UTF-8 con BOM, sin índice."""
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_destino, index=False, encoding="utf-8-sig")
    return ruta_destino


def seleccionar_metadata_sin_clase(metadata: pd.DataFrame) -> pd.DataFrame:
    """Selecciona clases nulas, vacías o con espacios sin alterar la metadata."""
    validar_identificadores_metadata(metadata)
    if "Clase Ana" not in metadata:
        raise ValueError("La metadata no incluye la columna Clase Ana.")
    clase = metadata["Clase Ana"]
    sin_clase = clase.isna() | clase.astype("string").str.strip().eq("").fillna(False)
    return metadata.loc[sin_clase].copy()


class DatasetImagenesSinClase(Dataset):
    """Busca imágenes por identificador, sin asignarles una etiqueta ficticia.

    Exige un único archivo por registro seleccionado en cualquier subcarpeta
    de data_dir. Mantiene el orden de la metadata y carga RGB como ImageFolder.
    """

    def __init__(
        self, metadata: pd.DataFrame, data_dir: Path, transformacion: Callable
    ):
        if len(seleccionar_metadata_sin_clase(metadata)) != len(metadata):
            raise ValueError(
                "El dataset de inferencia debe contener solo perfiles sin clase."
            )
        self.data_dir = data_dir.resolve()
        self.transformacion = transformacion
        identificadores = metadata["Identificador"].tolist()
        rutas_por_id = {}
        buscados = set(map(str, identificadores))
        for ruta in self.data_dir.rglob("*"):
            if (
                ruta.is_file()
                and ruta.suffix.lower() in IMG_EXTENSIONS
                and ruta.stem in buscados
            ):
                if ruta.stem in rutas_por_id:
                    raise ValueError(
                        f"Hay varias imágenes para el identificador {ruta.stem!r}."
                    )
                rutas_por_id[ruta.stem] = ruta
        faltantes = buscados - set(rutas_por_id)
        if faltantes:
            raise FileNotFoundError(
                f"Faltan imágenes para estos identificadores: {sorted(faltantes)[:10]}"
            )
        self.registros = [
            (identificador, rutas_por_id[str(identificador)])
            for identificador in identificadores
        ]

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, indice):
        identificador, ruta = self.registros[indice]
        imagen = self.transformacion(default_loader(str(ruta)))
        return imagen, identificador, ruta.relative_to(self.data_dir).as_posix()


def crear_loader_imagenes_sin_clase(
    metadata: pd.DataFrame,
    data_dir: Path,
    transformacion: Callable,
    batch_size: int,
    dispositivo: torch.device,
) -> DataLoader:
    """Prepara todas las imágenes seleccionadas, sin mezclar ni descartar filas."""
    return DataLoader(
        DatasetImagenesSinClase(metadata, data_dir, transformacion),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=dispositivo.type == "cuda",
    )


def predecir_imagenes_sin_clase(
    modelo: nn.Module,
    loader: DataLoader,
    nombres_clases: Sequence[str],
    dispositivo: torch.device,
) -> pd.DataFrame:
    """Devuelve clase y softmax máximo; no calcula aciertos sin referencia."""
    filas = []
    modelo.eval()
    with torch.inference_mode():
        for imagenes, identificadores, rutas in loader:
            salidas = modelo(
                imagenes.to(dispositivo, non_blocking=dispositivo.type == "cuda")
            )
            if salidas.ndim != 2 or salidas.shape[1] != len(nombres_clases):
                raise ValueError(
                    "Las salidas del modelo no coinciden con las clases indicadas."
                )
            confianza, indices = salidas.softmax(dim=1).max(dim=1)
            if isinstance(identificadores, torch.Tensor):
                identificadores = identificadores.tolist()
            for identificador, ruta, indice, valor in zip(
                identificadores,
                rutas,
                indices.cpu().tolist(),
                confianza.cpu().tolist(),
                strict=True,
            ):
                filas.append(
                    {
                        "Identificador": identificador,
                        "ruta_relativa": ruta,
                        "clase_predicha": nombres_clases[indice],
                        "confianza": valor,
                    }
                )
    return pd.DataFrame(
        filas, columns=["Identificador", "ruta_relativa", "clase_predicha", "confianza"]
    )


def validar_predicciones_sin_clase(
    predicciones: pd.DataFrame,
    metadata_sin_clase: pd.DataFrame,
    nombres_clases: Sequence[str],
) -> None:
    """Comprueba cobertura, identidad y resultados válidos de la inferencia."""
    columnas = ["Identificador", "ruta_relativa", "clase_predicha", "confianza"]
    if not set(columnas).issubset(predicciones.columns):
        raise ValueError("Faltan columnas requeridas en las predicciones sin clase.")
    if predicciones[columnas].isna().any().any():
        raise ValueError("Las predicciones sin clase contienen valores nulos.")
    if (
        len(predicciones) != len(metadata_sin_clase)
        or not predicciones["Identificador"].is_unique
        or not predicciones["ruta_relativa"].is_unique
        or set(predicciones["Identificador"])
        != set(metadata_sin_clase["Identificador"])
    ):
        raise ValueError(
            "Debe existir exactamente una predicción por perfil sin clase."
        )
    if (
        not predicciones["ruta_relativa"]
        .map(lambda ruta: Path(ruta).stem)
        .eq(predicciones["Identificador"].astype(str))
        .all()
    ):
        raise ValueError("Las rutas no corresponden a los identificadores predichos.")
    if not predicciones["clase_predicha"].isin(nombres_clases).all():
        raise ValueError("Hay clases predichas que no pertenecen al modelo.")
    if not predicciones["confianza"].between(0, 1).all():
        raise ValueError("La confianza softmax debe estar entre cero y uno.")
