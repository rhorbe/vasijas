from pathlib import Path


def get_ubicaciones(data: Path) -> list[str]:
    return sorted(
        item.name
        for item in data.iterdir()
        if item.is_dir()
    )


def contar_perfiles_por_ubicacion(data: Path) -> tuple[int, list[tuple[str, int]]]:
    conteo_png = {
        ruta_ubicacion.name: len(list(ruta_ubicacion.glob("*.png")))
        for ruta_ubicacion in data.iterdir()
        if ruta_ubicacion.is_dir()
    }

    total_png = sum(conteo_png.values())
    conteo_png_ordenado = sorted(conteo_png.items(), key=lambda item: item[1])
    return total_png, conteo_png_ordenado


def get_nombres_archivos_perfiles(data: Path) -> set[str]:
    return {p.stem for p in data.rglob("*.png")}
