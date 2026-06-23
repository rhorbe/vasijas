from pathlib import Path


def get_ubicaciones(data: Path) -> list[str]:
    return sorted(
        item.name
        for item in data.iterdir()
        if item.is_dir()
    )
