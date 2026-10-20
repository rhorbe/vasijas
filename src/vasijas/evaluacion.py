from pathlib import Path

from torch.utils.data import Dataset


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
