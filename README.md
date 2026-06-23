# vasijas

## Quarto

El EDA del notebook `notebooks/01.exploracion/01. EDA.ipynb` también está disponible como reporte Quarto en `notebooks/01.exploracion/01. EDA.qmd`.

Quarto se instala como herramienta externa, no como dependencia Python del `pyproject.toml`. Las dependencias Python necesarias para ejecutar el reporte ya están declaradas en el proyecto.

Para renderizarlo desde la raíz:

```bash
quarto render "notebooks/01.exploracion/01. EDA.qmd"
```

La salida se escribe en `output/quarto`.
