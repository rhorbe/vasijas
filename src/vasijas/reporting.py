from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass
class ProcessingReport:
    fecha: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def save_markdown(self, output_path: Path) -> None:
        lines = [
            "# Processing Report",
            f"Fecha: {self.fecha}",
            "",
            "## Métricas",
        ]

        for k, v in self.metrics.items():
            lines.append(f"- **{k}**: {v}")

        output_path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class ProcessingResult:
    df: pd.DataFrame
    report: ProcessingReport