from pathlib import Path

import pandas as pd

from models import StructuredDataInfo


class CustomerDataLoader:
    def __init__(self, structured_data_path: Path) -> None:
        self.structured_data_path = structured_data_path

    def get_data_info(self) -> StructuredDataInfo:
        csv_path = self._find_csv_file()
        if csv_path is None:
            return StructuredDataInfo(
                dataset_found=False,
                file_name=None,
                row_count=None,
                column_names=[],
            )

        dataframe = pd.read_csv(csv_path)
        return StructuredDataInfo(
            dataset_found=True,
            file_name=csv_path.name,
            row_count=len(dataframe.index),
            column_names=[str(column) for column in dataframe.columns],
        )

    def _find_csv_file(self) -> Path | None:
        if not self.structured_data_path.exists() or not self.structured_data_path.is_dir():
            return None

        csv_files = sorted(
            path for path in self.structured_data_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        )
        if not csv_files:
            return None

        return csv_files[0]
