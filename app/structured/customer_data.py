from pathlib import Path

import pandas as pd

from models import StructuredDataInfo


class CustomerDataLoader:
    def __init__(self, structured_data_path: Path) -> None:
        self.structured_data_path = structured_data_path
        self._csv_path: Path | None = None
        self._dataframe: pd.DataFrame | None = None
        self._load_error: str | None = None

    def get_data_info(self) -> StructuredDataInfo:
        dataframe = self.get_dataframe()
        csv_path = self._find_csv_file()
        if csv_path is None:
            return StructuredDataInfo(
                dataset_found=False,
                file_name=None,
                row_count=None,
                column_names=[],
            )

        if dataframe is None:
            return StructuredDataInfo(
                dataset_found=True,
                file_name=csv_path.name,
                row_count=None,
                column_names=[],
            )

        return StructuredDataInfo(
            dataset_found=True,
            file_name=csv_path.name,
            row_count=len(dataframe.index),
            column_names=[str(column) for column in dataframe.columns],
        )

    def get_dataframe(self) -> pd.DataFrame | None:
        if self._dataframe is not None:
            return self._dataframe.copy()

        if self._load_error is not None:
            return None

        csv_path = self._find_csv_file()
        if csv_path is None:
            return None

        try:
            self._dataframe = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            self._load_error = str(exc)
            return None

        return self._dataframe.copy()

    def get_dataset_file_name(self) -> str | None:
        csv_path = self._find_csv_file()
        if csv_path is None:
            return None
        return csv_path.name

    def get_load_error(self) -> str | None:
        return self._load_error

    def _find_csv_file(self) -> Path | None:
        if self._csv_path is not None:
            return self._csv_path

        if not self.structured_data_path.exists() or not self.structured_data_path.is_dir():
            return None

        csv_files = sorted(
            path for path in self.structured_data_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        )
        if not csv_files:
            return None

        self._csv_path = csv_files[0]
        return self._csv_path
