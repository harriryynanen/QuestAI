from pathlib import Path

import pandas as pd

from models import StructuredDataInfo, StructuredDatasetName


class CustomerDataLoader:
    """Load explicitly supported structured demo datasets from disk."""

    DATASET_FILES: dict[StructuredDatasetName, str] = {
        "customer_portfolio": "demo_customer_portfolio.csv",
        "advisory_case_pipeline": "demo_advisory_case_pipeline.csv",
    }

    def __init__(self, structured_data_path: Path) -> None:
        self.structured_data_path = structured_data_path
        self._dataframes: dict[StructuredDatasetName, pd.DataFrame] = {}
        self._load_errors: dict[StructuredDatasetName, str] = {}

    def get_data_info(
        self,
        dataset_name: StructuredDatasetName = "customer_portfolio",
    ) -> StructuredDataInfo:
        dataframe = self.get_dataframe(dataset_name)
        csv_path = self._get_csv_path(dataset_name)
        if csv_path is None:
            return StructuredDataInfo(
                dataset_found=False,
                file_name=None,
                row_count=None,
                column_names=[],
                dataset_name=dataset_name,
            )

        if dataframe is None:
            return StructuredDataInfo(
                dataset_found=True,
                file_name=csv_path.name,
                row_count=None,
                column_names=[],
                dataset_name=dataset_name,
            )

        return StructuredDataInfo(
            dataset_found=True,
            file_name=csv_path.name,
            row_count=len(dataframe.index),
            column_names=[str(column) for column in dataframe.columns],
            dataset_name=dataset_name,
        )

    def get_dataframe(
        self,
        dataset_name: StructuredDatasetName = "customer_portfolio",
    ) -> pd.DataFrame | None:
        cached = self._dataframes.get(dataset_name)
        if cached is not None:
            return cached.copy()

        if dataset_name in self._load_errors:
            return None

        csv_path = self._get_csv_path(dataset_name)
        if csv_path is None:
            return None

        try:
            self._dataframes[dataset_name] = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            self._load_errors[dataset_name] = str(exc)
            return None

        return self._dataframes[dataset_name].copy()

    def get_dataset_file_name(
        self,
        dataset_name: StructuredDatasetName = "customer_portfolio",
    ) -> str | None:
        csv_path = self._get_csv_path(dataset_name)
        if csv_path is None:
            return None
        return csv_path.name

    def get_load_error(
        self,
        dataset_name: StructuredDatasetName = "customer_portfolio",
    ) -> str | None:
        return self._load_errors.get(dataset_name)

    def _get_csv_path(self, dataset_name: StructuredDatasetName) -> Path | None:
        if not self.structured_data_path.exists() or not self.structured_data_path.is_dir():
            return None

        file_name = self.DATASET_FILES.get(dataset_name)
        if file_name is None:
            return None

        csv_path = self.structured_data_path / file_name
        if not csv_path.exists() or not csv_path.is_file():
            return None
        return csv_path
