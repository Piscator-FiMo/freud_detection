import logging
import os

import pandas as pd
LOGGER = logging.getLogger(__name__)

class DfGeneratorFromCSVs:
    def generate_df_from_csvs(self, dir_path: str) -> pd.DataFrame:
        csv_files = [file for file in os.listdir(dir_path) if file.endswith('.csv')]
        dataframes = [pd.read_csv(os.path.join(dir_path, file)) for file in csv_files]
        df_concat = pd.concat(dataframes, ignore_index=True)
        df_numeric_columns = self.ensure_numeric_dtypes(df_concat)
        LOGGER.warning("Removing %i rows containing NaN.", df_numeric_columns.isna().any(axis=1).sum())
        return df_numeric_columns.dropna()

    def ensure_numeric_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                LOGGER.warning(f"Column '{column}' is not numeric. Converting to numeric.")
                df[column] = pd.to_numeric(df[column], errors='coerce')
        return df
