import logging
import os
from pathlib import Path

import kagglehub
import pandas as pd

LOGGER = logging.getLogger(__name__)

class KaggleDatasetProvider:

    def __init__(self):
        self.df = None

    def fetch_data(self) -> pd.DataFrame:
        if self.df is None:
            path = str(Path.home()) + "/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3"
            if not os.path.isfile(path + "/creditcard.csv"):
                # Download data
                print("downloading creditcard.csv")
                path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
                print("Path to dataset files:", path)

            # Load data
            self.df = pd.read_csv(path + "/creditcard.csv")
        return self.df.copy()

