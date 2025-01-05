from dataclasses import dataclass

from pandas import Series, DataFrame


@dataclass
class DatasetSplits:
    def __init__(self, y_train: Series, y_test: Series, X_train: DataFrame, X_test: DataFrame, name: str,
                 X_val: DataFrame = None, y_val: Series = None):
        self.y_val = y_val
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.name = name
