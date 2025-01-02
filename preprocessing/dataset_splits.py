from dataclasses import dataclass


@dataclass
class DatasetSplits:
    def __init__(self, y_train, y_test, X_train, X_test, name):
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.name = name