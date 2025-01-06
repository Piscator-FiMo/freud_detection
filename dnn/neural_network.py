import datetime
import os
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data as data_utils
from pandas import Series
from torch import nn, Tensor

from preprocessing.dataset_splits import DatasetSplits


class NeuralNetwork(nn.Module):
    def __init__(self, dataset_split: DatasetSplits, normalise: bool = True):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self._prediction_column = 'Class'
        self.batch_size = 64

        self.train_loader = self.transform(dataset_split.X_train, dataset_split.y_train, shuffle=True, normalize=normalise)
        self.test_loader = self.transform(dataset_split.X_test, dataset_split.y_test, shuffle=False, normalize=normalise)


        if dataset_split.X_val is not None and dataset_split.y_val is not None:
            loader = self.transform(dataset_split.X_val, dataset_split.y_val, False, normalize=normalise)
            self.val_loader = loader

        self.n_inputs = dataset_split.X_train.shape[1]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_inputs, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 2).double(),
            nn.Softmax(dim=1)
        )

    def transform(self, data: pd.DataFrame, target: Series, shuffle: bool, normalize: bool = True):
        if normalize:
            data_val = self.normalize_numerical(data)
        else:
            data_val = data
        val_tensor = torch.tensor(data_val.values)
        val_target_tensor = torch.tensor(target.values)
        val_ds = data_utils.TensorDataset(val_tensor, val_target_tensor)
        return data_utils.DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=shuffle)

    def forward(self, x):
        x = self.flatten(x)

        return self.linear_relu_stack(x)

    def create_weighted_bceloss(self, target_tensor, target_value, weight_factor):
        # Create a weight tensor with all ones initially
        weights = torch.ones_like(target_tensor, dtype=torch.float)

        # Find the indices where the target value is present
        target_indices = target_tensor == target_value

        # Apply the weight factor to those indices
        weights[target_indices] = weight_factor

        return weights

    def count_values(self, target_tensor: Tensor, target_value: float) -> Tensor:
        # Use boolean indexing and sum to count occurrences:
        count = (target_tensor == target_value).sum()
        return count

    def normalize_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [column for column in df.select_dtypes(include=[float]).columns if 'scaled' not in column ]
        for column in numeric_columns:
            df = self.min_max_normalize(df, column)
        return df

    def min_max_normalize(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # Calculate the minimum and maximum values of the column
        column_min = df[column_name].min()
        column_max = df[column_name].max()

        # Apply the min-max normalization formula
        df[column_name] = (df[column_name] - column_min) / (column_max - column_min)
        return df

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.pt"

    def save_model(self, name: str):
        dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
        if not name:
            name = self._default_save_path()
        torch.save(self.state_dict(), dir_path.joinpath(f'models/{name}.pt'))
