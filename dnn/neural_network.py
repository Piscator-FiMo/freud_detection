import pandas as pd
import torch
import torch.utils.data as data_utils
from torch import nn, Tensor

from preprocessing.dataset_splits import DatasetSplits


class NeuralNetwork(nn.Module):
    def __init__(self, dataset_split: DatasetSplits):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self._prediction_column = 'Class'
        self.batch_size = 64


        target_tensor = torch.tensor(dataset_split.y_train.values)
        test_target_tensor = torch.tensor(dataset_split.y_test.values)
        train_tensor = torch.tensor(dataset_split.X_train.values)
        test_tensor = torch.tensor(dataset_split.X_test.values)
        train_ds = data_utils.TensorDataset(train_tensor, target_tensor)
        test_ds = data_utils.TensorDataset(test_tensor, test_target_tensor)
        self.train_loader = data_utils.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_loader = data_utils.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=False)
        self.n_inputs = dataset_split.X_train.shape[1]
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.n_inputs, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 32).double(),
            nn.ReLU().double(),
            nn.Linear(32, 2).double(),
            nn.Softmax()
        )

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

    def min_max_normalize(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        # Calculate the minimum and maximum values of the column
        column_min = df[column_name].min()
        column_max = df[column_name].max()

        # Apply the min-max normalization formula
        df[column_name] = (df[column_name] - column_min) / (column_max - column_min)
        return df

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

