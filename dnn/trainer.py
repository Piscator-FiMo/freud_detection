import os.path
from typing import Tuple, Any, List

import torch
from pandas import Series
from sklearn import metrics
from sympy import false
from torch import nn, Tensor

from KaggleDatasetProvider import KaggleDatasetProvider
from df_generator_from_csvs import DfGeneratorFromCSVs
from dnn.fusion_network import FusionNetwork
from dnn.neural_network import NeuralNetwork
from preprocessing.dataset_splits import DatasetSplits
from preprocessing.preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data

from visualizedata.visualize import *


class EarlyStopper:
    def __init__(self, patience=5, min_delta=10):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_loop(model: NeuralNetwork, optimizer, device: str) -> list:
    size = len(model.train_loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    loss_history = []
    for batch, (X, y) in enumerate(model.train_loader):
        # Compute prediction and loss
        data = X.to(device)
        pred = model(data)
        # weights = create_weighted_bceloss(y, 1, 3)
        loss_fn = nn.CrossEntropyLoss()
        y = y.to(device)
        loss = loss_fn(pred, y.long())
        # Backpropagation
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * model.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss_history


def test_loop(model: NeuralNetwork, device: str) -> tuple[int | Any, int | Any, int | Any, list[Any]]:
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(model.test_loader.dataset)
    num_batches = len(model.test_loader)
    test_loss, correct, precision, recall, f1_score = 0, 0, 0, 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    predictions = []
    with torch.no_grad():
        for X, y in model.test_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            predictions.extend(pred.argmax(1).tolist())
            # weights = create_weighted_bceloss(y, 1, 3)
            loss_fn = nn.CrossEntropyLoss()
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            predicted_positive = (pred.argmax(1) == 1).type(
                torch.float)  # Predictions for the positive class (Fraud)
            actual_positive = (y == 1).type(torch.float)  # Actual positive labels
            true_positives = (
                    predicted_positive * actual_positive).sum().item()  # Count of correctly predicted positives
            false_positives = (predicted_positive * (
                    1 - actual_positive)).sum().item()  # Count of incorrectly predicted positives
            false_negatives = ((
                                       1 - predicted_positive) * actual_positive).sum().item()  # Count of incorrectly predicted negatives
            batch_precision = true_positives / (true_positives + false_positives) if (
                                                                                             true_positives + false_positives) > 0 else 0  # Avoid division by zero
            batch_recall = true_positives / (true_positives + false_negatives) if (
                                                                                          true_positives + false_negatives) > 0 else 0  # Calculate recall for this batch

            precision += batch_precision
            recall += batch_recall
            f1_score += 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if (
                                                                                                           batch_precision + batch_recall) > 0 else 0

    test_loss /= num_batches
    correct /= size
    precision /= num_batches
    recall /= num_batches
    f1_score /= num_batches
    print(
        f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}, Avg Precision: {(100 * precision):0.1f}%, Avg Recall {(100 * recall):0.1f}%, Avg F1 Score {f1_score}")
    return precision, recall, f1_score, predictions


def validate(model: NeuralNetwork, device: str):
    model.eval()
    size = len(model.val_loader.dataset)
    num_batches = len(model.val_loader)
    test_loss, correct, precision, recall, f1_score = 0, 0, 0, 0, 0
    predictions = []
    with torch.no_grad():
        for X, y in model.val_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            predictions.extend(pred.argmax(1).tolist())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            predicted_positive = (pred.argmax(1) == 1).type(
                torch.float)  # Predictions for the positive class (Fraud)
            actual_positive = (y == 1).type(torch.float)  # Actual positive labels
            true_positives = (
                    predicted_positive * actual_positive).sum().item()  # Count of correctly predicted positives
            false_positives = (predicted_positive * (
                    1 - actual_positive)).sum().item()  # Count of incorrectly predicted positives
            false_negatives = ((
                                       1 - predicted_positive) * actual_positive).sum().item()  # Count of incorrectly predicted negatives
            batch_precision = true_positives / (true_positives + false_positives) if (
                                                                                             true_positives + false_positives) > 0 else 0  # Avoid division by zero
            batch_recall = true_positives / (true_positives + false_negatives) if (
                                                                                          true_positives + false_negatives) > 0 else 0  # Calculate recall for this batch

            precision += batch_precision
            recall += batch_recall
            f1_score += 2 * (batch_precision * batch_recall) / (batch_precision + batch_recall) if (
                                                                                                           batch_precision + batch_recall) > 0 else 0

        correct /= size
        precision /= num_batches
        recall /= num_batches
        f1_score /= num_batches
        print(
            f"Validation Error: Accuracy: {(100 * correct):>0.1f}%, Avg Precision: {(100 * precision):0.1f}%, Avg Recall {(100 * recall):0.1f}%, Avg F1 Score {f1_score}")
        return precision, recall, f1_score, predictions


def run_training(ds: DatasetSplits, name: str, model_class, epochs=20, precision_threshold=None,
                 f1_score_threshold=None,
                 recall_threshold=None) -> tuple:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    fraud_detection_model = model_class(ds).to(device)
    learning_rate = 1e-3
    # plot varianz der loss functions über die learning rates
    optimizer = torch.optim.Adam(fraud_detection_model.parameters(), lr=learning_rate)
    # epochs = 200
    all_loss_history = []
    saved = false
    old_f1_score = 0
    early_stopper = EarlyStopper()
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_loss_history = train_loop(fraud_detection_model, optimizer, device)
        all_loss_history.extend(epoch_loss_history)
        precision, recall, f1_score, predictions = test_loop(fraud_detection_model, device)
        if not saved or f1_score > old_f1_score:
            fraud_detection_model.save_model(name)
            saved = True
        old_f1_score = f1_score
        if early_stopper.early_stop(epoch_loss_history[-1]):
            print("Early stopping because of validation loss")
            return fraud_detection_model, all_loss_history, precision, recall, f1_score, predictions
        if precision_threshold and precision > precision_threshold:
            print("Early stopping for precision triggered")
            return fraud_detection_model, all_loss_history, precision, recall, f1_score, predictions
        if f1_score_threshold and f1_score > f1_score_threshold:
            print("Early stopping for f1_score triggered")
            return fraud_detection_model, all_loss_history, precision, recall, f1_score, predictions
        if recall_threshold and recall > recall_threshold:
            print("Early stopping for recall triggered")
            return fraud_detection_model, all_loss_history, precision, recall, f1_score, predictions
    print("Done!")
    return fraud_detection_model, all_loss_history, precision, recall, f1_score, predictions

def plot_roc_curve(target: Series, pred):
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=2)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('../data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    preprocessor = Preprocessor(df_synthetic=df_synthetic, df_original=df)
    undersampled_dataset_splits = preprocessor.split_undersampling()
    synthetic_dataset_splits = preprocessor.split_with_synthetic()
    df_synthetic_o1 = DfGeneratorFromCSVs().generate_df_from_csvs('../data_o1')
    df_synthetic_o1 = remove_test_data_from_train_data(df_synthetic_o1, df)
    o1_preprocessor = Preprocessor(df_synthetic=df_synthetic_o1, df_original=df)
    o1_dataset_splits = o1_preprocessor.split_with_synthetic()

    num_epochs = 200
    model, loss_hist, precision, recall, f1_score, predictions_fusion = run_training(ds=synthetic_dataset_splits,
                                                                                     model_class=FusionNetwork,
                                                                                     name='fusion_synthetic',
                                                                                     epochs=num_epochs)
    showConfusionMatrix(synthetic_dataset_splits.y_test, predictions_fusion,
                        "DNN with synthetic Data and Fusion Model")
    plot_roc_curve(synthetic_dataset_splits.y_test, predictions_fusion)
    model, loss_hist, precision, recall, f1_score, predictions_fusion = run_training(ds=undersampled_dataset_splits,
                                                                                     model_class=FusionNetwork,
                                                                                     name='fusion_undersampled',
                                                                                     epochs=num_epochs)
    showConfusionMatrix(undersampled_dataset_splits.y_test, predictions_fusion,
                        "DNN with Undersampled Data and Fusion Model")
    precision_val, recall_val, f1_score_val, predictions_val = validate(model, device="cuda")
    showConfusionMatrix(undersampled_dataset_splits.y_val, predictions_val,
                        "DNN with Undersampled Data Validation for fusion Model")

    model, loss_hist, precision, recall, f1_score, predictions_undersamp = run_training(ds=undersampled_dataset_splits,
                                                                                        name='undersampled',
                                                                                        model_class=NeuralNetwork,
                                                                                        epochs=num_epochs)
    showConfusionMatrix(undersampled_dataset_splits.y_test, predictions_undersamp, "DNN with Undersampled Data")
    precision_val, recall_val, f1_score_val, predictions_val = validate(model, device="cuda")

    showConfusionMatrix(undersampled_dataset_splits.y_val, predictions_val, "DNN with Undersampled Data Validation")

    model, loss_hist, precision, recall, f1_score, predictions_o1 = run_training(ds=o1_dataset_splits,
                                                                                 name='o1_synthetic',
                                                                                 model_class=NeuralNetwork,
                                                                                 epochs=num_epochs)
    showConfusionMatrix(o1_dataset_splits.y_test, predictions_o1, "DNN with Synthetic Data by o1")

    model, loss_hist, precision, recall, f1_score, predictions_synthetic = run_training(ds=synthetic_dataset_splits,
                                                                                        name='synthetic',
                                                                                        model_class=NeuralNetwork,
                                                                                        epochs=num_epochs)
    showConfusionMatrix(synthetic_dataset_splits.y_test, predictions_synthetic, "DNN with Synthetic Data")
