import torch
from torch import nn

from KaggleDatasetProvider import KaggleDatasetProvider
from df_generator_from_csvs import DfGeneratorFromCSVs
from neural_network import NeuralNetwork
from preprocessing.dataset_splits import DatasetSplits
from preprocessing.preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data


def train_loop(model: NeuralNetwork, optimizer) -> list:
    size = len(model.train_loader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    loss_history = []
    for batch, (X, y) in enumerate(model.train_loader):
        # Compute prediction and loss
        pred = model.model(X)
        # weights = create_weighted_bceloss(y, 1, 3)
        loss_fn = nn.CrossEntropyLoss()
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


def test_loop(model: NeuralNetwork):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(model.test_loader.dataset)
    num_batches = len(model.test_loader)
    test_loss, correct, precision, recall, f1_score = 0, 0, 0, 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in model.test_loader:
            pred = model.model(X)
            # weights = create_weighted_bceloss(y, 1, 3)
            loss_fn = nn.CrossEntropyLoss()
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            predicted_positive = (pred.argmax(1) == 1).type(
                torch.float)  # Predictions for the positive class (mountains)
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
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n, Avg Precision: {(100 * precision):0.1f}%, Avg Recall {(100 * recall):0.1f}%, Avg F1 Score {(f1_score)}")
    return precision, recall, f1_score


def run_training(ds: DatasetSplits, precision_threshold=None, f1_score_threshold=None,
                 recall_threshold=None) -> tuple:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork(ds).to(device)
    learning_rate = 1e-3
    # plot varianz der loss functions über die learning rates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 20
    all_loss_history = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        epoch_loss_history = train_loop(model, optimizer)
        all_loss_history.extend(epoch_loss_history)
        precision_val, recall, f1_score = test_loop(model)
        if precision_threshold and precision_val > precision_threshold:
            print("Early stopping for precision triggered")
            return model, all_loss_history, precision_val, recall, f1_score
        if f1_score_threshold and f1_score > f1_score_threshold:
            print("Early stopping for f1_score triggered")
            return model, all_loss_history, precision_val, recall, f1_score
        if recall_threshold and recall > recall_threshold:
            print("Early stopping for recall triggered")
            return model, all_loss_history, precision_val, recall, f1_score
    print("Done!")
    return (model, all_loss_history, precision_val, recall, f1_score)

def count_values(df, column_name, target_value):
    """
    Counts the occurrences of a specific value in a given column of a Pandas DataFrame.

    Args:
        df: Pandas DataFrame.
        column_name: Name of the column to analyze.
        target_value: The value to count occurrences of.

    Returns:
        int: The number of times the target_value appears in the specified column.
    """

    # Use boolean indexing and sum to count occurrences:
    count = (df[column_name] == target_value).sum()
    return count

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    preprocessor = Preprocessor(df_synthetic=df_synthetic, df_original=df)
    synthetic_dataset_splits = preprocessor.split_with_synthetic()
    model, loss_hist, precision, recall, f1_score = run_training(ds=synthetic_dataset_splits)





    non_fraud_count = count_values(synthetic_dataset_splits.y_train, 'Class', 0)
    fraud_count = count_values(synthetic_dataset_splits.y_train, 'Class', 1)
    total_count = non_fraud_count + fraud_count
    print(f"Non Fraud: {non_fraud_count} ({(non_fraud_count / total_count) * 100:.2f}%)")
    print(f"Fraud: {fraud_count} ({(fraud_count / total_count) * 100:.2f}%)")
