from df_generator_from_csvs import DfGeneratorFromCSVs
from preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data
from run_classification import prepare_undersampled_split, run_classifiers
from KaggleDatasetProvider import KaggleDatasetProvider
from dnn.neural_network import NeuralNetwork

from dnn.trainer import *
from visualizedata.visualize import  *



if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    preprocessor = Preprocessor(df_synthetic=df_synthetic, df_original=df)
    undersampled_dataset_splits = preprocessor.split_undersampling()
    synthetic_dataset_splits = preprocessor.split_with_synthetic()


    visualize_data(df)
    show_correlationMatrix(df, undersampled_dataset_splits.X_train, synthetic_dataset_splits.X_train)

    run_classifiers(X_train=undersampled_dataset_splits.X_train,
                    y_train=undersampled_dataset_splits.y_train,
                    X_test=undersampled_dataset_splits.X_test,
                    y_test=undersampled_dataset_splits.y_test,
                    name=undersampled_dataset_splits.name)

    run_classifiers(X_test=synthetic_dataset_splits.X_test,
                    X_train=synthetic_dataset_splits.X_train,
                    y_test=synthetic_dataset_splits.y_test,
                    y_train=synthetic_dataset_splits.y_train,
                    name=synthetic_dataset_splits.name)


    num_epochs = 200
    model, loss_hist, precision, recall, f1_score, predictions_undersamp = run_training(ds=undersampled_dataset_splits, epochs=num_epochs)
    showConfusionMatrix(undersampled_dataset_splits.y_test, predictions_undersamp, "DNN with Undersampled Data")

    #model, loss_hist, precision, recall, f1_score, predictions_synthetic = run_training(ds=synthetic_dataset_splits, epochs=num_epochs)
    #showConfusionMatrix(synthetic_dataset_splits.y_test, predictions_synthetic, "DNN with Synthetic Data")
