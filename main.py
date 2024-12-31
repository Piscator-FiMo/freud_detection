from KaggleDatasetProvider import KaggleDatasetProvider
from df_generator_from_csvs import DfGeneratorFromCSVs
from preprocessing.preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data
from run_classification import run_classifiers

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    preprocessor = Preprocessor(df_synthetic=df_synthetic, df_original=df)

    undersampled_dataset_splits = preprocessor.split_undersampling()

    run_classifiers(X_train=undersampled_dataset_splits.X_train,
                    y_train=undersampled_dataset_splits.y_train,
                    X_test=undersampled_dataset_splits.X_test,
                    y_test=undersampled_dataset_splits.y_test,
                    name=undersampled_dataset_splits.name)

    synthetic_dataset_splits = preprocessor.split_with_synthetic()
    run_classifiers(X_test=synthetic_dataset_splits.X_test,
                    X_train=synthetic_dataset_splits.X_train,
                    y_test=synthetic_dataset_splits.y_test,
                    y_train=synthetic_dataset_splits.y_train,
                    name=synthetic_dataset_splits.name)
