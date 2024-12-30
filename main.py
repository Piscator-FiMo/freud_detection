from df_generator_from_csvs import DfGeneratorFromCSVs
from preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data
from run_classification import prepare_undersampled_split, run_classifiers
from KaggleDatasetProvider import KaggleDatasetProvider

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    X_train, X_test, y_train, y_test = Preprocessor(df_synthetic=df_synthetic, df_original=df).split_undersampling()
    run_classifiers(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
