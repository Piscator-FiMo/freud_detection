from remove_test_data_from_train_data import remove_test_data_from_train_data
from df_generator_from_csvs import generate_df_from_csvs
from run_classification import prepare_undersampled_split, run_classifiers
from KaggleDatasetProvider import KaggleDatasetProvider

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    X_test, X_train, y_test, y_train = prepare_undersampled_split(df)
    run_classifiers(X_test, X_train, y_test, y_train)
