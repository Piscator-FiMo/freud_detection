from run_classification import prepare_undersampled_split, run_classifiers
from KaggleDatasetProvider import KaggleDatasetProvider

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    X_test, X_train, y_test, y_train = prepare_undersampled_split(df)
    run_classifiers(X_test, X_train, y_test, y_train)
