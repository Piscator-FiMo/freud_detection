from run_classification import run_classification
from KaggleDatasetProvider import KaggleDatasetProvider

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    run_classification(df)
