from df_generator_from_csvs import DfGeneratorFromCSVs
from preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data
from run_classification import prepare_undersampled_split, run_classifiers
from KaggleDatasetProvider import KaggleDatasetProvider

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    preprocessor = Preprocessor(df_synthetic=df_synthetic, df_original=df)

    run_classifiers(preprocessor.split_undersampling())
    run_classifiers(preprocessor.split_with_synthetic())
