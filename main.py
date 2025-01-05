from df_generator_from_csvs import DfGeneratorFromCSVs
from preprocessor import Preprocessor
from remove_test_data_from_train_data import remove_test_data_from_train_data
from run_catboost import CatBoostRunner
from run_classification import prepare_undersampled_split, run_classifiers
from KaggleDatasetProvider import KaggleDatasetProvider

if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    df_synthetic = DfGeneratorFromCSVs().generate_df_from_csvs('data')
    df_synthetic = remove_test_data_from_train_data(df_synthetic, df)
    preprocessor = Preprocessor(df_synthetic=df_synthetic, df_original=df)

    undersampled_dataset_splits = preprocessor.split_undersampling()
    run_classifiers(undersampled_dataset_splits)
    cbr = CatBoostRunner(undersampled_dataset_splits)
    cbr_best_params = cbr.parameter_tuning()
    # cbr_best_params = {'l2_leaf_reg': 1.0, 'learning_rate': 0.14684194639412482}
    cbr.run_catboost(cbr_best_params)

    synthetic_dataset_splits = preprocessor.split_with_synthetic()
    run_classifiers(synthetic_dataset_splits)
    cbr = CatBoostRunner(synthetic_dataset_splits)
    cbr_best_params = cbr.parameter_tuning()
    # cbr_best_params = {'l2_leaf_reg': 4.0, 'learning_rate': 0.044250811423877726}
    cbr.run_catboost(cbr_best_params)
