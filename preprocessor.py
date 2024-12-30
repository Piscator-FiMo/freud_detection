import pandas as pd
from sklearn.model_selection import train_test_split

from run_classification import traintest_split, scale_data


class Preprocessor:
    def __init__(self, df_synthetic:pd.DataFrame, df_original):
        self.df_synthetic = df_synthetic
        self.df_original = df_original

    def preprocess(self):
        pass

    def split_with_synthetic(self):
        # Remove the first 50 entries in the df
        pass

    def split_undersampling(self):
        df = self.df_original.copy()
        # create balanced dataset
        # Undersample Non-Fraud Transactions
        shuffled_df = df.sample(frac=1)
        # amount of fraud classes 492 rows.
        frauds_df = shuffled_df.loc[df['Class'] == 1]
        non_frauds_df = shuffled_df.loc[df['Class'] == 0][:492]
        balanced_df = pd.concat([frauds_df, non_frauds_df])
        # do train_test_split
        X = balanced_df.drop('Class', axis=1)
        y = balanced_df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = scale_data(X_train)
        X_test = scale_data(X_test)
        return X_train.values, X_test.values, y_train.values, y_test.values