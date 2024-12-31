import pandas as pd
from sklearn.model_selection import train_test_split

from run_classification import scale_data


class DatasetSplits:
    def __init__(self, y_train, y_test, X_train, X_test, name):
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.name = name

class Preprocessor:
    def __init__(self, df_synthetic:pd.DataFrame, df_original):
        self.df_synthetic = df_synthetic
        self.df_original = df_original

    def preprocess(self):
        pass

    def split_with_synthetic(self) -> DatasetSplits:
        df_synthetic = self.df_synthetic.copy()
        df_original = self.df_original.copy()

        # Remove the first 50 entries in the df, because they were used for prompt
        df_fraud_original = df_original[df_original['Class'] == 1]
        df_regular_original = df_original[df_original['Class'] == 0]
        df_fraud_for_testing = df_fraud_original.iloc[50:, :]
        df_regular_for_rest = df_regular_original.iloc[50:, :]

        df_regular_for_training = df_regular_for_rest.sample(n=df_synthetic.shape[0], random_state=42)
        #dropping the non-frauds from df_regular_for_rest that will be used for training to avoid leaking
        df_regular_for_testing_candidates = (pd.merge(df_regular_for_rest, df_regular_for_training, how='outer', indicator=True)
                    .query('_merge=="left_only"')
                    .drop(columns=['_merge'], axis=1))

        df_regular_for_testing = df_regular_for_testing_candidates.sample(n=df_fraud_for_testing.shape[0], random_state=42)

        df_for_testing = pd.concat([df_regular_for_testing, df_fraud_for_testing])
        df_for_training = pd.concat([df_regular_for_training, df_synthetic])

        X_test = df_for_testing.drop('Class', axis=1)
        y_test = df_for_testing['Class']
        X_train = df_for_training.drop('Class', axis=1)
        y_train = df_for_training['Class']

        X_train = scale_data(X_train)
        X_test = scale_data(X_test)

        return DatasetSplits(y_train=y_train, y_test=y_test, X_train=X_train, X_test=X_test, name="synthetic training frauds")


    def split_undersampling(self) -> DatasetSplits:
        df = self.df_original.copy()
        # create balanced dataset
        # Undersample Non-Fraud Transactions
        shuffled_df = df.sample(frac=1)
        # amount of fraud classes 492 rows.
        frauds_df = shuffled_df.loc[df['Class'] == 1]
        non_frauds_df = shuffled_df.loc[df['Class'] == 0][:frauds_df.shape[0]]
        balanced_df = pd.concat([frauds_df, non_frauds_df])
        # do train_test_split
        X = balanced_df.drop('Class', axis=1)
        y = balanced_df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = scale_data(X_train)
        X_test = scale_data(X_test)
        return DatasetSplits(y_train=y_train, y_test=y_test, X_train=X_train, X_test=X_test, name="undersampling")