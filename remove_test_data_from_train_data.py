import logging

import pandas as pd
LOGGER = logging.getLogger(__name__)


def remove_test_data_from_train_data(df_synthetic: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    df_clean = (pd.merge(df_synthetic, df_test, how='outer', indicator=True)
            .query('_merge=="left_only"')
            .drop(columns=['_merge'], axis=1))
    LOGGER.info("Removed %i rows since they were in test data", len(df_synthetic.index) - len(df_clean.index))
    return df_clean

