import datetime
import os
from io import StringIO
import logging

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from KaggleDatasetProvider import KaggleDatasetProvider

LOGGER = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.model = "gpt-4o"

    def generate_data(self, df_for_sample_data: pd.DataFrame):
        question = f"""
    Please generate 100 rows of new fraudulent transactions data. The provided CSV includes both fraudulent (Class column is 1) and non-fraudulent (Class column is 0) transactions. It contains only numerical input variables which are the result of a PCA transformation with the exception of the 'Time' and 'Amount' columns. Ensure the new data has the same columns as the example CSV below, but with new values for each column. The new data should represent fraudulent transactions, where the Class column is 1. Each row should have exactly 31 columns, with the same data types as the example data. 

    Example CSV:
        {df_for_sample_data.to_csv(index=False)}

        Respond only with the data.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    df = KaggleDatasetProvider().fetch_data()
    # How many good/fraudulent transactions are there?
    print(df['Class'].value_counts())

    # Take out the fraudulent transactions
    df_fraud = df[df['Class'] == 1]
    df_regular = df[df['Class'] == 0]
    df_fraud_for_generation = df_fraud.iloc[:50, :]
    df_fraud_for_testing = df_fraud.iloc[50:, :]
    df_regular_for_generation = df_regular.iloc[:50, :]
    df_regular_for_rest = df_regular.iloc[50:, :]
    df_for_generation = pd.concat([df_fraud_for_generation, df_regular_for_generation])

    generator = DataGenerator()

    for _ in tqdm(range(10)):
        data = generator.generate_data(pd.DataFrame(df_for_generation))
        data = data.split("\n", 1)[1]
        # Read the data into a pandas DataFrame
        try:
            df_generated = pd.read_csv(StringIO(data))
            df_generated = df_generated.dropna()
            # Save the DataFrame to a CSV file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df_generated.to_csv(f"data/synthetic_data_{timestamp}.csv", index=False)
            LOGGER.info("created file: synthetic_data_{timestamp}.csv")
        except Exception as e:
            LOGGER.error(e)