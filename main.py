import datetime
import random

import kagglehub
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

import os
import csv
from io import StringIO

# Load environment variables from .env file
load_dotenv()

# Initializing OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# MODEL = 'o1-preview'
MODEL = 'gpt-4o'

path = str(Path.home()) + "/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3"
if not os.path.isfile(path + "/creditcard.csv"):
    # Download data
    print("downloading creditcard.csv")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)

# Load data
df = pd.read_csv(path + "/creditcard.csv")
df.info()

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


# https://cookbook.openai.com/examples/sdg1
def generate_data(df: pd.DataFrame):
    question = f"""
Please generate 100 rows of new fraudulent transactions data. The provided CSV includes both fraudulent (Class column is 1) and non-fraudulent (Class column is 0) transactions. Ensure the new data has the same columns as the example CSV below, but with new values for each column. The new data should represent fraudulent transactions, where the Class column is 1. Each row should have exactly 31 columns, with the same data types as the example data.

Example CSV:
    {df.to_csv(index=False)}
    
    Respond only with the data.
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    for _ in range(10):
        data = generate_data(pd.DataFrame(df_for_generation))
        data = data.split("\n", 1)[1]
        # Read the data into a pandas DataFrame
        df_generated = pd.read_csv(StringIO(data))
        df_generated = df_generated.dropna()
        # Save the DataFrame to a CSV file
        df_generated.to_csv(f"data/synthetic_data_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv", index=False)
        print("created file")
