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
#MODEL = 'o1-preview'
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

df_fraud_for_generation = df_fraud.iloc[:50,:]
df_fraud_for_testing = df_fraud.iloc[50:,:]

df_regular_for_generation =  df_regular.iloc[:50,:]
df_regular_for_rest =  df_regular.iloc[50:,:]

df_for_generation = pd.concat([df_fraud_for_generation, df_regular_for_generation])


# https://cookbook.openai.com/examples/sdg1
def generate_data(df: pd.DataFrame):
    question = f"""
    Please generate a CSV file containing 50 rows of fraudulent transactions. The provided CSV includes both fraudulent (Class column is 1) and non-fraudulent (Class column is 0) transactions. Ensure the new CSV has the same column names as the example CSV below, with exactly 31 columns in each row. The new data should represent fraudulent transactions, where the Class column is 1    Example CSV:

    {df.to_csv(index=False)}
    
    Respond only with the new CSV file.
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
        data = data.split("\n",1)[1]
        # Read the data into a pandas DataFrame
        df_generated = pd.read_csv(StringIO(data))
        dropped = df_generated.dropna()
        # Save the DataFrame to a CSV file
        df_generated.to_csv(f"data/synthetic_data_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv", index=False)
