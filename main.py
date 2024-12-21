import kagglehub
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initializing OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# MODEL = 'o1-preview'
MODEL = 'gpt-4o-mini'

# Download data
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)

# Load data
df = pd.read_csv(path + "/creditcard.csv")
df.info()

# How many good/fraudulent transactions are there?
print(df['Class'].value_counts())

# Take out the fraudulent transactions
df_fraud = df[df['Class'] == 1]
df_splits = np.array_split(df_fraud, 6)

# print(df_fraud.sample(frac=.75, random_state=1).to_csv(index=False))
# print(pd.DataFrame(df_splits[0]).to_csv(index=False))


# https://cookbook.openai.com/examples/sdg1
# ToDo: better prompt!
def generate_data(df: pd.DataFrame):
    question = f"""
    Here are some examples of fraudulent transactions. Please generate synthetic data that looks similar to this data.

    {df.to_csv(index=False)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    generate_data(pd.DataFrame(df_splits[0]))
