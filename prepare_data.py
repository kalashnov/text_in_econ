# This is a sample Python script.
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from openai import OpenAI
from pathlib import Path


def format_example(row, treatment_column='Treatment'):
    return {
        "messages": [
            {"role": "system", "content": "You are a classifier. Given the following advice text, predict whether it is 'Deterministic' or 'Probabilistic'."},
            {"role": "user", "content": f"Advice: {row['Advice'].strip()}"},
            {"role": "assistant", "content": row[treatment_column]},
        ]
    }

def read_data(file_path):
    data = pd.read_stata(file_path)
    data = data[['Treatment', 'Advice', 'id']].drop_duplicates()
    data = data.loc[data['Treatment'].isin(['Probabilistic', 'Deterministic'])]
    return data

def send_to_openai(df, filename, openai_client, treatment_column='Treatment'):
    # Save the DataFrame to a temporary file
    with open(filename, "w") as f:
        for row in train_df.itertuples():
            json.dump(format_example(row._asdict(), treatment_column=treatment_column), f)
            f.write("\n")
    file_path = Path(filename)
    file = openai_client.files.create(
        file=file_path,
        purpose='fine-tune'
    )
    return file.id

if __name__ == '__main__':
    df = read_data('data/116200-V1/data/data_ACP/data_raw.dta')
    df['Advice_len'] = df['Advice'].str.len()
    df['Placebo']  = np.random.choice(['Probabilistic', 'Deterministic'], size=len(df), p=[0.5, 0.5])

    # wandb.ai/
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Treatment"], random_state=42)


    openai_client = OpenAI()
    file_id_train = send_to_openai(train_df, "data/train/train.jsonl", openai_client, treatment_column='Treatment')
    file_id_val = send_to_openai(val_df, "data/train/val.jsonl", openai_client, treatment_column='Treatment')
    file_id_train_placebo = send_to_openai(train_df, "data/train/train_placebo.jsonl", openai_client, treatment_column='Placebo')
    file_id_val_placebo = send_to_openai(val_df, "data/train/val_placebo.jsonl", openai_client, treatment_column='Placebo')


    json.dump(
    {
            'train': file_id_train, 'val': file_id_val,
            'train_placebo': file_id_train_placebo, 'val_placebo': file_id_val_placebo,
        },
        open('data/train/file_ids.json', 'w')
    )