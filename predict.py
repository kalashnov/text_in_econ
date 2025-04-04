from openai import OpenAI
import json
import numpy as np
from tqdm import tqdm
import time

def predict_treatment(messages, model_name, openai_client):
    completion = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        logprobs=True,
        stop=["\n"],
        temperature=0,
        max_completion_tokens=1,
        top_logprobs=4
    )

    top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
    probs = {logprob.token: round(np.exp(logprob.logprob), 3) for logprob in top_logprobs}
    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict treatment using OpenAI API')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for prediction')
    args = parser.parse_args()
    model_name = args.model_id

    openai_client = OpenAI()

    resulting_data = []
    with open("data/train/val.jsonl", "r") as f:
        for row in tqdm(f):
            result_row = {}
            example = json.loads(row)
            messages = example['messages'][0:2]
            result_row['Advice'] = example['messages'][1]['content']
            result_row['Prediction'] = predict_treatment(messages, model_name, openai_client)
            result_row['Treatment'] = example['messages'][2]['content']
            resulting_data.append(result_row)
            time.sleep(20)

    json.dump(resulting_data, open('data/train/val_predictions_placebo.json', 'w'))