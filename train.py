from openai import OpenAI
import json
import time

def wait_for_fine_tuning(openai_client, job_id):
    while True:
        # Retrieve the job details
        job = openai_client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        # Print the current status
        print(f"Fine-tuning job status: {status}")

        if status == 'succeeded':
            print("Fine-tuning completed successfully.")
            break
        elif status == 'failed':
            print("Fine-tuning failed.")
            break
        else:
            time.sleep(20)

if __name__ == '__main__':
    openai_client = OpenAI()
    wnb_integration = {
        "type": "wandb",
        "wandb": {
            "project": "custom-wandb-project",
        }
    }

    file_ids = json.load(open('data/train/file_ids.json'))

    train_file_id = file_ids['train']
    val_file_id = file_ids['val']

    fine_tune = openai_client.fine_tuning.jobs.create(training_file=train_file_id,
                                                      validation_file=val_file_id, model="gpt-4o-mini-2024-07-18",
                                                      integrations=[wnb_integration])

    # train_file_id = file_ids['train_placebo']
    # val_file_id = file_ids['val_placebo']
    #
    # fine_tune_placebo = openai_client.fine_tuning.jobs.create(training_file=train_file_id,
    #                                    validation_file=val_file_id, model="gpt-4o-mini-2024-07-18",
    #                                                  integrations=[wnb_integration])
