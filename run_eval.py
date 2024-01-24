from datasets import Dataset
from ragas.metrics import context_precision, faithfulness, answer_similarity, answer_correctness
from ragas import evaluate
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# import os

# os.environ["OPENAI_API_KEY"] = ""

def read_and_transform_csv(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    print(df.head(5))

    # Transform the DataFrame into the format expected by the function
    dataset_dict = {
        "question": df['question'].tolist(),
        "answer": df['answer'].tolist(),
        "contexts": df['contexts'].apply(eval).tolist(),
        "ground_truths": df['ground_truths'].apply(eval).tolist()  # Assuming contexts are stored as strings in the CSV
    }
    # Create a Hugging Face Dataset from the dictionary
    dataset = Dataset.from_dict(dataset_dict)

    return dataset

# correctness = context_precision.score(read_and_transform_csv("tests.csv"))
# print(correctness)

result = evaluate(
    dataset= read_and_transform_csv("tests.csv"),
    metrics=[
        context_precision,
        faithfulness,
        #answer_similarity,
        #answer_correctness,
    ],
)

df = result.to_pandas()
# Save the DataFrame to a CSV file
df.to_csv('evaluation.csv', index=False)
