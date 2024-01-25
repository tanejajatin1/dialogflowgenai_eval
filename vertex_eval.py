from datasets import Dataset
from ragas.metrics import context_precision, faithfulness, answer_similarity, answer_relevancy, context_recall
from ragas import evaluate
import pandas as pd
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

import google.auth
from langchain.chat_models import ChatVertexAI
from ragas.llms import LangchainLLM
from langchain.embeddings import VertexAIEmbeddings


config = {
    "project_id": os.getenv("project_id"),
    "location_id": os.getenv("location_id")
}

# authenticate to GCP
creds, _ = google.auth.default(quota_project_id= os.getenv("project_id"))
# create Langchain LLM and Embeddings
chat = ChatVertexAI(credentials=creds)
vertextai_embeddings = VertexAIEmbeddings(credentials=creds)

# create a wrapper around it
ragas_vertexai_llm = LangchainLLM(chat)

# list of metrics we're going to use
metrics = [
    faithfulness,
    answer_similarity,
    answer_relevancy,
    context_precision,
    context_recall
]

for m in metrics:
    # change LLM for metric
    m.__setattr__("llm", ragas_vertexai_llm)

    # check if this metric needs embeddings
    if hasattr(m, "embeddings"):
        # if so change with VertexAI Embeddings
        m.__setattr__("embeddings", vertextai_embeddings)

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
    metrics= metrics
)

df = result.to_pandas()
# Save the DataFrame to a CSV file
df.to_csv('evaluation.csv', index=False)
