# evaluate.py
# The evaluation harness. Runs every golden-dataset question through the
# pipeline, then scores the results with RAGAS. Run with: python evaluate.py
#
# Baseline scores (fill in after your first run):
# faithfulness: 0.8571
# answer_relevancy: 0.7619
# context_precision: 0.6486
# context_recall: 0.7750
# Date: 10/09/26
#
# New aggregate scores (with TOP_K set to 1):
# faithfulness: 0.4999
# answer_relevancy: 0.4332
# context_precision: 0.5500
# context_recall: 0.400
# Date: 10/09/26

import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from pipeline import ask, load_vector_store
from evaluation.ragas_config import get_ragas_llm, get_ragas_embeddings
from graph_agent import ask_agent, extract_contexts_from_messages

DATASET_PATH = Path("evaluation/golden_dataset.json")
OUTPUT_PATH  = Path("evaluation/pipeline_outputs.json")

def run_with_agent(sample):
    result = ask_agent(sample["question"])
    return {
        "answer": result["answer"],
        "contexts": extract_contexts_from_messages(result["messages"])
    }

def collect_outputs(vector_store) -> list[dict]:
    """Run every question in the golden dataset through the pipeline.

    Read DATASET_PATH. For each item, call ask() and collect the result.
    Write a list of dicts to OUTPUT_PATH. Each dict must have:
        "question":     str
        "ground_truth": str  (copied from the golden dataset)
        "answer":       str  (from the pipeline)
        "contexts":     list[str]  (from the pipeline)

    Print progress as you go.
    Returns the list of output dicts.
    """
    with open(DATASET_PATH) as f:
        golden_dataset = json.load(f)

    outputs = []
    total = len(golden_dataset)

    for i, item in enumerate(golden_dataset, start=1):
        question = item["question"]
        print(f"[{i}/{total}] Asking: {question}")

        # result = ask(question, vector_store)

        # outputs.append({
        #     "question": question,
        #     "ground_truth": item["ground_truth"],
        #     "answer": result["answer"],
        #     "contexts": result["contexts"],
        # })
        agent_result = run_with_agent(item)

        outputs.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": agent_result["answer"],
            "contexts": agent_result["contexts"]
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Wrote {len(outputs)} outputs to {OUTPUT_PATH}")
    return outputs
    

def run_ragas(outputs: list[dict]) -> None:
    """Score the collected outputs with RAGAS (given, no changes needed).

    Builds a HuggingFace Dataset from outputs and calls ragas.evaluate() with
    four metrics, using the Vertex AI judge and embeddings from ragas_config.
    Prints the aggregate scores and saves per-query results to CSV.
    """
    data = {
        "question":     [item["question"] for item in outputs],
        "answer":       [item["answer"] for item in outputs],
        "contexts":     [item["contexts"] for item in outputs],
        "ground_truth": [item["ground_truth"] for item in outputs],
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )
    print(result)
    result.to_pandas().to_csv("evaluation/ragas_results.csv", index=False)
    print("Per-query results saved to evaluation/ragas_results.csv")


if __name__ == "__main__":
    vector_store = load_vector_store()
    outputs = collect_outputs(vector_store)
    run_ragas(outputs)
