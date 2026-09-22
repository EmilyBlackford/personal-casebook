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

import sys
import argparse
import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from pipeline import ask, load_vector_store
from evaluation.ragas_config import get_ragas_llm, get_ragas_embeddings

from graph_agent import ask_agent, extract_contexts_from_messages


def run_with_agent(sample):
    result = ask_agent(sample["question"])
    return {
        "answer": result["answer"],
        "contexts": extract_contexts_from_messages(result["messages"])
    }


def collect_outputs(dataset_path: Path, output_path: Path, vector_store) -> list[dict]:
    """Run every question in the given dataset through the pipeline.

    Read dataset_path. For each item, call ask() and collect the result.
    Write a list of dicts to output_path. Each dict must have:
        "question":     str
        "ground_truth": str  (copied from the dataset)
        "answer":       str  (from the pipeline)
        "contexts":     list[str]  (from the pipeline)

    Print progress as you go.
    Returns the list of output dicts.
    """
    with dataset_path.open(encoding="utf-8") as f:
        golden = json.load(f)

    outputs = []
    for i, item in enumerate(golden, start=1):
        question = item["question"]
        print(f"[{i}/{len(golden)}] {question}")

        agent_result = run_with_agent(item)

        outputs.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "answer": agent_result["answer"],
            "contexts": agent_result["contexts"]
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(outputs)} outputs to {output_path}")

    return outputs


def run_ragas(outputs: list[dict]):
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
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="Write results to evaluation_results.json")
    parser.add_argument("--dataset", default="evaluation/golden_dataset.json", help="Path to the question set to evaluate")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path("evaluation/pipeline_outputs.json")

    vector_store = load_vector_store()
    outputs = collect_outputs(dataset_path, output_path, vector_store)
    result = run_ragas(outputs)

    df = result.to_pandas()
    results = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }

    if args.ci:
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results written to evaluation_results.json")
    else:
        for metric, score in results.items():
            print(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    main()