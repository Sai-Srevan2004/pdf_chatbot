
import requests
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.test_cases import TEST_CASES
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

BASE_URL  = "http://localhost:8000"
SESSION_ID = "SaiSrevan_Resume"
PDF_PATH   = "SaiSrevan_Resume.pdf"


def upload_pdf():
    """Upload test PDF"""
    print(f"Uploading {PDF_PATH}...")
    with open(PDF_PATH, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": (PDF_PATH, f, "application/pdf")}
        )
    result = response.json()
    print(f"✓ Uploaded: {result['chunks_indexed']} chunks")
    return result["session_id"]


def run_questions(session_id: str) -> list[dict]:
    """Run all test questions through RAG"""
    print(f"\nRunning {len(TEST_CASES)} test questions...")
    test_data = []

    for i, case in enumerate(TEST_CASES):
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"session_id": session_id, "question": case["question"]}
        )

        result  = response.json()
        answer  = result["answer"]
        contexts = [s["preview"] for s in result["sources"]]

        test_data.append({
            "question":     case["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": case["ground_truth"]
        })

        print(f"  [{i+1}/{len(TEST_CASES)}] {case['question'][:50]}")
        print(f"         → {answer[:80]}...")

    return test_data


def calculate_scores(test_data: list[dict]) -> dict:
    """Run RAGAS evaluation"""
    print("\nCalculating RAGAS scores...")

    dataset = Dataset.from_list(test_data)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    scores = {
        "faithfulness":     round(float(results["faithfulness"]), 3),
        "answer_relevancy": round(float(results["answer_relevancy"]), 3),
        "context_recall":   round(float(results["context_recall"]), 3),
    }
    scores["overall"] = round(
        sum(scores.values()) / len(scores), 3
    )
    return scores


def save_results(scores: dict, test_data: list[dict]):
    """Save evaluation results"""
    output = {
        "scores":     scores,
        "test_cases": test_data
    }
    with open("evaluation/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to evaluation/eval_results.json")


def main():
    # Upload PDF
    session_id = upload_pdf()

    # Run questions
    test_data = run_questions(session_id)

    # Calculate scores
    scores = calculate_scores(test_data)

    # Print results
    print("\n" + "="*40)
    print("RAGAS EVALUATION RESULTS")
    print("="*40)
    print(f"  Faithfulness:     {scores['faithfulness']}")
    print(f"  Answer Relevancy: {scores['answer_relevancy']}")
    print(f"  Context Recall:   {scores['context_recall']}")
    print(f"  Overall:          {scores['overall']}")
    print("="*40)

    # Save
    save_results(scores, test_data)

    return scores


if __name__ == "__main__":
    main()