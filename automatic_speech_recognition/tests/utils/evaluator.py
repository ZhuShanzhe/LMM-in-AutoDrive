from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

from .metrics import evaluate_pair
from .data_loader import load_test_json, save_results_to_json


class ASREvaluator:
    """
    Evaluator for speech recognition accuracy.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        self.tokenizer = tokenizer
        self.results = None

    def evaluate_from_json(self, json_file: str, output_json: Optional[str] = None) -> Dict[str, Any]:
        data = load_test_json(json_file)
        return self.evaluate(data, output_json)

    def evaluate(self, data: List[Dict[str, str]], output_json: Optional[str] = None) -> Dict[str, Any]:
        references = [item['reference'] for item in data]
        hypotheses = [item['hypothesis'] for item in data]
        return self.evaluate_lists(references, hypotheses, output_json)

    def evaluate_lists(self, references: List[str], hypotheses: List[str],
                       output_json: Optional[str] = None) -> Dict[str, Any]:
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match.")

        per_sample = []
        total_cer = 0.0
        total_wer = 0.0
        total_acc = 0
        n = len(references)

        for ref, hyp in zip(references, hypotheses):
            metrics = evaluate_pair(ref, hyp, self.tokenizer)
            per_sample.append({
                "reference": ref,
                "hypothesis": hyp,
                "metrics": metrics
            })
            total_cer += metrics['cer']
            total_wer += metrics['wer']
            if metrics['sentence_accuracy']:
                total_acc += 1

        overall = {
            "total_samples": n,
            "average_cer": total_cer / n if n else 0.0,
            "average_wer": total_wer / n if n else 0.0,
            "sentence_accuracy_rate": total_acc / n if n else 0.0,
        }

        result = {
            "overall": overall,
            "per_sample": per_sample,
        }

        self.results = result

        if output_json:
            save_results_to_json(result, output_json)

        return result

    def print_summary(self) -> None:
        if self.results is None:
            print("No results available. Run evaluate() first.")
            return
        overall = self.results['overall']
        print("=" * 50)
        print("ASR Evaluation Summary")
        print("=" * 50)
        print(f"Total samples: {overall['total_samples']}")
        print(f"Average CER: {overall['average_cer']:.4f}")
        print(f"Average WER: {overall['average_wer']:.4f}")
        print(f"Sentence Accuracy: {overall['sentence_accuracy_rate']:.2%}")
        print("=" * 50)
