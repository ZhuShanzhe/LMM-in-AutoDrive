import json
from typing import List, Dict, Any, Optional


def load_test_json(
    json_file: str,
    ref_key: str = "translation",
    audio_key: str = "audio_file",
    id_key: str = "index"
) -> List[Dict[str, str]]:
    """
    Load a test dataset JSON file.

    Expected format:
    [
        {"index": 1, "original": "...", "translation": "...", "audio_file": "..."},
        ...
    ]
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Test JSON must be a list of objects.")

    result = []
    for item in data:
        audio = item.get(audio_key, "")
        ref = item.get(ref_key, "")
        uid = str(item.get(id_key, len(result) + 1))
        result.append({
            "id": uid,
            "reference": ref,
            "audio_file": audio,
            "original": item.get("original", ""),
            "raw": item
        })
    return result


def save_results_to_json(results: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_file}")

