import json
import time
from src import Translation


def translate_commands(input_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            commands = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' Not Found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        return

    if not isinstance(commands, list):
        print(f"Error: Input Should be List, Get {type(commands)}")
        return
    if not all(isinstance(item, str) for item in commands):
        print("Error: Elements Should be String.")
        return

    service = Translation(
        load_type='local',
        model_path='models/Qwen2.5-3B-Instruct',
        src_lang='eng_Latn',
        tgt_lang='zho_Hans',
        num_beams=4,
        temperature=0.0,
    )

    print(f"Start Translating {len(commands)} commands...")
    start = time.perf_counter()

    translations = service.translate(commands)

    elapsed = time.perf_counter() - start
    print(f"Translation Finished, used {elapsed:.3f} seconds.")

    result = []
    for orig, trans in zip(commands, translations):
        result.append({
            "original": orig,
            "translation": trans
        })

    output_file = "data/translated_commands.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved at {output_file}")

    for i, item in enumerate(result[:3]):
        print(f"{i+1}. {item['original']}\n   -> {item['translation']}\n")


if __name__ == "__main__":
    translate_commands("data/commands.json")
