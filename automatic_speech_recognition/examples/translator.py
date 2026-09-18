import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Qwen3TranslatorService

def main() -> None:
    translator = Qwen3TranslatorService(model_id_or_path="models/Qwen3-1.7B")

    english = translator.translate_zh_to_en("前方路口请左转")
    print("[single]", english)

    english, meta = translator.translate_zh_to_en("请减速至每小时40公里", return_meta=True)
    print("[meta] translation:", english)
    print("[meta] time_seconds:", meta["time_seconds"])

    commands = ["请靠边停车", "在下一个红绿灯右转", "与前车保持30米安全距离"]
    results = translator.translate_zh_to_en(commands, output_json="outputs/translator_example.json")
    for src, dst in zip(commands, results):
        print(f"[batch] {src}  =>  {dst}")

    cfg_path = ROOT / "configs/translation/qwen3_translator.yaml"
    if cfg_path.exists():
        translator_yaml = Qwen3TranslatorService.from_yaml(str(cfg_path))
        print("[yaml] ", translator_yaml.translate_zh_to_en("请保持直行"))

if __name__ == "__main__":
    main()