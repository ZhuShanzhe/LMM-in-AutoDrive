import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import Qwen3TTSService

def main() -> None:
    service = Qwen3TTSService(
        model_id_or_path="models/Qwen3-TTS-12Hz-1.7B",
        output_dir="outputs/tts",
    )

    rec = service.generate("前方路口请左转", output_file="outputs/tts/standard.wav", speaker="Vivian")
    print("[single]", rec["audio_file"], "| speaker:", rec["speaker"])

    print("[dialects]", Qwen3TTSService.supported_dialects())
    rec = service.generate(
        "前头那个路口左拐一下",
        output_file="outputs/tts/sichuan.wav",
        dialect="sichuan",
    )
    print("[dialect]", rec["audio_file"], "| dialect:", rec["dialect"])

    rec = service.generate(
        "请减速至每小时40公里",
        output_file="outputs/tts/noisy.wav",
        speaker="Uncle_Fu",
        add_noise=True,
        noise_type="pink",
        snr_db=15.0,
    )
    print("[noisy]", rec["audio_file"], "| snr:", rec["snr_db"])

    records = service.generate_batch(
        texts=["请靠边停车", "在下一个红绿灯右转"],
        output_dir="outputs/tts/batch",
        file_names=["cmd_00001.wav", "cmd_00002.wav"],
        speakers=["Vivian", "Serena"],
    )
    for r in records:
        print("[batch]", r["text"], "=>", r["audio_file"])

if __name__ == "__main__":
    main()