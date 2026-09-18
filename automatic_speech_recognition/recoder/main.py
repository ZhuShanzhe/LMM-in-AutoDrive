import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recoder import AudioRecorder, RecorderConfig


def main():
    parser = argparse.ArgumentParser(description="Record audio from microphone and save as WAV.")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--output", type=str, help="Output WAV file name (optional)")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--no-countdown", action="store_true", help="Disable countdown before recording")
    args = parser.parse_args()

    config = RecorderConfig(
        sample_rate=args.samplerate,
        default_duration=args.duration,
        countdown=not args.no_countdown,
    )
    recorder = AudioRecorder(config)

    filepath = recorder.record_and_save(duration=args.duration, filename=args.output)
    print(f"Saved recording to: {filepath}")


if __name__ == "__main__":
    main()
