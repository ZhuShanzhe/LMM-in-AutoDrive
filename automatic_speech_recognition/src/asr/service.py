import json
import time
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from .config import FunASRConfig
from .funasr_model import FunASRModel


class FunASRService:
    """
    Speech recognition service using FunASR.

    Supports two modes:
        - 'single': Transcribes audio as a single sentence (with punctuation).
        - 'speaker': Transcribes with speaker diarization (identifies different speakers).
    """
    def __init__(
        self,
        config: Optional[FunASRConfig] = None,
        mode: str = "single",
        device: str = "cuda:0",
        **model_kwargs: Any,
    ):
        """
        Initialize the service.

        Args:
            config: Pre-configured FunASRConfig instance. If provided, mode/device/kwargs are ignored.
            mode: Recognition mode (used only if config is None).
            device: Device (used only if config is None).
            **model_kwargs: Additional model configuration overrides (used only if config is None).
        """
        if config is None:
            config = FunASRConfig(mode=mode, device=device, **model_kwargs)

        self.config = config
        self.model = FunASRModel(config)
        self.mode = config.get_mode()

        self.last_transcription_time = 0.0
        self.last_batch_time = 0.0

    def transcribe(
        self,
        audio_path: str,
        output_json: Optional[str] = None,
        print_result: bool = True,
        **generate_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to the input audio file.
            output_json: If provided, saves the formatted result to this JSON file.
            print_result: If True, prints the result to console.
            **generate_kwargs: Additional arguments for model.generate().

        Returns:
            A formatted dictionary containing transcription result and metadata.
        """
        if not audio_path:
            raise ValueError("audio_path cannot be empty.")

        start_time = time.perf_counter()

        raw_results = self.model.generate(audio_path, **generate_kwargs)
        if not raw_results:
            raise RuntimeError("No results returned from model.")
        raw_result = raw_results[0]

        elapsed = time.perf_counter() - start_time
        self.last_transcription_time = elapsed

        if self.mode == "single":
            formatted = self._format_single(raw_result, audio_path, elapsed)
        else:  # speaker
            formatted = self._format_speaker(raw_result, audio_path, elapsed)

        if print_result:
            self._print_result(formatted)

        if output_json:
            self._save_json(formatted, output_json)

        return formatted

    def transcribe_batch(
        self,
        audio_paths: List[str],
        output_json_dir: Optional[str] = None,
        print_result: bool = True,
        **generate_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files.
            output_json_dir: Directory to save JSON files (one per input). If None, no JSON saved.
            print_result: If True, prints each result.
            **generate_kwargs: Additional arguments for model.generate().

        Returns:
            A list of formatted result dictionaries.
        """
        results = []
        total_start = time.perf_counter()

        for i, path in enumerate(audio_paths, 1):
            print(f"Processing file {i}/{len(audio_paths)}: {path}")
            json_out = None
            if output_json_dir:
                stem = Path(path).stem
                json_out = Path(output_json_dir) / f"{stem}.json"
            result = self.transcribe(
                audio_path=path,
                output_json=str(json_out) if json_out else None,
                print_result=print_result,
                **generate_kwargs,
            )
            results.append(result)

        total_elapsed = time.perf_counter() - total_start
        self.last_batch_time = total_elapsed
        print(f"Batch transcription finished in {total_elapsed:.3f} seconds total.")
        return results

    def _format_single(self, raw: Dict[str, Any], audio_path: str, elapsed: float) -> Dict[str, Any]:
        """Format result for 'single' mode."""
        return {
            "audio_file": audio_path,
            "mode": self.mode,
            "text": raw.get("text", ""),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": elapsed,
        }

    def _format_speaker(self, raw: Dict[str, Any], audio_path: str, elapsed: float) -> Dict[str, Any]:
        """Format result for 'speaker' mode."""
        sentences = raw.get("sentence_info", [])
        formatted_sentences = [
            {
                "speaker": sent.get("spk", "UNKNOWN"),
                "text": sent.get("text", ""),
                "start": sent.get("start", 0.0),
                "end": sent.get("end", 0.0),
            }
            for sent in sentences
        ]
        return {
            "audio_file": audio_path,
            "mode": self.mode,
            "total_speakers": len({s.get("spk") for s in sentences if "spk" in s}),
            "sentences": formatted_sentences,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": elapsed,
        }

    def _print_result(self, result: Dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print("TRANSCRIPTION RESULT")
        print("=" * 50)

        if self.mode == "single":
            print(f"Text: {result.get('text', 'N/A')}")
        else:  # speaker
            print(f"Total Speakers Detected: {result.get('total_speakers', 0)}")
            print("-" * 30)
            for sent in result.get("sentences", []):
                print(f"[Speaker {sent['speaker']}] {sent['start']:.2f}s - {sent['end']:.2f}s: {sent['text']}")

        print(f"\nProcessing Time: {result.get('processing_time_seconds', 0):.3f} seconds")
        print("=" * 50 + "\n")

    def _save_json(self, data: Dict[str, Any], filepath: str) -> None:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Result saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")
