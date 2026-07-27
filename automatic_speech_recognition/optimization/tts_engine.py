import os
import torchaudio
from CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice
from .config import Config


class TtsEngine:
    """
    Core TTS engine using CosyVoice.
    Accepts a Config object for flexible parameter control.
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None

    def _load_model(self):
        """Lazy-load the CosyVoice model."""
        if self._model is None:
            print(f"Loading CosyVoice model from: {self.config.model_dir}")
            self._model = CosyVoice(self.config.model_dir)
        return self._model

    def synthesize(
        self,
        text: str,
        speaker: str = None,
        instruct_text: str = None,
        output_path: str = None,
        stream: bool = False,
    ) -> str:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            speaker: Voice speaker (overrides default).
            instruct_text: Natural language instruction (e.g., 'speak in Sichuan dialect').
            output_path: Path to save the output WAV file.
            stream: Whether to use streaming generation.

        Returns:
            Path to the generated WAV file.
        """
        model = self._load_model()
        speaker = speaker or self.config.default_speaker

        if instruct_text:
            generator = model.inference_instruct(
                text, speaker, instruct_text, stream=stream
            )
        else:
            generator = model.inference_sft(text, speaker, stream=stream)

        for result in generator:
            audio_tensor = result['tts_speech']
            break
        else:
            raise RuntimeError("No output generated from TTS model.")

        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "output.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, audio_tensor, self.config.sample_rate)

        return output_path
