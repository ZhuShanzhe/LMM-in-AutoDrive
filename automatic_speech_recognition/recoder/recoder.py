import os
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
from typing import Optional
from .config import RecorderConfig


class AudioRecorder:
    """
    A simple audio recorder that captures microphone input and saves to WAV file.
    """

    def __init__(self, config: Optional[RecorderConfig] = None):
        self.config = config or RecorderConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._input_device = self._get_default_input_device()

    def _get_default_input_device(self) -> Optional[int]:
        """Return the default input device index, or None if no device found."""
        try:
            devices = sd.query_devices()
            # Find the first device with max_input_channels > 0
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"Using input device: {dev['name']}")
                    return idx
            print("No input device found. Please check your audio hardware.")
            return None
        except Exception as e:
            print(f"Error querying devices: {e}")
            return None

    def _countdown(self):
        """Display a countdown before recording starts."""
        if not self.config.countdown:
            return
        seconds = self.config.countdown_seconds
        print("Recording will start in:")
        for i in range(seconds, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        print("Recording now...")

    def record(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record audio from the default microphone.

        Args:
            duration: Recording duration in seconds. If None, uses default from config.

        Returns:
            np.ndarray: Audio data as float32, shape (samples, channels)
        """
        if duration is None:
            duration = self.config.default_duration

        self._countdown()

        recording = sd.rec(
            int(duration * self.config.sample_rate),
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            device=self._input_device,
        )
        sd.wait()
        print("Recording finished.")
        return recording

    def save_to_wav(self, audio: np.ndarray, filename: Optional[str] = None) -> str:
        """
        Save audio data to a WAV file.

        Args:
            audio: Audio data (float32 array).
            filename: Output filename. If None, generates a timestamp-based name.

        Returns:
            str: Path to the saved file.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"

        filepath = os.path.join(self.config.output_dir, filename)
        sf.write(filepath, audio, self.config.sample_rate)
        print(f"Audio saved to: {filepath}")
        return filepath

    def record_and_save(self, duration: Optional[float] = None, filename: Optional[str] = None) -> str:
        """
        Record audio and save directly to a WAV file in one step.

        Args:
            duration: Recording duration (seconds).
            filename: Output filename.

        Returns:
            str: Path to the saved file.
        """
        audio = self.record(duration)
        return self.save_to_wav(audio, filename)
