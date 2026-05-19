import queue
import threading
from typing import Callable

from src.ui.models_state import AppState
from src.train import create_models

class AppController:
    def __init__(self, state: AppState, log_callback: Callable[[str], None]):
        self.state = state
        self.log = log_callback
        self.available_models = list(create_models().keys())
        
    def update_state(self, **kwargs):
                for k, v in kwargs.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
                self.state.notify()
        
            def play_audio(self, path: str):
                import sounddevice as sd
                from scipy.io import wavfile
                try:
                    fs, data = wavfile.read(path)
                    self.log(f"Playing {Path(path).name}...")
                    threading.Thread(target=self._play_thread, args=(data, fs), daemon=True).start()
                except Exception as e:
                    self.log(f"Error playing audio: {e}")
        
            def _play_thread(self, data, fs):
                import sounddevice as sd
                sd.play(data, fs)
                sd.wait()
                self.log("Playback finished.")
        