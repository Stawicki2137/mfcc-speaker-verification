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

    def start_training(self):
        if not self.state.target_dir or not self.state.others_dir:
            self.log("Error: Target or Others directory not set.")
            return
            
        self.log(f"Starting training with model: {self.state.selected_model_type}")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        from src.dataset import load_dataset
        from src.train import create_models
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        import joblib
        from pathlib import Path

        try:
            mfcc_params = {
                "num_coefficients": self.state.num_coefficients,
                "num_filters": self.state.num_filters,
                "frame_size_ms": self.state.frame_size_ms,
                "frame_step_ms": self.state.frame_step_ms
            }
            
            X, y, file_paths = load_dataset(self.state.target_dir, self.state.others_dir, **mfcc_params)
            
            X_train, X_test, y_train, y_test, _, paths_test = train_test_split(
                X, y, file_paths, test_size=0.25, random_state=42, stratify=y
            )
            
            models = create_models()
            model = models[self.state.selected_model_type]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Find mislabels
            mislabels = []
            for i in range(len(y_test)):
                if y_test[i] != y_pred[i]:
                    mislabels.append({
                        "path": str(paths_test[i]),
                        "true_label": int(y_test[i]),
                        "pred_label": int(y_pred[i])
                    })
            
            save_path = Path("models") / f"{self.state.model_name}.pkl"
            save_path.parent.mkdir(exist_ok=True)
            joblib.dump({"model": model, "mfcc_params": mfcc_params}, save_path)
            
            self.log(f"Training finished! F1: {f1:.4f}")
            self.update_state(
                mislabels=mislabels,
                active_model_path=save_path.name,
                active_model=model,
                active_mfcc_params=mfcc_params
            )
            
        except Exception as e:
            self.log(f"Training Error: {str(e)}")