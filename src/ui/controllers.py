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
        from pathlib import Path
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
            kwargs = getattr(self.state, "model_kwargs", {})
            
            if self.state.selected_model_type == "Logistic Regression":
                model.set_params(classifier__C=kwargs.get("C", 1.0), classifier__max_iter=kwargs.get("max_iter", 1000))
            elif self.state.selected_model_type == "SVM":
                model.set_params(classifier__C=kwargs.get("C", 1.0))
            elif self.state.selected_model_type == "Random Forest":
                model.set_params(n_estimators=kwargs.get("n_estimators", 100))
            elif self.state.selected_model_type == "k-NN":
                model.set_params(classifier__n_neighbors=kwargs.get("n_neighbors", 3))
            
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

    def start_recording(self):
        if self.state.active_model is None:
            self.log("Error: Please train or load a model first.")
            return False
            
        import sounddevice as sd
        self.audio_queue = queue.Queue()
        self.fs = 44100
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())
            
        self.stream = sd.InputStream(samplerate=self.fs, channels=1, dtype='int16', callback=callback)
        self.stream.start()
        self.log("Recording started...")
        return True

    def stop_recording_and_predict(self, callback_ui=None):
        if not hasattr(self, 'stream') or not self.stream.active:
            return
            
        self.stream.stop()
        self.stream.close()
        self.log("Recording stopped. Processing...")
        
        threading.Thread(target=self._process_audio_thread, args=(callback_ui,), daemon=True).start()

    def predict_from_file(self, path: str, callback_ui=None):
        if self.state.active_model is None:
            self.log("Error: Please train or load a model first.")
            return
        
        self.log(f"Processing file: {path}")
        threading.Thread(target=self._process_file_thread, args=(path, callback_ui), daemon=True).start()

    def _process_file_thread(self, path, callback_ui):
        from pathlib import Path
        from src.predict import get_target_probability
        from src.features import extract_feature_vector_from_file
        try:
            _, f_vec = extract_feature_vector_from_file(Path(path), **self.state.active_mfcc_params)
            prob = get_target_probability(self.state.active_model, f_vec.reshape(1, -1))
            
            is_target = prob >= 0.5
            self.log(f"File {Path(path).name}: {prob*100:.1f}% -> {'GRANTED' if is_target else 'DENIED'}")
            
            if callback_ui:
                callback_ui(prob, is_target)
        except Exception as e:
            self.log(f"Prediction error (file): {e}")
        
    def _process_audio_thread(self, callback_ui):
        import numpy as np
        import time
        from pathlib import Path
        from scipy.io import wavfile
        from src.predict import get_target_probability
        from src.features import extract_feature_vector_from_file
        
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
            
        if not chunks:
            self.log("No audio recorded.")
            return
            
        recording = np.concatenate(chunks, axis=0)
        temp_path = Path("temp_audio") / f"rec_{int(time.time())}.wav"
        temp_path.parent.mkdir(exist_ok=True)
        wavfile.write(temp_path, self.fs, recording)
        
        try:
            _, f_vec = extract_feature_vector_from_file(temp_path, **self.state.active_mfcc_params)
            prob = get_target_probability(self.state.active_model, f_vec.reshape(1, -1))
            
            res_text = f"Target Probability: {prob*100:.1f}%"
            is_target = prob >= 0.5
            self.log(res_text + (" -> ACCESS GRANTED" if is_target else " -> DENIED"))
            
            if callback_ui:
                callback_ui(prob, is_target)
                
        except Exception as e:
            self.log(f"Prediction error: {e}")

    def load_model(self, path: str):
        import joblib
        from pathlib import Path
        try:
            data = joblib.load(path)
            if isinstance(data, dict):
                self.update_state(
                    active_model=data["model"],
                    active_mfcc_params=data["mfcc_params"],
                    active_model_path=Path(path).name
                )
            else:
                self.update_state(
                    active_model=data,
                    active_mfcc_params={},
                    active_model_path=Path(path).name
                )
            self.log(f"Loaded model: {Path(path).name}")
        except Exception as e:
            self.log(f"Failed to load model: {e}")