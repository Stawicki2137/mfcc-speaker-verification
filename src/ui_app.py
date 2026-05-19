import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from pathlib import Path
import joblib
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Import from existing project files
from dataset import load_dataset
from predict import get_target_probability
from features import extract_feature_vector_from_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp_audio"

class SpeakerVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Verification System")
        self.root.geometry("800x900")
        self.root.resizable(False, False)

        # State variables
        self.target_dir = tk.StringVar(value="")
        self.others_dir = tk.StringVar(value="")
        self.model_name = tk.StringVar(value="my_model")
        
        # MFCC Parameters
        self.num_coefficients = tk.IntVar(value=13)
        self.num_filters = tk.IntVar(value=26)
        self.frame_size_ms = tk.DoubleVar(value=25.0)
        self.frame_step_ms = tk.DoubleVar(value=10.0)
        
        # Model Parameters
        self.max_iter = tk.IntVar(value=1000)
        self.c_value = tk.DoubleVar(value=1.0)
        
        # Active Model State
        self.active_model_path = tk.StringVar(value="None")
        self.active_model = None
        self.active_mfcc_params = {}
        
        # Logging Queue for Thread-safe UI updates
        self.log_queue = queue.Queue()
        self.root.after(100, self.process_logs)

        self._create_widgets()
        
        # Ensure temp dir exists
        TEMP_DIR.mkdir(exist_ok=True)

    def _create_widgets(self):
        # Create main canvas with scrollbar if needed, or just a PanedWindow/Frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Style
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))
        style.configure("SubHeader.TLabel", font=("Helvetica", 11, "bold"))
        
        # --- TOP SECTION: TRAINING ---
        train_frame = ttk.LabelFrame(main_frame, text=" Model Training ", padding="10")
        train_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._build_training_section(train_frame)
        
        # --- BOTTOM SECTION: PREDICTION ---
        predict_frame = ttk.LabelFrame(main_frame, text=" Prediction & Inference ", padding="10")
        predict_frame.pack(fill=tk.X, pady=(0, 10))
        
        self._build_prediction_section(predict_frame)
        
        # --- LOG SECTION ---
        log_frame = ttk.LabelFrame(main_frame, text=" Logs ", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, bg="#f4f4f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_training_section(self, parent):
        # Data Selection
        data_frame = ttk.Frame(parent)
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(data_frame, text="Select Target Folder", command=self.select_target_dir).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(data_frame, textvariable=self.target_dir, width=60, relief="sunken").grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Button(data_frame, text="Select Others Folder", command=self.select_others_dir).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(data_frame, textvariable=self.others_dir, width=60, relief="sunken").grid(row=1, column=1, padx=5, pady=2)
        
        # Parameters Frame
        params_frame = ttk.Frame(parent)
        params_frame.pack(fill=tk.X, pady=10)
        
        # MFCC Params
        mfcc_frame = ttk.LabelFrame(params_frame, text="MFCC Parameters")
        mfcc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(mfcc_frame, text="Num Coefficients:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(mfcc_frame, from_=5, to=40, textvariable=self.num_coefficients, width=8).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(mfcc_frame, text="Num Filters:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Spinbox(mfcc_frame, from_=10, to=80, textvariable=self.num_filters, width=8).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(mfcc_frame, text="Frame Size (ms):").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(mfcc_frame, textvariable=self.frame_size_ms, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(mfcc_frame, text="Frame Step (ms):").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(mfcc_frame, textvariable=self.frame_step_ms, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # Model Params
        model_frame = ttk.LabelFrame(params_frame, text="Logistic Regression Parameters")
        model_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(model_frame, text="Max Iterations:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.max_iter, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(model_frame, text="C (Regularization):").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.c_value, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Execution
        exec_frame = ttk.Frame(parent)
        exec_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(exec_frame, text="Model Name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(exec_frame, textvariable=self.model_name, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Label(exec_frame, text=".pkl").pack(side=tk.LEFT)
        
        self.train_btn = ttk.Button(exec_frame, text="TRAIN MODEL", command=self.start_training)
        self.train_btn.pack(side=tk.RIGHT, padx=5)
        
        self.train_status = ttk.Label(exec_frame, text="Ready", foreground="blue", font=("Helvetica", 10, "bold"))
        self.train_status.pack(side=tk.RIGHT, padx=20)

    def _build_prediction_section(self, parent):
        # Load Model
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(top_frame, text="Load Existing Model", command=self.load_model_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="Active Model:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Label(top_frame, textvariable=self.active_model_path, foreground="blue").pack(side=tk.LEFT)
        
        # Audio Source
        src_frame = ttk.Frame(parent)
        src_frame.pack(fill=tk.X, pady=15)
        
        self.record_btn = ttk.Button(src_frame, text="🎙️ Record from Microphone (3s)", command=self.record_audio)
        self.record_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        ttk.Button(src_frame, text="📁 Load WAV File", command=self.predict_from_file).pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)
        
        # Results
        res_frame = ttk.Frame(parent)
        res_frame.pack(fill=tk.X, pady=10)
        
        self.prob_label = ttk.Label(res_frame, text="Target Probability: --%", font=("Helvetica", 12))
        self.prob_label.pack(pady=5)
        
        self.decision_label = ttk.Label(res_frame, text="WAITING FOR INPUT", font=("Helvetica", 16, "bold"), foreground="gray")
        self.decision_label.pack(pady=5)

    def log(self, message):
        self.log_queue.put(message)
        
    def process_logs(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(100, self.process_logs)

    # --- ACTIONS ---
    def select_target_dir(self):
        d = filedialog.askdirectory(initialdir=PROJECT_ROOT / "data" / "raw")
        if d: self.target_dir.set(d)

    def select_others_dir(self):
        d = filedialog.askdirectory(initialdir=PROJECT_ROOT / "data" / "raw")
        if d: self.others_dir.set(d)
        
    def start_training(self):
        target_dir = self.target_dir.get()
        others_dir = self.others_dir.get()
        
        if not target_dir or not others_dir:
            messagebox.showerror("Error", "Please select both Target and Others directories.")
            return
            
        model_name = self.model_name.get().strip()
        if not model_name:
            messagebox.showerror("Error", "Please enter a valid model name.")
            return

        self.train_btn.config(state=tk.DISABLED)
        self.train_status.config(text="Training in progress...", foreground="orange")
        self.log(f"Starting training for model: {model_name}.pkl")
        
        # Collect parameters
        mfcc_params = {
            "num_coefficients": self.num_coefficients.get(),
            "num_filters": self.num_filters.get(),
            "frame_size_ms": self.frame_size_ms.get(),
            "frame_step_ms": self.frame_step_ms.get()
        }
        
        model_params = {
            "max_iter": self.max_iter.get(),
            "C": self.c_value.get()
        }
        
        threading.Thread(target=self._train_thread, args=(target_dir, others_dir, model_name, mfcc_params, model_params), daemon=True).start()

    def _train_thread(self, target_dir, others_dir, model_name, mfcc_params, model_params):
        try:
            self.log(f"Loading dataset with MFCC params: {mfcc_params}")
            X, y, file_paths = load_dataset(target_dir, others_dir, **mfcc_params)
            
            if X.shape[0] < 4:
                raise ValueError("Dataset is too small.")
                
            self.log(f"Dataset loaded. X shape: {X.shape}. Training model...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )
            
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=model_params["max_iter"],
                    C=model_params["C"],
                    random_state=42,
                    class_weight="balanced"
                ))
            ])
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Save
            MODELS_DIR.mkdir(exist_ok=True)
            save_path = MODELS_DIR / f"{model_name}.pkl"
            
            # Save model and params together
            save_dict = {
                "model": model,
                "mfcc_params": mfcc_params
            }
            joblib.dump(save_dict, save_path)
            
            self.log(f"Training finished! F1 Score: {f1:.4f}")
            self.log(f"Model saved to: {save_path}")
            
            # Update UI safely
            self.root.after(0, self._train_success, save_path, save_dict, f1)
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.root.after(0, self._train_error, str(e))

    def _train_success(self, path, save_dict, f1):
        self.train_btn.config(state=tk.NORMAL)
        self.train_status.config(text=f"Done! F1: {f1:.2f}", foreground="green")
        self.active_model_path.set(path.name)
        self.active_model = save_dict["model"]
        self.active_mfcc_params = save_dict["mfcc_params"]
        messagebox.showinfo("Success", f"Model trained and saved successfully.\nF1 Score: {f1:.4f}")
        
    def _train_error(self, error_msg):
        self.train_btn.config(state=tk.NORMAL)
        self.train_status.config(text="Error occurred", foreground="red")
        messagebox.showerror("Training Error", error_msg)

    def load_model_dialog(self):
        file_path = filedialog.askopenfilename(initialdir=MODELS_DIR, filetypes=[("Pickle Files", "*.pkl")])
        if not file_path:
            return
            
        try:
            data = joblib.load(file_path)
            if isinstance(data, dict) and "model" in data and "mfcc_params" in data:
                self.active_model = data["model"]
                self.active_mfcc_params = data["mfcc_params"]
            else:
                self.active_model = data
                self.active_mfcc_params = {}
                
            self.active_model_path.set(Path(file_path).name)
            self.log(f"Model loaded: {self.active_model_path.get()}")
            messagebox.showinfo("Success", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
        
    def record_audio(self):
        if self.active_model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
            
        self.record_btn.config(state=tk.DISABLED, text="Recording...")
        self.decision_label.config(text="RECORDING...", foreground="orange")
        self.root.update()
        
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        try:
            fs = 44100
            duration = 3.0
            self.log(f"Recording for {duration} seconds...")
            
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            
            save_path = TEMP_DIR / f"rec_{int(time.time())}.wav"
            wavfile.write(save_path, fs, recording)
            self.log(f"Audio saved to {save_path.name}")
            
            self.root.after(0, self._record_success, save_path)
            
        except Exception as e:
            self.log(f"Recording Error: {str(e)}")
            self.root.after(0, self._record_error, str(e))

    def _record_success(self, save_path):
        self.record_btn.config(state=tk.NORMAL, text="🎙️ Record from Microphone (3s)")
        self._run_prediction(save_path)
        
    def _record_error(self, error_msg):
        self.record_btn.config(state=tk.NORMAL, text="🎙️ Record from Microphone (3s)")
        messagebox.showerror("Error", f"Recording failed:\n{error_msg}")
        
    def predict_from_file(self):
        file_path = filedialog.askopenfilename(initialdir=PROJECT_ROOT, filetypes=[("WAV Files", "*.wav")])
        if not file_path:
            return
            
        self._run_prediction(Path(file_path))

    def _run_prediction(self, wav_path):
        if self.active_model is None:
            messagebox.showerror("Error", "Please load a model first.")
            return
            
        try:
            self.log(f"Extracting features from {wav_path.name}...")
            _, feature_vector = extract_feature_vector_from_file(wav_path, **self.active_mfcc_params)
            
            feature_vector = feature_vector.reshape(1, -1)
            target_prob = get_target_probability(self.active_model, feature_vector)
            
            prob_percent = target_prob * 100
            self.prob_label.config(text=f"Target Probability: {prob_percent:.2f}%")
            
            if target_prob >= 0.5:
                self.decision_label.config(text="ACCESS GRANTED (Target)", foreground="green")
                self.log("Prediction: Target Speaker (ACCESS GRANTED)")
            else:
                self.decision_label.config(text="ACCESS DENIED (Other)", foreground="red")
                self.log("Prediction: Other Speaker (ACCESS DENIED)")
                
        except Exception as e:
            self.log(f"Prediction Error: {str(e)}")
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeakerVerificationApp(root)
    root.mainloop()