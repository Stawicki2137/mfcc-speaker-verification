import tkinter as tk
from tkinter import ttk
import queue

from src.ui.models_state import AppState
from src.ui.controllers import AppController

class SpeakerVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Verification System")
        self.root.geometry("900x950")
        self.root.resizable(False, False)
        
        self.state = AppState()
        
        self.log_queue = queue.Queue()
        self.controller = AppController(self.state, self._enqueue_log)
        
        self._build_ui()
        self._process_logs()

    def _enqueue_log(self, msg: str) -> None:
        self.log_queue.put(msg)
        
    def _process_logs(self) -> None:
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if hasattr(self, 'log_text'):
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        self.root.after(100, self._process_logs)
        
    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text=" Logs ", padding="5")
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, bg="#f4f4f4")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        from src.ui.components.training_tab import TrainingTab
        self.train_tab = TrainingTab(self.notebook, self.controller, self.state)
        self.notebook.add(self.train_tab, text="Training")
        
        from src.ui.components.prediction_tab import PredictionTab
        self.pred_tab = PredictionTab(self.notebook, self.controller, self.state)
        self.notebook.add(self.pred_tab, text="Prediction")

def run_app():
    root = tk.Tk()
    app = SpeakerVerificationApp(root)
    root.mainloop()