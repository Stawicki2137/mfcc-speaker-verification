import tkinter as tk
from tkinter import ttk, filedialog

class PredictionTab(ttk.Frame):
    def __init__(self, parent, controller, state):
        super().__init__(parent, padding="10")
        self.controller = controller
        self.state = state
        self.state.add_callback(self._on_state_change)
        
        self.model_lbl_var = tk.StringVar(value="None")
        self._build()
        
    def _build(self):
        # Load Model
        tf = ttk.Frame(self)
        tf.pack(fill=tk.X, pady=5)
        ttk.Button(tf, text="Load Model (.pkl)", command=self._load).pack(side=tk.LEFT)
        ttk.Label(tf, text="Active:").pack(side=tk.LEFT, padx=5)
        ttk.Label(tf, textvariable=self.model_lbl_var, foreground="blue").pack(side=tk.LEFT)
        
        # Audio
        af = ttk.Frame(self)
        af.pack(fill=tk.X, pady=20)
        self.start_btn = ttk.Button(af, text="🎙️ Start Recording", command=self._start_rec)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.stop_btn = ttk.Button(af, text="⏹️ Stop Recording", command=self._stop_rec, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Result
        rf = ttk.Frame(self)
        rf.pack(fill=tk.X, pady=10)
        self.prob_lbl = ttk.Label(rf, text="Probability: --%", font=("Helvetica", 12))
        self.prob_lbl.pack(pady=5)
        self.dec_lbl = ttk.Label(rf, text="WAITING", font=("Helvetica", 16, "bold"), foreground="gray")
        self.dec_lbl.pack(pady=5)
        
    def _load(self):
        p = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl")])
        if p:
            self.controller.load_model(p)
            
    def _on_state_change(self):
        self.model_lbl_var.set(self.state.active_model_path)
        
    def _start_rec(self):
        if self.controller.start_recording():
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.dec_lbl.config(text="RECORDING...", foreground="orange")
            
    def _stop_rec(self):
        self.stop_btn.config(state=tk.DISABLED)
        self.controller.stop_recording_and_predict(self._show_result)
        
    def _show_result(self, prob, is_target):
        # Must be thread-safe for Tkinter, so we use after()
        self.after(0, self._update_labels, prob, is_target)
        
    def _update_labels(self, prob, is_target):
        self.start_btn.config(state=tk.NORMAL)
        self.prob_lbl.config(text=f"Probability: {prob*100:.1f}%")
        if is_target:
            self.dec_lbl.config(text="TARGET SPEAKER", foreground="green")
        else:
            self.dec_lbl.config(text="OTHER SPEAKER", foreground="red")