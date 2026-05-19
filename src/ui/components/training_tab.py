import tkinter as tk
from tkinter import ttk, filedialog
from src.ui.components.mislabels_panel import MislabelsPanel

class TrainingTab(ttk.Frame):
    def __init__(self, parent, controller, state):
        super().__init__(parent, padding="10")
        self.controller = controller
        self.state = state
        
        self.target_var = tk.StringVar(value=state.target_dir)
        self.others_var = tk.StringVar(value=state.others_dir)
        self.model_type_var = tk.StringVar(value=state.selected_model_type)
        self.model_name_var = tk.StringVar(value=state.model_name)
        
        self._build()
        
    def _build(self):
        # Directories
        df = ttk.Frame(self)
        df.pack(fill=tk.X, pady=5)
        ttk.Button(df, text="Select Target", command=self._sel_target).grid(row=0, column=0, padx=5)
        ttk.Label(df, textvariable=self.target_var).grid(row=0, column=1)
        
        ttk.Button(df, text="Select Others", command=self._sel_others).grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(df, textvariable=self.others_var).grid(row=1, column=1)
        
        # Model Selection
        mf = ttk.Frame(self)
        mf.pack(fill=tk.X, pady=10)
        ttk.Label(mf, text="Model Algorithm:").pack(side=tk.LEFT, padx=5)
        algo_cb = ttk.Combobox(mf, textvariable=self.model_type_var, values=self.controller.available_models, state="readonly")
        algo_cb.pack(side=tk.LEFT)
        algo_cb.bind("<<ComboboxSelected>>", lambda e: self.controller.update_state(selected_model_type=self.model_type_var.get()))
        
        ttk.Label(mf, text="Save As:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(mf, textvariable=self.model_name_var).pack(side=tk.LEFT)
        
        # Train button
        ttk.Button(self, text="TRAIN MODEL", command=self._train).pack(pady=10)
        
        # Mislabels Panel
        self.mislabels = MislabelsPanel(self, self.controller, self.state)
        self.mislabels.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def _sel_target(self):
        d = filedialog.askdirectory()
        if d:
            self.target_var.set(d)
            self.controller.update_state(target_dir=d)
            
    def _sel_others(self):
        d = filedialog.askdirectory()
        if d:
            self.others_var.set(d)
            self.controller.update_state(others_dir=d)
            
    def _train(self):
        self.controller.update_state(model_name=self.model_name_var.get())
        self.controller.start_training()