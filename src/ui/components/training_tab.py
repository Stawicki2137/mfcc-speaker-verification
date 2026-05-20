import tkinter as tk
from tkinter import ttk, filedialog
from src.ui.components.mislabels_panel import MislabelsPanel

class TrainingTab(ttk.Frame):
    def __init__(self, parent, controller, state):
        super().__init__(parent, padding="10")
        self.controller = controller
        self.state = state
        
        # Directory Vars
        self.target_var = tk.StringVar(value=state.target_dir)
        self.others_var = tk.StringVar(value=state.others_dir)
        
        # MFCC Vars
        self.num_coeff_var = tk.IntVar(value=state.num_coefficients)
        self.num_filters_var = tk.IntVar(value=state.num_filters)
        self.frame_size_var = tk.DoubleVar(value=state.frame_size_ms)
        self.frame_step_var = tk.DoubleVar(value=state.frame_step_ms)
        
        # Model Vars
        self.model_type_var = tk.StringVar(value=state.selected_model_type)
        self.model_name_var = tk.StringVar(value=state.model_name)
        
        self.param_c_var = tk.DoubleVar(value=1.0)
        self.param_max_iter_var = tk.IntVar(value=1000)
        self.param_n_estimators_var = tk.IntVar(value=100)
        self.param_n_neighbors_var = tk.IntVar(value=3)
        self.param_svm_kernel_var = tk.StringVar(value=state.svm_kernel)
        
        self._build()
        self._update_model_params_ui()
        
    def _build(self):
        # 1. Directories
        df = ttk.LabelFrame(self, text=" Dataset ", padding="5")
        df.pack(fill=tk.X, pady=5)
        
        ttk.Button(df, text="Select Target", command=self._sel_target).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(df, textvariable=self.target_var).grid(row=0, column=1, pady=2)
        
        ttk.Button(df, text="Select Others", command=self._sel_others).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(df, textvariable=self.others_var).grid(row=1, column=1, pady=2)
        
        # 2. Parameters Frame (MFCC + Model)
        params_frame = ttk.Frame(self)
        params_frame.pack(fill=tk.X, pady=5)
        
        # MFCC
        mfcc_f = ttk.LabelFrame(params_frame, text=" MFCC Parameters ", padding="5")
        mfcc_f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Label(mfcc_f, text="Num Coeffs:").grid(row=0, column=0, sticky=tk.W, padx=2)
        ttk.Spinbox(mfcc_f, from_=5, to=40, textvariable=self.num_coeff_var, width=8).grid(row=0, column=1, pady=2)
        
        ttk.Label(mfcc_f, text="Num Filters:").grid(row=1, column=0, sticky=tk.W, padx=2)
        ttk.Spinbox(mfcc_f, from_=10, to=80, textvariable=self.num_filters_var, width=8).grid(row=1, column=1, pady=2)
        
        ttk.Label(mfcc_f, text="Frame Size (ms):").grid(row=2, column=0, sticky=tk.W, padx=2)
        ttk.Entry(mfcc_f, textvariable=self.frame_size_var, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Label(mfcc_f, text="Frame Step (ms):").grid(row=3, column=0, sticky=tk.W, padx=2)
        ttk.Entry(mfcc_f, textvariable=self.frame_step_var, width=10).grid(row=3, column=1, pady=2)
        
        # Model specific parameters
        self.model_f = ttk.LabelFrame(params_frame, text=" Model Parameters ", padding="5")
        self.model_f.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.param_frames = {}
        
        # LR Frame
        f_lr = ttk.Frame(self.model_f)
        ttk.Label(f_lr, text="Max Iterations:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(f_lr, textvariable=self.param_max_iter_var, width=10).grid(row=0, column=1, pady=2)
        ttk.Label(f_lr, text="C (Reg):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(f_lr, textvariable=self.param_c_var, width=10).grid(row=1, column=1, pady=2)
        self.param_frames["Logistic Regression"] = f_lr
        
        # SVM Frame
        f_svm = ttk.Frame(self.model_f)
        ttk.Label(f_svm, text="C (Reg):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(f_svm, textvariable=self.param_c_var, width=10).grid(row=0, column=1, pady=2)
        ttk.Label(f_svm, text="Kernel:").grid(row=1, column=0, sticky=tk.W)
        ttk.Combobox(f_svm, textvariable=self.param_svm_kernel_var, values=["linear", "poly", "rbf", "sigmoid"], state="readonly", width=8).grid(row=1, column=1, pady=2)
        self.param_frames["SVM"] = f_svm
        
        # RF Frame
        f_rf = ttk.Frame(self.model_f)
        ttk.Label(f_rf, text="Estimators:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(f_rf, textvariable=self.param_n_estimators_var, width=10).grid(row=0, column=1, pady=2)
        self.param_frames["Random Forest"] = f_rf
        
        # KNN Frame
        f_knn = ttk.Frame(self.model_f)
        ttk.Label(f_knn, text="Neighbors (k):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(f_knn, textvariable=self.param_n_neighbors_var, width=10).grid(row=0, column=1, pady=2)
        self.param_frames["k-NN"] = f_knn
        
        # 3. Execution
        exec_f = ttk.Frame(self)
        exec_f.pack(fill=tk.X, pady=10)
        
        ttk.Label(exec_f, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        algo_cb = ttk.Combobox(exec_f, textvariable=self.model_type_var, values=self.controller.available_models, state="readonly", width=17)
        algo_cb.pack(side=tk.LEFT)
        algo_cb.bind("<<ComboboxSelected>>", self._on_model_select)
        
        ttk.Label(exec_f, text="Save As:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(exec_f, textvariable=self.model_name_var, width=15).pack(side=tk.LEFT)
        ttk.Label(exec_f, text=".pkl").pack(side=tk.LEFT)
        
        ttk.Button(exec_f, text="TRAIN MODEL", command=self._train).pack(side=tk.RIGHT, padx=5)
        
        # Mislabels Panel
        self.mislabels = MislabelsPanel(self, self.controller, self.state)
        self.mislabels.pack(fill=tk.BOTH, expand=True, pady=5)

    def _on_model_select(self, event=None):
        self.controller.update_state(selected_model_type=self.model_type_var.get())
        self._update_model_params_ui()
        
    def _update_model_params_ui(self):
        # Hide all
        for f in self.param_frames.values():
            f.pack_forget()
            
        # Show selected
        algo = self.model_type_var.get()
        if algo in self.param_frames:
            self.param_frames[algo].pack(fill=tk.BOTH, expand=True)
        
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
        self.controller.update_state(
            model_name=self.model_name_var.get(),
            num_coefficients=self.num_coeff_var.get(),
            num_filters=self.num_filters_var.get(),
            frame_size_ms=self.frame_size_var.get(),
            frame_step_ms=self.frame_step_var.get(),
            svm_kernel=self.param_svm_kernel_var.get(),
            model_kwargs={
                "C": self.param_c_var.get(),
                "max_iter": self.param_max_iter_var.get(),
                "n_estimators": self.param_n_estimators_var.get(),
                "n_neighbors": self.param_n_neighbors_var.get(),
                "kernel": self.param_svm_kernel_var.get()
            }
        )
        self.controller.start_training()