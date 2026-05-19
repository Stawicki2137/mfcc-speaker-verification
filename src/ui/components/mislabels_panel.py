import tkinter as tk
from tkinter import ttk
from typing import Callable
from pathlib import Path

class MislabelsPanel(ttk.LabelFrame):
    def __init__(self, parent, controller, state):
        super().__init__(parent, text=" Training Errors (Mislabels) ", padding="10")
        self.controller = controller
        self.state = state
        self.state.add_callback(self.refresh)
        
        # Treeview for errors
        columns = ("file", "true", "pred", "action")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=5)
        self.tree.heading("file", text="File Path")
        self.tree.heading("true", text="True Label")
        self.tree.heading("pred", text="Predicted")
        self.tree.heading("action", text="Action")
        
        self.tree.column("file", width=300)
        self.tree.column("true", width=80)
        self.tree.column("pred", width=80)
        self.tree.column("action", width=80)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self._on_double_click)

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for err in self.state.mislabels:
            file_name = Path(err['path']).name
            self.tree.insert("", tk.END, values=(
                file_name, 
                "Target" if err['true_label'] == 1 else "Other",
                "Target" if err['pred_label'] == 1 else "Other",
                "Double-click to play"
            ), tags=(err['path'],))
            
    def _on_double_click(self, event):
        item = self.tree.selection()[0]
        tags = self.tree.item(item, "tags")
        if tags:
            file_path = tags[0]
            self.controller.play_audio(file_path)