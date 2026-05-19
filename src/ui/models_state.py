import dataclasses
from pathlib import Path
from typing import Any, Callable

@dataclasses.dataclass
class AppState:
    target_dir: str = ""
    others_dir: str = ""
    model_name: str = "my_model"
    
    num_coefficients: int = 13
    num_filters: int = 26
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    
    selected_model_type: str = "Logistic Regression"
    
    active_model_path: str = "None"
    active_model: Any = None
    active_mfcc_params: dict = dataclasses.field(default_factory=dict)
    
    mislabels: list[dict] = dataclasses.field(default_factory=list)
    
    # Simple observer pattern
    _callbacks: list[Callable] = dataclasses.field(default_factory=list)
    
    def add_callback(self, callback: Callable) -> None:
        self._callbacks.append(callback)
        
    def notify(self) -> None:
        for cb in self._callbacks:
            cb()