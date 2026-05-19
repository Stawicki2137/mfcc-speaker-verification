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