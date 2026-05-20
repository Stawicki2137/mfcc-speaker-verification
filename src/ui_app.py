import sys
import tkinter as tk
from pathlib import Path

# Add project root to sys.path to support package imports if run as a script
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.app import SpeakerVerificationApp

def main():
    root = tk.Tk()
    app = SpeakerVerificationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()