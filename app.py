import sys
import os
import torch
from PySide6.QtWidgets import QApplication
from model import AppState
from view import MainWindow
from controller import MainController

# --- CONFIGURATION ---
MODEL_SIZE = 'small' 
MAX_SIZE = 672 
REPO_DIR = '/home/thedude/Documents/dima/dinov3/'

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    model_state = AppState(REPO_DIR)
    view = MainWindow()
    
    controller = MainController(
        model=model_state, 
        view=view, 
        model_size=MODEL_SIZE, 
        max_size=MAX_SIZE
    )

    print(f"DINO Manager prêt sur {model_state.dino.device}")
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()