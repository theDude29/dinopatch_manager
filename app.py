import sys
import os
import json
import torch
from PySide6.QtWidgets import QApplication
import ctypes
from PySide6.QtGui import QIcon

# Import MVC modules
from model import AppState
from view import MainWindow
from controller import MainController

def load_config(config_path="config.json"):
    """
    Loads configuration parameters from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        dict: Configuration dictionary with model settings and paths.
    """
    default_config = {
        "model_size": "small",
        "max_size": 672,
        "path_repo_dino": "/home/thedude/Documents/dima/dinov3/"
    }
    
    if not os.path.exists(config_path):
        print(f"⚠️ Configuration file {config_path} not found. Falling back to default parameters.")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error while reading JSON: {e}")
        return default_config

def main():

    # --- ASTUCE WINDOWS POUR LA BARRE DES TÂCHES ---
    # Cela permet à Windows de regrouper les fenêtres sous l'icône de RaptorVision
    # plutôt que sous l'icône générique de Python.
    myappid = 'mycompany.raptorvision.v1' # Chaîne arbitraire unique
    if sys.platform == 'win32':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    
    # --- DÉFINIR L'ICÔNE DE L'APPLICATION ---
    # --- CHEMIN ABSOLU (Indispensable sous Linux) ---
    base_path = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_path, "assets", "icon.png")
    
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    else:
        print(f"⚠️ Erreur : Icône non trouvée à l'adresse {icon_path}")
    app.setWindowIcon(app_icon)

    # 1. Load configuration settings
    config = load_config()
    
    MODEL_SIZE = config.get("model_size", "small")
    MAX_SIZE = config.get("max_size", 672)
    REPO_DIR = config.get("path_repo_dino", "")

    # 2. Initialize the Qt Application
    app.setStyle("Fusion") # Dark/Professional look across all OS
    app.setDesktopFileName("raptorvision")

    # 3. Initialize the MODEL (State)
    model_state = AppState(REPO_DIR)
    
    # Automatic hardware detection (GPU/CPU)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model_state.dino.device = torch.device(device_name)

    # 4. Initialize the VIEW (UI Interface)
    view = MainWindow()
    view.setWindowIcon(app_icon)

    # 5. Initialize the CONTROLLER
    # Inject variables loaded from the JSON config
    controller = MainController(
        model=model_state, 
        view=view, 
        model_size=MODEL_SIZE, 
        max_size=MAX_SIZE
    )
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()