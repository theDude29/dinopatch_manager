import sys
import os
import torch
from PySide6.QtWidgets import QApplication

# Import de nos modules MVC
from model import AppState
from view import MainWindow
from controller import MainController

# ==========================================================
# CONFIGURATION GLOBALE (CENTRALISÉE)
# ==========================================================
# Choisissez la version de DINO : 'small', 'base', 'large'
MODEL_SIZE = 'small' 

# Taille maximale de l'image pour l'IA (Toujours un multiple de 16)
# Plus c'est grand, plus c'est précis mais gourmand en VRAM
MAX_SIZE = 672 

# Chemins d'accès
REPO_DIR = '/home/thedude/Documents/dima/dinov3/'
DEFAULT_IMAGES_DIR = './animal/'

# Détection automatique du matériel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================================

def main():
    # 1. Création de l'application Qt
    app = QApplication(sys.argv)
    
    # Style "Fusion" pour un look sombre et pro sur tous les OS
    app.setStyle("Fusion")

    # 2. Initialisation du MODÈLE (State)
    # On passe le chemin du repo et le device au démarrage
    model_state = AppState(REPO_DIR)
    model_state.dino.device = torch.device(DEVICE)

    # 3. Initialisation de la VUE (Interface)
    view = MainWindow()

    # 4. Initialisation du CONTRÔLEUR (Le Cerveau)
    # On lui injecte les constantes globales pour qu'il configure 
    # le modèle et les sauvegardes de patchs en conséquence.
    controller = MainController(
        model=model_state, 
        view=view, 
        model_size=MODEL_SIZE, 
        max_size=MAX_SIZE
    )

    # 5. Lancement
    print(f"--- DINO Patch Manager Pro ---")
    print(f"📍 Modèle : {MODEL_SIZE.upper()}")
    print(f"📍 Résolution : {MAX_SIZE}px")
    print(f"📍 Device : {DEVICE.upper()}")
    print(f"------------------------------")
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()