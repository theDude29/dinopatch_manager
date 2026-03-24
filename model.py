import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import json
import shutil
import numpy as np
from PIL import Image

# --- CONFIGURATION DES MODÈLES ---
MODEL_CONFIGS = {
    'small': {
        'arch': 'dinov3_vits16',
        'ckpt': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dim': 384
    },
    'base': {
        'arch': 'dinov3_vitb16',
        'ckpt': 'dinov3_vitb16_pretrain.pth',
        'dim': 768
    },
    'large': {
        'arch': 'dinov3_vitl16',
        'ckpt': 'dinov3_vitl16_pretrain.pth',
        'dim': 1024
    }
}

class PatchLibrary:
    """Gère le stockage physique d'une bibliothèque de patchs."""
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.images_dir = os.path.join(lib_path, "images")
        self.metadata_path = os.path.join(lib_path, "metadata.json")
        self.vectors_path = os.path.join(lib_path, "vectors.pt")
        
        # Initialisation de l'arborescence
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.metadata = []  # Liste de dict : {"image_name": str, "coords": (x, y)}
        self.vectors = None # Tenseur PyTorch [N, Dim]
        
        # Chargement si la bibliothèque existe déjà
        if os.path.exists(self.metadata_path):
            self.load()

    def add_patch(self, vector, source_img_path, coords, dino_version, input_size):
        """
        Ajoute un patch avec métadonnées enrichies.
        coords: (py, px)
        dino_version: str (ex: 'small')
        input_size: int (ex: 672)
        """
        file_name = os.path.basename(source_img_path)
        dest_path = os.path.join(self.images_dir, file_name)
        
        if not os.path.exists(dest_path):
            shutil.copy2(source_img_path, dest_path)
        
        # On enregistre tout ce qui permet de reproduire le contexte du patch
        self.metadata.append({
            "image_name": file_name,
            "coords": coords,
            "dino_version": dino_version,
            "input_size": input_size
        })
        
        new_vec = vector.detach().cpu().reshape(1, -1)
        if self.vectors is None:
            self.vectors = new_vec
        else:
            self.vectors = torch.cat([self.vectors, new_vec], dim=0)
        
        self.save()

    def remove_patch(self, index):
        """Retire un patch de la bibliothèque et met à jour les fichiers."""
        if 0 <= index < len(self.metadata):
            self.metadata.pop(index)
            # Retrait du vecteur correspondant
            self.vectors = torch.cat([self.vectors[:index], self.vectors[index+1:]])
            
            # Note : On ne supprime pas l'image du dossier 'images/' car elle peut 
            # être utilisée par d'autres patchs de la même bibliothèque.
            self.save()

    def save(self):
        """Sauvegarde atomique des vecteurs et du JSON."""
        if self.vectors is not None:
            torch.save(self.vectors, self.vectors_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def load(self):
        """Charge les données depuis le disque."""
        if os.path.exists(self.vectors_path):
            self.vectors = torch.load(self.vectors_path, map_location='cpu')
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

class DinoManager:
    """Gère le modèle IA DINOv3 et l'inférence GPU."""
    def __init__(self, repo_dir, device="cuda"):
        self.repo_dir = repo_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_config = None
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self, size):
        """Charge une architecture DINOv3 spécifique."""
        config = MODEL_CONFIGS[size]
        print(f"🧠 Chargement du modèle {size.upper()}...")
        
        # Chargement via torch.hub (local)
        self.model = torch.hub.load(self.repo_dir, config['arch'], source='local', pretrained=False)
        ckpt_path = os.path.join(self.repo_dir, 'checkpoints', config['ckpt'])
        
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        
        self.model.to(self.device).eval()
        self.current_config = config
        return True

    @torch.inference_mode()
    def get_features(self, pil_img, max_size=672):
        """Redimensionne l'image et extrait les features DINOv3."""
        if self.model is None: return None
        
        w, h = pil_img.size
        ratio = min(max_size / w, max_size / h)
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        
        img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        img_t = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast(device_type=self.device.type):
            features = self.model.get_intermediate_layers(img_t, n=1)[0]
            hp, wp = new_h // 16, new_w // 16
            # Normalisation et reshape spatial [H_patch, W_patch, Dim]
            features = F.normalize(features[0, -(hp*wp):, :].reshape(hp, wp, -1), dim=-1)
            
        return features, (new_w, new_h)

class AppState:
    """
    Conteneur global pour l'état de l'application (MVC Model).
    Il contient la bibliothèque active et le gestionnaire de modèle.
    """
    def __init__(self, repo_dir):
        self.dino = DinoManager(repo_dir)
        self.active_library = None
        self.image_folder = None
        self.image_list = []
        self.current_image_idx = 0
