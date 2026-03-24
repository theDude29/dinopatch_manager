import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import json
import shutil
import numpy as np
from PIL import Image

MODEL_CONFIGS = {
    'small': {'arch': 'dinov3_vits16', 'ckpt': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 'dim': 384},
    'base': {'arch': 'dinov3_vitb16', 'ckpt': 'dinov3_vitb16_pretrain.pth', 'dim': 768},
    'large': {'arch': 'dinov3_vitl16', 'ckpt': 'dinov3_vitl16_pretrain.pth', 'dim': 1024}
}

class PatchLibrary:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.images_dir = os.path.join(lib_path, "images")
        self.metadata_path = os.path.join(lib_path, "metadata.json")
        self.vectors_path = os.path.join(lib_path, "vectors.pt")
        os.makedirs(self.images_dir, exist_ok=True)
        self.metadata = []
        self.vectors = None
        if os.path.exists(self.metadata_path):
            self.load()

    @staticmethod
    def merge(path_a, path_b, path_out):
        """Fusionne deux bibliothèques dans une troisième."""
        # 1. Charger les deux bibliothèques source
        lib_a = PatchLibrary(path_a)
        lib_b = PatchLibrary(path_b)
        
        # 2. Créer la bibliothèque de sortie
        lib_out = PatchLibrary(path_out)
        
        # 3. Fusionner les métadonnées
        # On combine simplement les listes
        lib_out.metadata = lib_a.metadata + lib_b.metadata
        
        # 4. Fusionner les vecteurs
        if lib_a.vectors is not None and lib_b.vectors is not None:
            lib_out.vectors = torch.cat([lib_a.vectors, lib_b.vectors], dim=0)
        elif lib_a.vectors is not None:
            lib_out.vectors = lib_a.vectors.clone()
        elif lib_b.vectors is not None:
            lib_out.vectors = lib_b.vectors.clone()
            
        # 5. Copier toutes les images physiques
        for lib_src in [lib_a, lib_b]:
            for img_name in os.listdir(lib_src.images_dir):
                src_file = os.path.join(lib_src.images_dir, img_name)
                dst_file = os.path.join(lib_out.images_dir, img_name)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
        
        # 6. Sauvegarder la nouvelle librairie
        lib_out.save()
        return lib_out

    def add_patch(self, vector, source_img_path, coords, dino_version, input_size):
        file_name = os.path.basename(source_img_path)
        dest_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(dest_path):
            shutil.copy2(source_img_path, dest_path)
        
        self.metadata.append({
            "image_name": file_name,
            "coords": coords,
            "dino_version": dino_version,
            "input_size": input_size
        })
        
        new_vec = vector.detach().cpu().reshape(1, -1)
        self.vectors = new_vec if self.vectors is None else torch.cat([self.vectors, new_vec], dim=0)
        self.save()

    def remove_patch(self, index):
        if self.vectors is not None and 0 <= index < len(self.metadata):
            self.metadata.pop(index)
            if len(self.metadata) == 0:
                self.vectors = None
                if os.path.exists(self.vectors_path): os.remove(self.vectors_path)
            else:
                self.vectors = torch.cat([self.vectors[:index], self.vectors[index+1:]])
            self.save()
            return True
        return False

    def save(self):
        if self.vectors is not None:
            torch.save(self.vectors, self.vectors_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def load(self):
        if os.path.exists(self.vectors_path):
            self.vectors = torch.load(self.vectors_path, map_location='cpu')
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

class DinoManager:
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
        config = MODEL_CONFIGS[size]
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
        if self.model is None: return None
        w, h = pil_img.size
        ratio = min(max_size / w, max_size / h)
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
        img_t = self.transform(img_resized).unsqueeze(0).to(self.device)
        with torch.amp.autocast(device_type=self.device.type):
            features = self.model.get_intermediate_layers(img_t, n=1)[0]
            hp, wp = new_h // 16, new_w // 16
            features = F.normalize(features[0, -(hp*wp):, :].reshape(hp, wp, -1), dim=-1)
        return features, (new_w, new_h)

class AppState:
    def __init__(self, repo_dir):
        self.dino = DinoManager(repo_dir)
        self.active_library = None
        self.image_folder = None
        self.image_list = []
        self.current_image_idx = 0