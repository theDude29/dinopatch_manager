import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import json
import shutil
import uuid
import numpy as np
from PIL import Image

# DINOv3 architecture configurations
MODEL_CONFIGS = {
    'small': {'arch': 'dinov3_vits16', 'ckpt': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 'dim': 384},
    'base':  {'arch': 'dinov3_vitb16', 'ckpt': 'dinov3_vitb16_pretrain.pth', 'dim': 768},
    'large': {'arch': 'dinov3_vitl16', 'ckpt': 'dinov3_vitl16_pretrain.pth', 'dim': 1024}
}

class FastDataset(torch.utils.data.Dataset):
    """Optimized dataset for high-speed batch inference."""
    def __init__(self, folder, max_size=672):
        self.folder = folder
        self.max_size = max_size
        self.files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.folder, name)
        img = Image.open(img_path).convert('RGB')
        
        # Resize logic (multiple of 16)
        w, h = img.size
        ratio = min(self.max_size / w, self.max_size / h)
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        return self.normalize(img_resized), name

class PatchLibrary:
    """
    Handles persistent storage, metadata management, and visual rendering cache 
    for the selected semantic patches.
    """
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.images_dir = os.path.join(lib_path, "images")
        self.heatmaps_dir = os.path.join(lib_path, "heatmaps") # Cache folder
        self.metadata_path = os.path.join(lib_path, "metadata.json")
        self.vectors_path = os.path.join(lib_path, "vectors.pt")
        
        # Initialize directory structure
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.heatmaps_dir, exist_ok=True)
        
        self.metadata = []
        self.vectors = None
        if os.path.exists(self.metadata_path):
            self.load()

    def add_patch(self, vector, source_img_path, coords, dino_version, input_size, heatmap_pixmap):
        """
        Adds a patch to the library and saves its rendered heatmap to the cache 
        to ensure instantaneous display in the gallery.
        """
        patch_uid = str(uuid.uuid4())[:8]
        file_name = os.path.basename(source_img_path)
        
        # 1. Save Heatmap rendering to cache (PNG)
        hm_name = f"hm_{patch_uid}.png"
        heatmap_pixmap.save(os.path.join(self.heatmaps_dir, hm_name), "PNG")
        
        # 2. Copy source image to library
        dest_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(dest_path):
            shutil.copy2(source_img_path, dest_path)
        
        # 3. Update Metadata
        self.metadata.append({
            "image_name": file_name,
            "heatmap_cache": hm_name,
            "coords": coords,
            "dino_version": dino_version,
            "input_size": input_size
        })
        
        # Update vector tensor
        new_vec = vector.detach().cpu().reshape(1, -1)
        self.vectors = new_vec if self.vectors is None else torch.cat([self.vectors, new_vec], dim=0)
        self.save()

    def remove_patch(self, index):
        """Removes a patch and deletes its associated cache file."""
        if self.vectors is not None and 0 <= index < len(self.metadata):
            # Clean up cached image file
            hm_name = self.metadata[index].get("heatmap_cache")
            if hm_name:
                hm_path = os.path.join(self.heatmaps_dir, hm_name)
                if os.path.exists(hm_path): os.remove(hm_path)
                
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
        """Serializes vectors and metadata to disk."""
        if self.vectors is not None:
            torch.save(self.vectors, self.vectors_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def load(self):
        """Loads vectors and metadata from disk."""
        if os.path.exists(self.vectors_path):
            self.vectors = torch.load(self.vectors_path, map_location='cpu')
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    @staticmethod
    def merge(path_a, path_b, path_out):
        """Merges two existing libraries into a new destination (includes images and cache)."""
        lib_a, lib_b = PatchLibrary(path_a), PatchLibrary(path_b)
        lib_out = PatchLibrary(path_out)
        lib_out.metadata = lib_a.metadata + lib_b.metadata
        
        # Merge tensors
        vecs = [v for v in [lib_a.vectors, lib_b.vectors] if v is not None]
        if vecs:
            lib_out.vectors = torch.cat(vecs, dim=0)
            
        # Migrate physical files
        for lib in [lib_a, lib_b]:
            # Copy source images
            for img in os.listdir(lib.images_dir):
                shutil.copy2(os.path.join(lib.images_dir, img), os.path.join(lib_out.images_dir, img))
            # Copy heatmap cache
            for hm in os.listdir(lib.heatmaps_dir):
                shutil.copy2(os.path.join(lib.heatmaps_dir, hm), os.path.join(lib_out.heatmaps_dir, hm))
        
        lib_out.save()
        return lib_out

class DinoManager:
    """Handles DINOv3 inference and feature extraction."""
    def __init__(self, repo_dir, model_size='small'):
        self.repo_dir = repo_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_config = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        # 1. Initialisation de l'attribut
        self.current_size = model_size

    def load_model(self, size):
        """Loads the specified DINOv3 model size and its checkpoint."""

        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Mise à jour de l'attribut lors du chargement
        self.current_size = size

        config = MODEL_CONFIGS[size]
        self.model = torch.hub.load(self.repo_dir, config['arch'], source='local', pretrained=False)
        ckpt_path = os.path.join(self.repo_dir, 'checkpoints', config['ckpt'])
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.to(self.device).eval()
        self.current_config = config

    @torch.inference_mode()
    def get_features(self, pil_img, max_size=672):
        """Extracts normalized semantic features from an input image."""
        w, h = pil_img.size
        ratio = min(max_size / w, max_size / h)
        # Resize to multiples of 16 for ViT compatibility
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        img_t = self.transform(pil_img.resize((new_w, new_h), Image.BILINEAR)).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast(device_type=self.device.type):
            feat = self.model.get_intermediate_layers(img_t, n=1)[0]
            hp, wp = new_h // 16, new_w // 16
            # Reshape and normalize descriptors
            feat = F.normalize(feat[0, -(hp*wp):, :].reshape(hp, wp, -1), dim=-1)
        return feat, (new_w, new_h)

class AppState:
    """Maintains the global state of the application."""
    def __init__(self, repo_dir):
        self.dino = DinoManager(repo_dir)
        self.active_library = None
        self.image_folder = None
        self.image_list = []
        self.current_image_idx = 0