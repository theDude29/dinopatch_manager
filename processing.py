import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
LIB_PATH = "./lib0"
IMAGE_INPUT_DIR = "./animal/ena24"
OUTPUT_DIR = "./output_ena24"

# --- PARAMÈTRES DE PERFORMANCE ---
MAX_SIZE = 672     # Taille maximum (votre max_size)
BATCH_SIZE = 1      # Garder à 1 si les ratios d'images varient
NUM_WORKERS = 4     
MODEL_SIZE = 'small'
REPO_DIR = '/home/thedude/Documents/dima/dinov3/'
DEVICE = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DATASET AVEC REDIMENSIONNEMENT À LA VOLÉE
# ==========================================

class FastDataset(Dataset):
    """Charge et redimensionne les images à la volée en respectant le ratio."""
    def __init__(self, folder, max_size=1024):
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
        
        # --- LOGIQUE DE REDIMENSIONNEMENT (Multiple de 16) ---
        w, h = img.size
        ratio = min(self.max_size / w, self.max_size / h)
        new_w, new_h = (int(w * ratio) // 16) * 16, (int(h * ratio) // 16) * 16
        
        # Redimensionnement BILINEAR comme dans votre fonction
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        return self.normalize(img_resized), name

# ==========================================
# 3. MOTEUR D'INFÉRENCE BATCHÉ
# ==========================================

class BatchScanner:
    def __init__(self, repo_dir, size, device):
        self.device = torch.device(device)
        arch = {'small': 'dinov3_vits16', 'base': 'dinov3_vitb16', 'large': 'dinov3_vitl16'}[size]
        ckpt = {'small': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 
                'base': 'dinov3_vitb16_pretrain.pth', 
                'large': 'dinov3_vitl16_pretrain.pth'}[size]
        
        self.model = torch.hub.load(repo_dir, arch, source='local', pretrained=False)
        ckpt_path = os.path.join(repo_dir, 'checkpoints', ckpt)
        sd = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.to(self.device).eval()
        self.dim = {'small': 384, 'base': 768, 'large': 1024}[size]

    @torch.inference_mode()
    def scan_batch(self, batch_tensors, lib_vectors):
        with torch.amp.autocast(device_type='cuda'):
            features = self.model.get_intermediate_layers(batch_tensors.to(self.device), n=1)[0]
            features = F.normalize(features, dim=-1)
            similarity_matrix = torch.matmul(features, lib_vectors.to(self.device).T)
            best_patch_scores, _ = torch.max(similarity_matrix, dim=2)
            global_scores, _ = torch.max(best_patch_scores, dim=1)
            
        return global_scores, best_patch_scores

# ==========================================
# 4. BOUCLE D'EXÉCUTION PRINCIPALE
# ==========================================

def main():
    scanner = BatchScanner(REPO_DIR, MODEL_SIZE, DEVICE)
    # On passe MAX_SIZE au dataset
    dataset = FastDataset(IMAGE_INPUT_DIR, max_size=MAX_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    lib_vec_path = os.path.join(LIB_PATH, "vectors.pt")
    if not os.path.exists(lib_vec_path):
        print(f"Erreur : Librairie introuvable à {lib_vec_path}")
        return
    lib_vectors = torch.load(lib_vec_path, map_location=DEVICE)

    print(f"🚀 Scan de {len(dataset)} images (Max Size: {MAX_SIZE}px)...")

    results_file = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_file, "w") as f_log:
        f_log.write("filename,best_score\n")

        for batch_t, batch_names in tqdm(loader):
            # Récupération dynamique de la taille après redimensionnement
            _, _, H, W = batch_t.shape
            hp, wp = H // 16, W // 16
            
            globals, heatmaps_batch = scanner.scan_batch(batch_t, lib_vectors)
            
            for i in range(len(batch_names)):
                name = batch_names[i]
                score = globals[i].item()
                
                # Reshape basé sur la taille actuelle de l'image redimensionnée
                hm_data = heatmaps_batch[i].reshape(hp, wp).cpu().numpy()
                
                # Rendu net (repeat 16) pour correspondre à la nouvelle taille
                hm_img_data = (cm.jet(np.clip(hm_data, 0, 1))[:, :, :3] * 255).astype(np.uint8).repeat(16, axis=0).repeat(16, axis=1)
                
                out_img_name = f"score_{score:.4f}_{name}"
                Image.fromarray(hm_img_data).save(os.path.join(OUTPUT_DIR, out_img_name))
                
                f_log.write(f"{name},{score:.4f}\n")

    print(f"\n✅ Terminé. Résultats dans {OUTPUT_DIR}")

if __name__ == "__main__":
    main()