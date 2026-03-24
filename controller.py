import os
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox, QLabel
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter
from PySide6.QtCore import Qt
import matplotlib.cm as cm
import torch
import numpy as np

class MainController:
    def __init__(self, model, view, model_size, max_size):
        self.model = model  # AppState
        self.view = view    # MainWindow
        
        # On stocke les constantes injectées depuis app.py
        self.model_size = model_size
        self.max_size = max_size
        
        # Valeur du seuil par défaut (Slider)
        self.threshold = 0.60
        
        # Configuration initiale de l'IA
        self.model.dino.load_model(self.model_size)
        
        # Valeur du seuil par défaut
        self.threshold = 0.60

        # Connexion du Slider
        self.view.slider_threshold.valueChanged.connect(self.update_threshold_from_slider)
        
        # État temporaire pour l'interaction
        self.current_features = None
        self.last_click_coords = None
        self.last_click_vector = None

        # 1. Connexion des actions du Menu
        self.view.action_open_folder.triggered.connect(self.open_image_folder)
        self.view.action_new_lib.triggered.connect(self.create_new_library)
        self.view.action_open_lib.triggered.connect(self.open_library)
        self.view.action_edit_lib.triggered.connect(self.toggle_edit_mode)
        self.view.action_save.triggered.connect(self.save_library)

        # 2. Connexion des boutons de navigation
        self.view.btn_next.clicked.connect(self.next_image)
        self.view.btn_prev.clicked.connect(self.prev_image)
        self.view.btn_cancel.clicked.connect(self.cancel_last_click)

        # 3. Connexion des clics sur l'image (via un filtre d'événement ou une méthode déléguée)
        # Pour simplifier dans cette version, on redéfinit le clic sur le widget de vue locale
        self.view.view_local.label_image.mousePressEvent = self.handle_image_click

        # Initialisation du modèle DINO (par défaut en 'small')
        self.model.dino.load_model('small')

        self.view.view_memory.label_image.mousePressEvent = self.handle_memory_click

    def handle_memory_click(self, event):
        if not hasattr(self, 'memory_match_indices'): return

        # 1. Mapping des coordonnées (identique à la vue locale)
        label = self.view.view_memory.label_image
        pixmap = label.pixmap()
        offset_x = (label.width() - pixmap.width()) // 2
        offset_y = (label.height() - pixmap.height()) // 2
        img_x = event.position().x() - offset_x
        img_y = event.position().y() - offset_y

        px, py = int(img_x // 16), int(img_y // 16)

        # 2. Vérifier si on est sur un patch activé (au-dessus du seuil)
        if 0 <= py < self.memory_scores.shape[0] and 0 <= px < self.memory_scores.shape[1]:
            if self.memory_scores[py, px] < self.threshold:
                return # On ignore les clics sur les zones froides

            # 3. Récupérer l'index du patch source
            patch_idx = self.memory_match_indices[py, px]
            metadata = self.model.active_library.metadata[patch_idx]
            
            # 4. Afficher la source dans l'inspecteur
            self.show_source_in_inspector(metadata)

    def show_source_in_inspector(self, metadata):
        """Affiche l'image d'origine et encadre le patch source."""
        img_name = metadata['image_name']
        coords = metadata['coords'] # (py, px) du patch dans l'image source
        
        img_path = os.path.join(self.model.active_library.images_dir, img_name)
        
        if os.path.exists(img_path):
            # Charger l'image source
            source_pix = QPixmap(img_path)
            
            # On dessine un carré rouge sur le patch source pour bien le voir
            painter = QPainter(source_pix)
            painter.setPen(QColor(255, 0, 0, 255)) # Rouge
            # Dessiner le carré du patch (16x16)
            painter.drawRect(coords[1]*16, coords[0]*16, 16, 16)
            painter.end()
            
            # Mise à jour de la vue (vignette redimensionnée)
            self.view.label_source_preview.setPixmap(source_pix.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.view.label_source_info.setText(f"Fichier : {img_name}\nPos : Patch [{coords[0]}, {coords[1]}]")

    def update_threshold_from_slider(self, value):
        """Appelé quand on bouge le curseur"""
        self.threshold = value / 100.0
        self.view.label_thresh_val.setText(f"🎯 Seuil : {self.threshold:.2f}")
        
        # On rafraîchit les vues immédiatement pour voir l'effet du nouveau seuil
        if self.last_click_vector is not None:
            self.handle_image_click() # On sépare la logique du clic pour la réutiliser
        self.update_memory_view()

    def update_status_info(self):
        """Met à jour le nombre de patchs en mémoire"""
        count = 0
        if self.model.active_library and self.model.active_library.vectors is not None:
            count = self.model.active_library.vectors.shape[0]
        
        self.view.label_mem_count.setText(f"📦 Mémoire : {count} patchs")

    # --- GESTION DES IMAGES ---
    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self.view, "Sélectionner le dossier d'images")
        if folder:
            self.model.image_folder = folder
            self.model.image_list = sorted([
                f for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.model.current_image_idx = 0
            if self.model.image_list:
                self.load_current_image()
                self.view.show_explorer()

    def load_current_image(self):
        img_name = self.model.image_list[self.model.current_image_idx]
        img_path = os.path.join(self.model.image_folder, img_name)
        
        pil_img = Image.open(img_path).convert('RGB')
        # On récupère les features et la taille calculée (multiple de 16)
        self.current_features, (target_w, target_h) = self.model.dino.get_features(pil_img)
        
        # On stocke cette taille pour display_heatmap
        self.target_size = (target_w, target_h)

        # On crée le pixmap de base UNE SEULE FOIS
        pixmap = QPixmap(img_path).scaled(
            target_w, target_h, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # On applique EXACTEMENT le même pixmap aux deux labels
        self.view_local_base = pixmap.copy() # On garde une copie propre
        self.view.view_local.label_image.setPixmap(self.view_local_base)
        self.view.view_memory.label_image.setPixmap(self.view_local_base)
        
        self.update_memory_view()

    def next_image(self):
        if not self.model.image_list: return
        
        # Sauvegarde avant de passer à l'image suivante
        if self.last_click_vector is not None and self.model.active_library:
            img_path = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
            
            # On passe les nouvelles infos : MODEL_SIZE et MAX_SIZE
            self.model.active_library.add_patch(
                self.last_click_vector, 
                img_path, 
                self.last_click_coords,
                self.model_size, # La variable globale de ton script
                self.max_size    # La variable globale de ton script
            )
            self.update_status_info()
        
        self.model.current_image_idx = (self.model.current_image_idx + 1) % len(self.model.image_list)
        self.load_current_image()

    def show_source_in_inspector(self, metadata):
        img_name = metadata['image_name']
        py, px = metadata['coords']
        dino_v = metadata.get('dino_version', 'inconnue')
        size = metadata.get('input_size', 512)
        
        img_path = os.path.join(self.model.active_library.images_dir, img_name)
        
        if os.path.exists(img_path):
            # 1. Charger et redimensionner l'image à sa taille de calcul
            source_pix = QPixmap(img_path)
            # On calcule le ratio pour garder l'aspect ratio comme lors de l'inférence
            scaled_pix = source_pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 2. Dessiner le carré rouge (Net)
            painter = QPainter(scaled_pix)
            painter.setPen(QColor(255, 0, 0)) # Rouge pur
            painter.setBrush(QColor(255, 0, 0, 50)) # Fond rouge très transparent (optionnel)
            
            # Un patch DINO fait 16x16 pixels
            painter.drawRect(px * 16, py * 16, 16, 16)
            painter.end()
            
            # 3. Affichage dans le label de l'inspecteur
            self.view.label_source_preview.setPixmap(
                scaled_pix.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            
            # 4. Mise à jour du texte d'info
            info_text = (f"📄 Fichier : {img_name}\n"
                         f"🧠 Modèle : {dino_v}\n"
                         f"📐 Taille Inférence : {size}px\n"
                         f"📍 Patch : [{py}, {px}]")
            self.view.label_source_info.setText(info_text)

    def prev_image(self):
        if not self.model.image_list: return
        self.model.current_image_idx = (self.model.current_image_idx - 1) % len(self.model.image_list)
        self.load_current_image()

    # --- INTERACTION PATCH ---
    def handle_image_click(self, event):
        if self.current_features is None: return
        
        # 1. Récupérer le label et le pixmap affiché
        label = self.view.view_local.label_image
        pixmap = label.pixmap()
        if not pixmap: return

        # 2. Calculer les offsets si l'image est centrée (Qt.AlignCenter)
        # L'image ne commence pas forcément à (0,0) dans le label
        offset_x = (label.width() - pixmap.width()) // 2
        offset_y = (label.height() - pixmap.height()) // 2

        # 3. Traduire les coordonnées du clic en coordonnées "Image"
        # On soustrait l'offset pour que (0,0) soit le coin haut-gauche de l'image
        img_x = event.position().x() - offset_x
        img_y = event.position().y() - offset_y

        # 4. Vérifier que le clic est bien DANS l'image
        if img_x < 0 or img_y < 0 or img_x >= pixmap.width() or img_y >= pixmap.height():
            return # Clic en dehors de l'image (dans les bandes noires/vides)

        # 5. Conversion en coordonnées de patch (DINO = patches de 16x16)
        px = int(img_x // 16)
        py = int(img_y // 16)
        
        # Sécurité sur les limites de la grille de features
        max_py, max_px = self.current_features.shape[0], self.current_features.shape[1]
        py = min(max(0, py), max_py - 1)
        px = min(max(0, px), max_px - 1)

        # 6. Extraction et Heatmap (identique à avant)
        self.last_click_coords = (py, px)
        self.last_click_vector = self.current_features[py, px, :]
        
        with torch.no_grad():
            flat_f = self.current_features.reshape(-1, self.model.dino.current_config['dim'])
            sim = torch.matmul(flat_f, self.last_click_vector)
            heatmap = sim.reshape(max_py, max_px).cpu().numpy()
        
        self.display_heatmap(heatmap, self.view.view_local)

    def cancel_last_click(self):
        self.last_click_vector = None
        self.last_click_coords = None
        # Rafraîchir l'affichage pour enlever la heatmap locale
        if self.model.image_list:
            img_path = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
            self.view.view_local.label_image.setPixmap(QPixmap(img_path).scaled(self.view.view_local.label_image.size(), Qt.KeepAspectRatio))

    # --- GESTION LIBRAIRIE ---
    def create_new_library(self):
        path = QFileDialog.getExistingDirectory(self.view, "Choisir un dossier pour la nouvelle librairie")
        if path:
            from model import PatchLibrary
            self.model.active_library = PatchLibrary(path)
            QMessageBox.information(self.view, "Succès", f"Librairie '{os.path.basename(path)}' créée.")

    def open_library(self):
        path = QFileDialog.getExistingDirectory(self.view, "Ouvrir une librairie existante")
        if path:
            from model import PatchLibrary
            self.model.active_library = PatchLibrary(path)
            self.update_memory_view()

        self.update_status_info()
        self.update_memory_view()

    def toggle_edit_mode(self):
        if self.view.central_stack.currentIndex() == 0:
            self.populate_edit_grid()
            self.view.show_editor()
        else:
            self.view.show_explorer()

    def populate_edit_grid(self):
        """Remplit la grille avec les miniatures des patchs de la bibliothèque."""
        # Nettoyage de la grille existante
        for i in reversed(range(self.view.patch_grid.count())): 
            self.view.patch_grid.itemAt(i).widget().setParent(None)
            
        if not self.model.active_library: return
        
        for i, entry in enumerate(self.model.active_library.metadata):
            # Charger la miniature (crop du patch)
            img_path = os.path.join(self.model.active_library.images_dir, entry['image_name'])
            patch_icon = QLabel()
            pix = QPixmap(img_path).copy(entry['coords'][1]*16, entry['coords'][0]*16, 16, 16).scaled(64, 64)
            patch_icon.setPixmap(pix)
            self.view.patch_grid.addWidget(patch_icon, i // 5, i % 5)

    def save_library(self):
        if self.model.active_library:
            self.model.active_library.save()
            QMessageBox.information(self.view, "Sauvegarde", "Librairie enregistrée.")

    def update_memory_view(self):
        if self.model.active_library is None or self.current_features is None: return
        if self.model.active_library.vectors is None: return
        
        with torch.no_grad():
            flat_f = self.current_features.reshape(-1, self.model.dino.current_config['dim'])
            lib_vectors = self.model.active_library.vectors.to(self.model.dino.device)
            
            scores = torch.matmul(flat_f, lib_vectors.T) # [N_patches_img, N_patches_lib]
            
            # --- ACTION ---
            # On récupère les scores ET les indices des meilleurs matchs
            best_scores, best_indices = torch.max(scores, dim=1)
            
            # On stocke la grille des indices pour pouvoir la consulter au clic
            hp, wp = self.current_features.shape[0], self.current_features.shape[1]
            self.memory_match_indices = best_indices.reshape(hp, wp).cpu().numpy()
            self.memory_scores = best_scores.reshape(hp, wp).cpu().numpy()
            
            heatmap = self.memory_scores
        
        self.display_heatmap(heatmap, self.view.view_memory)

    def display_heatmap(self, heatmap, target_widget):
        """
        Convertit une matrice NumPy en QPixmap coloré avec des carrés nets (sans smoothing) 
        et l'affiche par-dessus l'image originale.
        """
        if heatmap is None: return

        # 1. Paramètres de rendu (seuil dynamique du contrôleur)
        current_threshold = self.threshold 
        alpha_max = 180  # Opacité max (0-255)
        
        # 2. Normalisation et Masquage
        # On s'assure que les scores sont entre 0 et 1
        heatmap_norm = np.clip(heatmap, 0, 1)
        
        # 3. Application de la Colormap (Jet ou Magma)
        # colormap(x) retourne un tableau (H, W, 4) avec RGBA en float [0, 1]
        color_data = cm.jet(heatmap_norm) 
        
        # 4. Gestion de la transparence (Alpha channel)
        # On met l'alpha à 0 pour tout ce qui est sous le seuil
        # Et on crée un dégradé d'opacité pour le reste
        alphas = np.where(heatmap > current_threshold, alpha_max, 0).astype(np.uint8)
        
        # Conversion en uint8 pour Qt (0-255)
        color_data = (color_data[:, :, :3] * 255).astype(np.uint8)
        
        # Reconstruction du tableau RGBA final (à la taille de la grille de patchs, ex: 42x32)
        rgba_data = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba_data[:, :, :3] = color_data
        rgba_data[:, :, 3] = alphas

        # 5. Conversion NumPy -> QImage (Petite taille, taille de la grille de patchs)
        h_small, w_small, channels = rgba_data.shape
        bytes_per_line = channels * w_small
        # q_img_small est une image aliasée (pixel-perfect) à la taille de la grille
        q_img_small = QImage(rgba_data.data, w_small, h_small, bytes_per_line, QImage.Format_RGBA8888)
        
        # 6. Superposition sur l'image d'origine
        # On repart TOUJOURS de la copie propre stockée lors du chargement
        canvas = self.view_local_base.copy()
        
        painter = QPainter(canvas)
        
        # --- FIX POUR CARRES NETS ---
        # On désactive explicitement le lissage implicite lors du dessin de q_img_small sur le grand canvas.
        # painter.setRenderHint(QPainter.SmoothPixmapTransform) # <--- ON RETIRE CETTE LIGNE !
        painter.setRenderHint(QPainter.Antialiasing, False) # Désactive l'antialiasing global
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False) # Désactive explicitement le lissage lors du redimensionnement implicite par drawImage

        # On dessine la heatmap sur toute la surface du canvas. Le redimensionnement implicite 
        # se fait maintenant sans lissage (pixel-perfect) grâce aux hints ci-dessus.
        # Chaque pixel de q_img_small deviendra un carré de 16x16 pixels net sur le canvas.
        painter.drawImage(canvas.rect(), q_img_small)
        painter.end()

        target_widget.label_image.setPixmap(canvas)
