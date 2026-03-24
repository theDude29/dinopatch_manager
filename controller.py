import os
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter
from PySide6.QtCore import Qt
import matplotlib.cm as cm
import torch
import numpy as np
from model import PatchLibrary
from view import PatchCard

class MainController:
    def __init__(self, model, view, model_size, max_size):
        self.model = model
        self.view = view
        self.model_size = model_size
        self.max_size = max_size
        self.threshold = 0.60
        self.current_inspected_idx = None
        self.current_features = None
        self.last_click_coords = None
        self.last_click_vector = None

        # IA Init
        self.model.dino.load_model(self.model_size)

        # Connexions Menu
        self.view.action_open_folder.triggered.connect(self.open_image_folder)
        self.view.action_new_lib.triggered.connect(self.create_new_library)
        self.view.action_open_lib.triggered.connect(self.open_library)
        self.view.action_edit_lib.triggered.connect(self.toggle_edit_mode)
        self.view.action_save.triggered.connect(self.save_library)

        # Connexions Boutons/Sliders
        self.view.btn_next.clicked.connect(self.next_image)
        self.view.btn_prev.clicked.connect(self.prev_image)
        self.view.btn_cancel.clicked.connect(self.cancel_last_click)
        self.view.slider_threshold.valueChanged.connect(self.update_threshold_from_slider)
        self.view.btn_delete_patch.clicked.connect(self.delete_inspected_patch)

        # Clics Images
        self.view.view_local.label_image.mousePressEvent = self.handle_image_click
        self.view.view_memory.label_image.mousePressEvent = self.handle_memory_click

        # Nouvelles connexions pour le mode édition
        self.view.btn_delete_lib.clicked.connect(self.delete_current_library)
        self.view.btn_merge.clicked.connect(self.merge_libraries_dialog)

    def delete_current_library(self):
        """Supprime le dossier complet de la librairie."""
        if not self.model.active_library:
            QMessageBox.warning(self.view, "Erreur", "Aucune librairie chargée.")
            return

        lib_path = self.model.active_library.lib_path
        msg = f"Voulez-vous supprimer DÉFINITIVEMENT la librairie :\n{lib_path} ?"
        if QMessageBox.critical(self.view, "Danger", msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            import shutil
            shutil.rmtree(lib_path) # Suppression du dossier
            self.model.active_library = None
            self.update_status_info()
            self.update_memory_view()
            self.populate_edit_grid()
            QMessageBox.information(self.view, "Info", "Librairie supprimée.")

    def merge_libraries_dialog(self):
        """Dialogue pour choisir 2 sources et 1 destination."""
        p_a = QFileDialog.getExistingDirectory(self.view, "Sélectionner Librairie A")
        if not p_a: return
        p_b = QFileDialog.getExistingDirectory(self.view, "Sélectionner Librairie B")
        if not p_b: return
        p_out = QFileDialog.getExistingDirectory(self.view, "Dossier de destination (VIDE)")
        if not p_out: return

        try:
            self.view.status_bar.showMessage("Fusion en cours...", 0)
            merged = PatchLibrary.merge(p_a, p_b, p_out)
            self.model.active_library = merged
            self.update_status_info()
            QMessageBox.information(self.view, "Succès", "Fusion réussie !")
        except Exception as e:
            QMessageBox.critical(self.view, "Erreur", str(e))
        finally:
            self.view.status_bar.clearMessage()

    def open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self.view, "Sélectionner le dossier d'images")
        if folder:
            self.model.image_folder = folder
            self.model.image_list = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.model.current_image_idx = 0
            if self.model.image_list:
                self.load_current_image()
                self.view.show_explorer()

    def create_heatmap_pixmap(self, vector, img_path, input_size):
        """Génère un QPixmap d'une image avec la heatmap d'un vecteur spécifique."""
        if not os.path.exists(img_path): return None
        
        # 1. Charger l'image et extraire les features
        pil_img = Image.open(img_path).convert('RGB')
        features, (tw, th) = self.model.dino.get_features(pil_img, input_size)
        
        # 2. Calculer la similarité (Heatmap)
        with torch.no_grad():
            flat_f = features.reshape(-1, self.model.dino.current_config['dim'])
            # On compare toute l'image au vecteur du patch
            sim = torch.matmul(flat_f, vector.to(self.model.dino.device))
            heatmap = sim.reshape(features.shape[0], features.shape[1]).cpu().numpy()

        # 3. Créer le Pixmap de base
        base_pixmap = QPixmap(img_path).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 4. Générer la surcouche Heatmap (Jet)
        heatmap_norm = np.clip(heatmap, 0, 1)
        color_data = (cm.jet(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
        # On utilise un seuil pour la transparence (ex: 0.5)
        alphas = np.where(heatmap > 0.5, 160, 0).astype(np.uint8)
        
        rgba_data = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba_data[:, :, :3] = color_data
        rgba_data[:, :, 3] = alphas

        q_img = QImage(rgba_data.data, heatmap.shape[1], heatmap.shape[0], 4 * heatmap.shape[1], QImage.Format_RGBA8888)
        
        # 5. Fusionner Heatmap et Image
        result_pixmap = base_pixmap.copy()
        painter = QPainter(result_pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False) # Carrés nets
        painter.drawImage(result_pixmap.rect(), q_img)
        painter.end()
        
        return result_pixmap

    def set_scaled_pixmap(self, pixmap, target_label):
        """Redimensionne un pixmap pour qu'il s'ajuste au label sans le déformer."""
        if pixmap is None or pixmap.isNull(): return
        
        # On prend la taille actuelle du widget
        w = target_label.width()
        h = target_label.height()
        
        # On redimensionne en gardant les proportions
        scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target_label.setPixmap(scaled)

    def load_current_image(self):
        img_name = self.model.image_list[self.model.current_image_idx]
        img_path = os.path.join(self.model.image_folder, img_name)
        
        pil_img = Image.open(img_path).convert('RGB')
        # On récupère les features et la taille de calcul (ex: 672px)
        self.current_features, (tw, th) = self.model.dino.get_features(pil_img, self.max_size)
        
        # Image de référence en haute résolution
        self.view_local_base = QPixmap(img_path).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Affichage adapté à la fenêtre
        self.set_scaled_pixmap(self.view_local_base, self.view.view_local.label_image)
        self.set_scaled_pixmap(self.view_local_base, self.view.view_memory.label_image)
        
        self.update_memory_view()

    def handle_image_click(self, event):
        if self.current_features is None: return
        label = self.view.view_local.label_image
        displayed_pix = label.pixmap()
        if not displayed_pix: return

        # 1. Calcul des offsets (car l'image est centrée dans le label)
        off_x = (label.width() - displayed_pix.width()) // 2
        off_y = (label.height() - displayed_pix.height()) // 2

        # 2. Position du clic relative à l'image affichée
        local_x = event.position().x() - off_x
        local_y = event.position().y() - off_y

        # Vérification si clic hors image
        if local_x < 0 or local_y < 0 or local_x >= displayed_pix.width() or local_y >= displayed_pix.height():
            return

        # 3. MAPPING : Règle de trois vers l'image de calcul (view_local_base)
        ratio_x = local_x / displayed_pix.width()
        ratio_y = local_y / displayed_pix.height()
        
        real_x = ratio_x * self.view_local_base.width()
        real_y = ratio_y * self.view_local_base.height()

        # 4. Conversion en patch
        px, py = int(real_x // 16), int(real_y // 16)
        max_py, max_px = self.current_features.shape[0], self.current_features.shape[1]
        py, px = min(max(0, py), max_py - 1), min(max(0, px), max_px - 1)

        self.last_click_coords = (py, px)
        self.last_click_vector = self.current_features[py, px, :]
        
        with torch.no_grad():
            flat_f = self.current_features.reshape(-1, self.model.dino.current_config['dim'])
            sim = torch.matmul(flat_f, self.last_click_vector)
            heatmap = sim.reshape(max_py, max_px).cpu().numpy()
            
        self.display_heatmap(heatmap, self.view.view_local)

    def handle_memory_click(self, event):
        if not hasattr(self, 'memory_match_indices') or self.memory_scores is None: return
        label = self.view.view_memory.label_image
        displayed_pix = label.pixmap()
        if not displayed_pix: return

        off_x = (label.width() - displayed_pix.width()) // 2
        off_y = (label.height() - displayed_pix.height()) // 2
        local_x = event.position().x() - off_x
        local_y = event.position().y() - off_y

        if 0 <= local_x < displayed_pix.width() and 0 <= local_y < displayed_pix.height():
            # Mapping
            ratio_x = local_x / displayed_pix.width()
            ratio_y = local_y / displayed_pix.height()
            real_x = ratio_x * self.view_local_base.width()
            real_y = ratio_y * self.view_local_base.height()
            
            px, py = int(real_x // 16), int(real_y // 16)
            
            if 0 <= py < self.memory_scores.shape[0] and 0 <= px < self.memory_scores.shape[1]:
                if self.memory_scores[py, px] >= self.threshold:
                    self.current_inspected_idx = int(self.memory_match_indices[py, px])
                    metadata = self.model.active_library.metadata[self.current_inspected_idx]
                    self.show_source_in_inspector(metadata)
                    self.view.btn_delete_patch.setVisible(True)

    def show_source_in_inspector(self, metadata):
        """Affiche la source avec la heatmap du patch sélectionné."""
        img_path = os.path.join(self.model.active_library.images_dir, metadata['image_name'])
        # On récupère le vecteur correspondant dans la lib
        vector = self.model.active_library.vectors[self.current_inspected_idx]
        
        pixmap = self.create_heatmap_pixmap(vector, img_path, metadata.get('input_size', 512))
        
        if pixmap:
            self.view.label_source_preview.setPixmap(pixmap.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.view.label_source_info.setText(f"📄 {metadata['image_name']}\n📍 Patch : {metadata['coords']}")

    def delete_inspected_patch(self):
        if self.current_inspected_idx is not None and self.model.active_library:
            if QMessageBox.question(self.view, "Suppression", "Supprimer ce patch ?") == QMessageBox.Yes:
                self.model.active_library.remove_patch(self.current_inspected_idx)
                self.current_inspected_idx = None
                self.view.btn_delete_patch.setVisible(False)
                self.view.label_source_preview.setText("Supprimé")
                self.update_status_info()
                self.update_memory_view()

    def update_memory_view(self):
        if not self.model.active_library or self.current_features is None or self.model.active_library.vectors is None or self.model.active_library.vectors.shape[0] == 0:
            if hasattr(self, 'view_local_base'): self.view.view_memory.label_image.setPixmap(self.view_local_base)
            self.memory_scores = None
            return

        with torch.no_grad():
            flat_f = self.current_features.reshape(-1, self.model.dino.current_config['dim'])
            lib_vectors = self.model.active_library.vectors.to(self.model.dino.device)
            scores = torch.matmul(flat_f, lib_vectors.T)
            best_scores, best_indices = torch.max(scores, dim=1)
            hp, wp = self.current_features.shape[0], self.current_features.shape[1]
            self.memory_match_indices = best_indices.reshape(hp, wp).cpu().numpy()
            self.memory_scores = best_scores.reshape(hp, wp).cpu().numpy()
            self.display_heatmap(self.memory_scores, self.view.view_memory)

    def display_heatmap(self, heatmap, target_widget):
        if heatmap is None or not hasattr(self, 'view_local_base'): return
        
        heatmap_norm = np.clip(heatmap, 0, 1)
        color_data = (cm.jet(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
        alphas = np.where(heatmap > self.threshold, 180, 0).astype(np.uint8)
        
        rgba_data = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba_data[:, :, :3], rgba_data[:, :, 3] = color_data, alphas

        q_img = QImage(rgba_data.data, heatmap.shape[1], heatmap.shape[0], 4 * heatmap.shape[1], QImage.Format_RGBA8888)
        
        # On dessine sur le canvas haute résolution
        canvas = self.view_local_base.copy()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(canvas.rect(), q_img)
        painter.end()
        
        # --- CHANGEMENT ICI : On affiche la version adaptée au label ---
        self.set_scaled_pixmap(canvas, target_widget.label_image)

    def update_threshold_from_slider(self, value):
        self.threshold = value / 100.0
        self.view.label_thresh_val.setText(f"🎯 Seuil : {self.threshold:.2f}")
        self.update_memory_view()

    def update_status_info(self):
        """Synchronise les labels de la barre d'état avec l'état réel du modèle."""
        count = 0
        lib_display_name = "Aucune"
        
        if self.model.active_library:
            # On récupère le nombre de patchs
            if self.model.active_library.vectors is not None:
                count = self.model.active_library.vectors.shape[0]
            
            # On extrait le nom du dossier à partir du chemin complet
            # ex: "/home/user/ma_lib" devient "ma_lib"
            lib_display_name = os.path.basename(self.model.active_library.lib_path)
        
        # Mise à jour des labels de la Vue
        self.view.label_lib_name.setText(f"📂 Lib : {lib_display_name}")
        self.view.label_mem_count.setText(f"📦 Mémoire : {count} patchs")
        self.view.label_thresh_val.setText(f"🎯 Seuil : {self.threshold:.2f}")

    def next_image(self):
        if not self.model.image_list: return
        if self.last_click_vector is not None and self.model.active_library:
            img_path = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
            self.model.active_library.add_patch(self.last_click_vector, img_path, self.last_click_coords, self.model_size, self.max_size)
            self.cancel_last_click()
        self.model.current_image_idx = (self.model.current_image_idx + 1) % len(self.model.image_list)
        self.load_current_image()
        self.update_status_info()

    def prev_image(self):
        if self.model.image_list:
            self.model.current_image_idx = (self.model.current_image_idx - 1) % len(self.model.image_list)
            self.load_current_image()

    def cancel_last_click(self):
        """Réinitialise la sélection actuelle et efface la heatmap locale."""
        # 1. On oublie le vecteur et les coordonnées du dernier clic
        self.last_click_vector = None
        self.last_click_coords = None
        
        # 2. On rafraîchit l'affichage de la vue locale
        # On utilise 'set_scaled_pixmap' pour s'assurer que l'image propre 
        # se redimensionne parfaitement à la taille actuelle du widget.
        if hasattr(self, 'view_local_base') and self.view_local_base is not None:
            self.set_scaled_pixmap(self.view_local_base, self.view.view_local.label_image)
        
        # 3. Petit message informatif dans la barre d'état
        self.view.status_bar.showMessage("Sélection annulée.", 2000)

    def create_new_library(self):
        path = QFileDialog.getExistingDirectory(self.view, "Nouvelle Librairie")
        if path:
            self.model.active_library = PatchLibrary(path)
            self.update_status_info()

    def open_library(self):
        path = QFileDialog.getExistingDirectory(self.view, "Ouvrir Librairie")
        if path:
            self.model.active_library = PatchLibrary(path)
            self.update_status_info()
            self.update_memory_view()

    def toggle_edit_mode(self):
        """Bascule entre l'explorateur et la grille d'édition."""
        if self.view.central_stack.currentIndex() == 0:
            # On passe en mode édition
            self.populate_edit_grid()
            self.view.central_stack.setCurrentIndex(1)
            self.view.action_edit_lib.setText("🔙 Retour à l'Explorateur")
        else:
            # On revient à l'explorateur
            self.view.central_stack.setCurrentIndex(0)
            self.view.action_edit_lib.setText("✏️ Modifier la Librairie")
            self.update_memory_view()

    def populate_edit_grid(self):
        """Remplit la grille avec des cartes montrant la heatmap de chaque patch."""
        # Nettoyage
        while self.view.patch_grid.count():
            child = self.view.patch_grid.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not self.model.active_library: return

        # On affiche un message de chargement car cela peut être long
        self.view.status_bar.showMessage("Génération des miniatures (IA)... Veuillez patienter", 0)
        QApplication.processEvents() # Force la mise à jour de l'UI pour afficher le message

        for i, entry in enumerate(self.model.active_library.metadata):
            img_path = os.path.join(self.model.active_library.images_dir, entry['image_name'])
            vector = self.model.active_library.vectors[i]
            
            # Génération de l'image avec Heatmap
            hm_pixmap = self.create_heatmap_pixmap(vector, img_path, entry.get('input_size', 512))
            
            if hm_pixmap:
                card = PatchCard(hm_pixmap, f"Patch #{i}", i, self.delete_patch_from_grid)
                self.view.patch_grid.addWidget(card, i // 4, i % 4)

        self.view.status_bar.clearMessage()

    def delete_patch_from_grid(self, index):
        """Supprime un patch directement depuis la grille d'édition."""
        confirm = QMessageBox.question(self.view, "Supprimer", f"Supprimer le patch #{index} ?")
        if confirm == QMessageBox.Yes:
            self.model.active_library.remove_patch(index)
            self.populate_edit_grid() # On rafraîchit la grille
            self.update_status_info()
            self.update_memory_view()

    def save_library(self):
        if self.model.active_library: self.model.active_library.save()