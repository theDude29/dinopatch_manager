import os
import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter
from PySide6.QtCore import Qt
from model import PatchLibrary

class MainController:
    """
    Contrôleur central orchestrant l'IA DINOv3, la navigation et la gestion 
    de la base de données de patchs sémantiques.
    """
    
    def __init__(self, model, view, model_size, max_size):
        self.model = model
        self.view = view
        self.model_size = model_size
        self.max_size = max_size
        
        # --- INITIALISATION DE L'ÉTAT DE SESSION ---
        self.threshold = 0.60
        self.current_features = None      # Tenseur de features de l'image actuelle
        self.last_click_vector = None     # Vecteur du dernier patch cliqué
        self.last_click_coords = None     # Coordonnées (y, x) du dernier patch
        self.current_inspected_idx = None # Index du patch affiché dans l'inspecteur
        self.view_local_base = None       # Pixmap original (sans heatmap) pour le rendu
        
        # Chargement du modèle IA
        self.model.dino.load_model(self.model_size)
        
        # --- CONNEXIONS DES ACTIONS MENU ---
        self.view.action_open_folder.triggered.connect(self.open_image_folder)
        self.view.action_new_lib.triggered.connect(self.create_new_library)
        self.view.action_open_lib.triggered.connect(self.open_library)
        self.view.action_edit_lib.triggered.connect(self.toggle_edit_mode)

        # --- CONNEXIONS DES BOUTONS ET INTERFACE ---
        self.view.btn_next.clicked.connect(self.next_image)
        self.view.btn_prev.clicked.connect(self.prev_image)
        self.view.btn_cancel.clicked.connect(self.cancel_last_click)
        self.view.slider_threshold.valueChanged.connect(self.update_threshold)
        self.view.btn_delete_patch.clicked.connect(self.delete_inspected_patch)
        self.view.btn_delete_lib.clicked.connect(self.delete_current_library)
        self.view.btn_merge.clicked.connect(self.merge_libraries_dialog)

        # --- ÉVÉNEMENTS SOURIS ---
        self.view.view_local.label_image.mousePressEvent = self.handle_local_click
        self.view.view_memory.label_image.mousePressEvent = self.handle_memory_click

    def _refresh_ui(self):
        """Met à jour l'ensemble de l'interface suite à un changement de données."""
        self.update_status_info()
        self.update_memory_view()
        # Si nous sommes dans l'onglet 'Gestion', on rafraîchit la galerie de patchs
        if self.view.central_stack.currentIndex() == 1:
            self.populate_edit_grid()

    def toggle_edit_mode(self):
        """Bascule entre l'explorateur d'images et le gestionnaire de bibliothèque."""
        is_explorer = self.view.central_stack.currentIndex() == 0
        if is_explorer:
            self.populate_edit_grid()
            self.view.central_stack.setCurrentIndex(1)
            self.view.action_edit_lib.setText("🖼️ Retour Image")
        else:
            self.view.central_stack.setCurrentIndex(0)
            self.view.action_edit_lib.setText("⚙️ Gérer la librairie")
            self.update_memory_view()

    # =========================================================
    # LOGIQUE IA ET GÉNÉRATION DE RENDU (HEATMAPS)
    # =========================================================

    def create_heatmap_pixmap(self, vector, img_path, input_size):
        """Génère un rendu visuel fusionnant l'image source et sa réponse sémantique."""
        if not os.path.exists(img_path): return None
        
        # Inférence IA locale pour l'image concernée
        pil_img = Image.open(img_path).convert('RGB')
        features, (tw, th) = self.model.dino.get_features(pil_img, input_size)
        
        with torch.no_grad():
            flat = features.reshape(-1, self.model.dino.current_config['dim'])
            sim = torch.matmul(flat, vector.to(self.model.dino.device))
            heatmap = sim.reshape(features.shape[0], features.shape[1]).cpu().numpy()

        # Création du canvas
        base = QPixmap(img_path).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rgba = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = (cm.jet(np.clip(heatmap, 0, 1))[:,:,:3]*255).astype(np.uint8)
        rgba[:,:,3] = np.where(heatmap > 0.5, 160, 0).astype(np.uint8)
        
        q_img = QImage(rgba.data, heatmap.shape[1], heatmap.shape[0], 4*heatmap.shape[1], QImage.Format_RGBA8888)
        res = base.copy()
        p = QPainter(res); p.drawImage(res.rect(), q_img); p.end()
        return res

    def display_dynamic_heatmap(self, heatmap, target_widget):
        """Applique une heatmap en temps réel sur l'image affichée."""
        if heatmap is None or self.view_local_base is None: return
        
        rgba = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = (cm.jet(np.clip(heatmap, 0, 1))[:,:,:3]*255).astype(np.uint8)
        rgba[:,:,3] = np.where(heatmap > self.threshold, 180, 0).astype(np.uint8)
        
        q_img = QImage(rgba.data, heatmap.shape[1], heatmap.shape[0], 4*heatmap.shape[1], QImage.Format_RGBA8888)
        canvas = self.view_local_base.copy()
        p = QPainter(canvas)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False) # Carrés nets
        p.drawImage(canvas.rect(), q_img)
        p.end()
        self.set_scaled_pixmap(canvas, target_widget.label_image)

    # =========================================================
    # INTERACTIONS SOURIS ET NAVIGATION
    # =========================================================

    def handle_local_click(self, event):
        """Capture un patch sur l'image locale et affiche sa similarité."""
        label = self.view.view_local.label_image
        pix = label.pixmap()
        if not pix or self.current_features is None: return
        
        # Mapping coordonnées écran -> coordonnées IA (0.0 à 1.0)
        ox, oy = (label.width()-pix.width())//2, (label.height()-pix.height())//2
        rx, ry = (event.position().x()-ox)/pix.width(), (event.position().y()-oy)/pix.height()
        
        if 0 <= rx <= 1 and 0 <= ry <= 1:
            px, py = int(rx * self.view_local_base.width() // 16), int(ry * self.view_local_base.height() // 16)
            hp, wp = self.current_features.shape[:2]
            px, py = min(px, wp-1), min(py, hp-1)
            
            self.last_click_vector = self.current_features[py, px, :]
            self.last_click_coords = (py, px)
            
            with torch.no_grad():
                sim = torch.matmul(self.current_features.reshape(-1, self.model.dino.current_config['dim']), self.last_click_vector)
                self.display_dynamic_heatmap(sim.reshape(hp, wp).cpu().numpy(), self.view.view_local)

    def handle_memory_click(self, event):
        """Identifie et inspecte la source d'une activation mémoire."""
        label = self.view.view_memory.label_image
        pix = label.pixmap()
        if not pix or not hasattr(self, 'memory_scores'): return
        
        ox, oy = (label.width()-pix.width())//2, (label.height()-pix.height())//2
        rx, ry = (event.position().x()-ox)/pix.width(), (event.position().y()-oy)/pix.height()
        
        if 0 <= rx <= 1 and 0 <= ry <= 1:
            px, py = int(rx * self.view_local_base.width() // 16), int(ry * self.view_local_base.height() // 16)
            if self.memory_scores[py, px] >= self.threshold:
                self.current_inspected_idx = int(self.memory_match_indices[py, px])
                meta = self.model.active_library.metadata[self.current_inspected_idx]
                # Génération de la vue d'inspection
                hm = self.create_heatmap_pixmap(self.model.active_library.vectors[self.current_inspected_idx], 
                                                os.path.join(self.model.active_library.images_dir, meta['image_name']), meta['input_size'])
                self.view.label_source_preview.setPixmap(hm.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.view.label_source_info.setText(f"📄 {meta['image_name']}\n📍 Coord: {meta['coords']}")
                self.view.btn_delete_patch.setVisible(True)

    # =========================================================
    # GESTION DES LIBRAIRIES ET GALERIE
    # =========================================================

    def open_library(self):
        p = QFileDialog.getExistingDirectory(self.view, "Ouvrir Librairie")
        if p: 
            self.model.active_library = PatchLibrary(p)
            self._refresh_ui()

    def create_new_library(self):
        p = QFileDialog.getExistingDirectory(self.view, "Dossier Nouvelle Librairie")
        if p: 
            self.model.active_library = PatchLibrary(p)
            self._refresh_ui()

    def populate_edit_grid(self):
        """Construit dynamiquement la galerie de patchs avec feedback IA."""
        while self.view.patch_grid.count():
            w = self.view.patch_grid.takeAt(0).widget()
            if w: w.deleteLater()
            
        if not self.model.active_library or self.model.active_library.vectors is None: return
        
        self.view.status_bar.showMessage("Calcul des miniatures...", 0)
        QApplication.processEvents() # Maintient l'UI réactive
        
        from view import PatchCard
        for i, meta in enumerate(self.model.active_library.metadata):
            img_p = os.path.join(self.model.active_library.images_dir, meta['image_name'])
            hm = self.create_heatmap_pixmap(self.model.active_library.vectors[i], img_p, meta['input_size'])
            if hm:
                card = PatchCard(hm, f"#{i} {meta['image_name'][:12]}", i, self.delete_patch_from_grid)
                self.view.patch_grid.addWidget(card, i//4, i%4)
        self.view.status_bar.clearMessage()

    # =========================================================
    # UTILITAIRES ET NAVIGATION
    # =========================================================

    def set_scaled_pixmap(self, pixmap, target_label):
        if pixmap:
            s = pixmap.scaled(target_label.width(), target_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            target_label.setPixmap(s)

    def open_image_folder(self):
        f = QFileDialog.getExistingDirectory(self.view, "Sélectionner Dossier d'Images")
        if f: 
            self.model.image_folder = f
            self.model.image_list = sorted([x for x in os.listdir(f) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.model.current_image_idx = 0
            self.load_current_image()

    def load_current_image(self):
        if not self.model.image_list: return
        img_p = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
        self.current_features, (tw, th) = self.model.dino.get_features(Image.open(img_p).convert('RGB'), self.max_size)
        self.view_local_base = QPixmap(img_p).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.set_scaled_pixmap(self.view_local_base, self.view.view_local.label_image)
        self.set_scaled_pixmap(self.view_local_base, self.view.view_memory.label_image)
        self.update_memory_view()

    def update_memory_view(self):
        if not self.model.active_library or self.model.active_library.vectors is None:
            if self.view_local_base: self.set_scaled_pixmap(self.view_local_base, self.view.view_memory.label_image)
            return
        with torch.no_grad():
            f = self.current_features.reshape(-1, self.model.dino.current_config['dim'])
            s = torch.matmul(f, self.model.active_library.vectors.to(self.model.dino.device).T)
            best_s, best_i = torch.max(s, dim=1)
            hp, wp = self.current_features.shape[:2]
            self.memory_scores, self.memory_match_indices = best_s.reshape(hp, wp).cpu().numpy(), best_i.reshape(hp, wp).cpu().numpy()
            self.display_dynamic_heatmap(self.memory_scores, self.view.view_memory)

    def next_image(self):
        """Sauvegarde le patch sélectionné et passe à l'image suivante."""
        if self.last_click_vector is not None and self.model.active_library:
            p = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
            self.model.active_library.add_patch(self.last_click_vector, p, self.last_click_coords, self.model_size, self.max_size)
        
        if self.model.image_list:
            self.model.current_image_idx = (self.model.current_image_idx + 1) % len(self.model.image_list)
            self.last_click_vector = None
            self.load_current_image()
            self.update_status_info()

    def update_threshold(self, v):
        self.threshold = v/100.0
        self.view.label_thresh_val.setText(f"🎯 Seuil : {self.threshold:.2f}")
        self.update_memory_view()

    def update_status_info(self):
        name = os.path.basename(self.model.active_library.lib_path) if self.model.active_library else "-"
        count = len(self.model.active_library.metadata) if self.model.active_library else 0
        self.view.label_lib_name.setText(f"📂 Lib : {name}")
        self.view.label_mem_count.setText(f"📦 {count} patchs")

    def delete_patch_from_grid(self, idx):
        if QMessageBox.question(self.view, "Suppression", "Retirer ce patch ?") == QMessageBox.Yes:
            self.model.active_library.remove_patch(idx)
            self.populate_edit_grid()
            self.update_status_info()

    def delete_inspected_patch(self):
        if self.current_inspected_idx is not None:
            self.model.active_library.remove_patch(self.current_inspected_idx)
            self.view.btn_delete_patch.setVisible(False)
            self._refresh_ui()

    def cancel_last_click(self):
        self.last_click_vector = None
        self.set_scaled_pixmap(self.view_local_base, self.view.view_local.label_image)

    def prev_image(self):
        if self.model.image_list:
            self.model.current_image_idx = (self.model.current_image_idx - 1) % len(self.model.image_list)
            self.load_current_image()

    def delete_current_library(self):
        if self.model.active_library and QMessageBox.critical(self.view, "⚠️ Suppression", "Détruire la librairie ?", QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:
            import shutil
            shutil.rmtree(self.model.active_library.lib_path)
            self.model.active_library = None
            self._refresh_ui()

    def merge_libraries_dialog(self):
        a, b, o = QFileDialog.getExistingDirectory(self.view, "Lib A"), QFileDialog.getExistingDirectory(self.view, "Lib B"), QFileDialog.getExistingDirectory(self.view, "Destination")
        if a and b and o: 
            PatchLibrary.merge(a, b, o)
            QMessageBox.information(self.view, "Succès", "Librairies fusionnées.")