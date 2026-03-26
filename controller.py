import os
import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter
from PySide6.QtCore import Qt
from model import PatchLibrary
from PySide6.QtCore import QThread, Signal, QObject

class ProcessingWorker(QObject):
    finished = Signal()
    progress = Signal(int)
    log = Signal(str)

    def __init__(self, model, lib_vectors, input_dir, output_dir, max_size, export_hm, copy_orig, threshold, batch_size):
        super().__init__()
        self.model = model
        self.lib_vectors = lib_vectors
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_size = max_size
        self.export_hm = export_hm
        self.copy_orig = copy_orig # Nouveau paramètre
        self.threshold = threshold
        self.batch_size = batch_size

    def run(self):
        try:
            import os
            import torch
            import shutil
            import numpy as np
            import matplotlib.cm as cm
            from PIL import Image
            from torch.utils.data import DataLoader
            from model import FastDataset
            
            self.log.emit(f"Initializing dataset: {self.input_dir}")
            dataset = FastDataset(self.input_dir, max_size=self.max_size)
            loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)
            
            results_file = os.path.join(self.output_dir, "matching_results.txt")
            
            with open(results_file, "w") as f_log:
                f_log.write("filename,best_score\n")
                
                with torch.inference_mode():
                    for i, (batch_t, batch_names) in enumerate(loader):
                        B, _, H, W = batch_t.shape
                        hp, wp = H // 16, W // 16
                        
                        features = self.model.dino.model.get_intermediate_layers(batch_t.to(self.model.dino.device), n=1)[0]
                        features = torch.nn.functional.normalize(features, dim=-1)
                        
                        sim = torch.matmul(features, self.lib_vectors.to(self.model.dino.device).T)
                        best_patch_scores, _ = torch.max(sim, dim=2)
                        global_scores, _ = torch.max(best_patch_scores, dim=1)
                        
                        for j in range(len(batch_names)):
                            name = batch_names[j]
                            score = global_scores[j].item()
                            
                            # Séparation du nom et de l'extension (ex: 'photo', '.jpg')
                            name_no_ext, ext = os.path.splitext(name)
                            
                            f_log.write(f"{name},{score:.4f}\n")
                            
                            if score >= self.threshold:
                                # A. Export Heatmap (Format: heatmap_nom_score.extension)
                                if self.export_hm:
                                    hm_data = best_patch_scores[j].reshape(hp, wp).cpu().numpy()
                                    hm_rgb = (cm.jet(np.clip(hm_data, 0, 1))[:, :, :3] * 255).astype(np.uint8)
                                    hm_upscaled = hm_rgb.repeat(16, axis=0).repeat(16, axis=1)
                                    
                                    # Nouveau format de nommage
                                    hm_name = f"heatmap_{name_no_ext}_{score:.3f}{ext}"
                                    Image.fromarray(hm_upscaled).save(os.path.join(self.output_dir, hm_name))

                                # B. Copy Top Match (Format: match_nom_score.extension)
                                if self.copy_orig:
                                    src_path = os.path.join(self.input_dir, name)
                                    # Nouveau format de nommage cohérent
                                    dst_name = f"match_{name_no_ext}_{score:.3f}{ext}"
                                    shutil.copy2(src_path, os.path.join(self.output_dir, dst_name))
                        
                        processed_count = min((i + 1) * self.batch_size, len(dataset))
                        self.progress.emit(int((processed_count / len(dataset)) * 100))
                        
                        if i % 5 == 0 or processed_count == len(dataset):
                            self.log.emit(f"Processed {processed_count}/{len(dataset)} images...")

            self.log.emit(f"✅ Batch completed. Results in: {self.output_dir}")

        except Exception as e:
            self.log.emit(f"❌ Error: {str(e)}")
        finally:
            self.finished.emit()

class MainController:
    """
    Main Controller handling the logic between the DINO model, 
    the Patch Library, and the User Interface.
    """
    def __init__(self, model, view, model_size, max_size):
        self.model, self.view = model, view
        self.model_size, self.max_size = model_size, max_size
        self.threshold = 0.60
        
        # State initialization
        self.current_features = None
        self.last_click_vector = None
        self.last_click_coords = None
        self.current_inspected_idx = None
        self.view_local_base = None
        
        # Initialize AI model
        self.model.dino.load_model(self.model_size)
        
        # UI Signal Connections
        self.view.action_open_folder.triggered.connect(self.open_image_folder)
        self.view.action_new_lib.triggered.connect(self.create_new_library)
        self.view.action_open_lib.triggered.connect(self.open_library)
        self.view.action_edit_lib.triggered.connect(self.toggle_edit_mode)
        self.view.btn_next.clicked.connect(self.next_image)
        self.view.btn_prev.clicked.connect(self.prev_image)
        self.view.btn_cancel.clicked.connect(self.cancel_last_click)
        self.view.slider_threshold.valueChanged.connect(self.update_threshold)
        self.view.btn_delete_patch.clicked.connect(self.delete_inspected_patch)
        self.view.btn_delete_lib.clicked.connect(self.delete_current_library)
        self.view.btn_merge.clicked.connect(self.merge_libraries_dialog)
        self.view.action_about.triggered.connect(self.show_about_page)
        self.view.action_how_to.triggered.connect(self.show_how_to_page)
        
        # Mouse Event Connections
        self.view.view_local.label_image.mousePressEvent = self.handle_local_click
        self.view.view_memory.label_image.mousePressEvent = self.handle_memory_click

        self.view.action_processing.triggered.connect(lambda: self.view.central_stack.setCurrentIndex(4))
        self.view.btn_select_input.clicked.connect(self.select_proc_input)
        self.view.btn_select_output.clicked.connect(self.select_proc_output)
        self.view.btn_start_proc.clicked.connect(self.start_batch_processing)

    def change_model_parameters(self):
        new_size = self.view.combo_model.currentText()
        new_res = int(self.view.combo_res.currentText())
        
        # Confirmation car cela va vider la bibliothèque actuelle (incompatible)
        reply = QMessageBox.question(self.view, 'Reload Model', 
            f"Changing to {new_size} at {new_res}px will clear the current library. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.view.setCursor(Qt.WaitCursor)
            try:
                self.max_size = new_res
                self.model.dino.load_model(new_size) # Ta méthode load_model existante
                self.view.proc_logs.appendPlainText(f"🚀 Model updated to {new_size} - {new_res}px")
                QMessageBox.information(self.view, "Success", "Model reloaded successfully.")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", f"Failed to reload model: {str(e)}")
            finally:
                self.view.setCursor(Qt.ArrowCursor)

    def select_proc_input(self):
        path = QFileDialog.getExistingDirectory(self.view, "Select Images to Scan")
        if path: self.view.lbl_input_path.setText(path)

    def select_proc_output(self):
        path = QFileDialog.getExistingDirectory(self.view, "Select Results Destination")
        if path: self.view.lbl_output_path.setText(path)

    def start_batch_processing(self):
        """
        Initializes and starts the high-speed batch inference process in a separate thread.
        """
        # 1. Validation de la bibliothèque active
        if not self.model.active_library or self.model.active_library.vectors is None:
            QMessageBox.warning(self.view, "Error", "Please load a library containing at least one reference patch.")
            return
        
        # 2. Récupération des dossiers depuis les labels de l'interface
        input_dir = self.view.lbl_input_path.text()
        output_dir = self.view.lbl_output_path.text()
        
        # Vérification de la validité des chemins
        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            QMessageBox.warning(self.view, "Error", "Please select valid Input and Output directories.")
            return

        # 3. Récupération des paramètres de filtrage et d'exportation
        export_hm = self.view.check_export_hm.isChecked()
        threshold = self.view.spin_proc_thresh.value()

        # 4. Configuration du Worker et du Thread
        # On crée le Thread
        self.thread = QThread()

        batch_size = self.view.spin_batch_size.value()
        
        input_dir = self.view.lbl_input_path.text()
        output_dir = self.view.lbl_output_path.text()
        
        # Récupération des paramètres
        export_hm = self.view.check_export_hm.isChecked()
        copy_orig = self.view.check_copy_orig.isChecked() # <--- Nouveau

        self.thread = QThread()
        self.worker = ProcessingWorker(
            model=self.model, 
            lib_vectors=self.model.active_library.vectors, 
            input_dir=input_dir, 
            output_dir=output_dir, 
            max_size=self.max_size,
            export_hm=export_hm,
            copy_orig=copy_orig, # <--- Passage au worker
            threshold=threshold,
            batch_size=batch_size
        )
        
        # On déplace le worker dans le thread pour l'exécution parallèle
        self.worker.moveToThread(self.thread)
        
        # --- Connexions des Signaux ---
        
        # Lancement du calcul au démarrage du thread
        self.thread.started.connect(self.worker.run)
        
        # Mise à jour de la barre de progression
        self.worker.progress.connect(self.view.progress_bar.setValue)
        
        # Affichage des logs dans la console
        self.worker.log.connect(self.view.proc_logs.appendPlainText)
        
        # Nettoyage à la fin du travail
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        # Gestion de l'état du bouton Start (Désactivation/Réactivation)
        self.view.btn_start_proc.setEnabled(False)
        self.thread.finished.connect(lambda: self.view.btn_start_proc.setEnabled(True))
        
        # 5. Lancement effectif
        self.view.proc_logs.appendPlainText(f"--- Starting Batch Scan ---")
        self.view.proc_logs.appendPlainText(f"Threshold: {threshold} | Export: {export_hm}")
        self.thread.start()

    def _refresh_ui(self):
        """Refreshes all UI components based on the current state."""
        self.update_status_info()
        self.update_memory_view()
        if self.view.central_stack.currentIndex() == 1: 
            self.populate_edit_grid()

    def toggle_edit_mode(self):
        """Switches between the Image Explorer and the Library Management view."""
        is_explorer = self.view.central_stack.currentIndex() == 0
        if is_explorer:
            self.populate_edit_grid()
            self.view.central_stack.setCurrentIndex(1)
            self.view.action_edit_lib.setText("🖼️ Back to Image")
        else:
            self.view.central_stack.setCurrentIndex(0)
            self.view.action_edit_lib.setText("⚙️ Manage Library")
            self.update_memory_view()

    # --- AI & RENDERING ---

    def create_heatmap_pixmap(self, vector, img_path, input_size):
        """Generates a heatmap QPixmap overlaying the source image."""
        if not os.path.exists(img_path): return None
        feat, (tw, th) = self.model.dino.get_features(Image.open(img_path).convert('RGB'), input_size)
        with torch.no_grad():
            sim = torch.matmul(feat.reshape(-1, self.model.dino.current_config['dim']), vector.to(self.model.dino.device))
            heatmap = sim.reshape(feat.shape[0], feat.shape[1]).cpu().numpy()
            
        base = QPixmap(img_path).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rgba = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = (cm.jet(np.clip(heatmap, 0, 1))[:,:,:3]*255).astype(np.uint8)
        rgba[:,:,3] = np.where(heatmap > 0.5, 160, 0).astype(np.uint8)
        
        q_img = QImage(rgba.data, heatmap.shape[1], heatmap.shape[0], 4*heatmap.shape[1], QImage.Format_RGBA8888)
        res = base.copy()
        p = QPainter(res)
        p.drawImage(res.rect(), q_img)
        p.end()
        return res

    def display_dynamic_heatmap(self, heatmap, target_widget):
        """Renders a dynamic heatmap with sharp block edges (no interpolation)."""
        if heatmap is None or self.view_local_base is None: return
        rgba = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
        rgba[:,:,:3] = (cm.jet(np.clip(heatmap, 0, 1))[:,:,:3]*255).astype(np.uint8)
        rgba[:,:,3] = np.where(heatmap > self.threshold, 180, 0).astype(np.uint8)
        
        q_img = QImage(rgba.data, heatmap.shape[1], heatmap.shape[0], 4*heatmap.shape[1], QImage.Format_RGBA8888)
        canvas = self.view_local_base.copy()
        p = QPainter(canvas)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False) # Sharp patch edges
        p.drawImage(canvas.rect(), q_img)
        p.end()
        self.set_scaled_pixmap(canvas, target_widget.label_image)

    # --- CLICKS & NAVIGATION ---

    def handle_local_click(self, event):
        """Handles user patch selection on the local image view."""
        label = self.view.view_local.label_image
        pix = label.pixmap()
        if not pix or self.current_features is None: return
        
        # Screen to Image coordinate mapping
        ox, oy = (label.width()-pix.width())//2, (label.height()-pix.height())//2
        rx, ry = (event.position().x()-ox)/pix.width(), (event.position().y()-oy)/pix.height()
        
        if 0 <= rx <= 1 and 0 <= ry <= 1:
            px, py = int(rx * self.view_local_base.width() // 16), int(ry * self.view_local_base.height() // 16)
            hp, wp = self.current_features.shape[:2]
            px, py = min(px, wp-1), min(py, hp-1)
            
            self.last_click_vector, self.last_click_coords = self.current_features[py, px, :], (py, px)
            with torch.no_grad():
                sim = torch.matmul(self.current_features.reshape(-1, self.model.dino.current_config['dim']), self.last_click_vector)
                self.display_dynamic_heatmap(sim.reshape(hp, wp).cpu().numpy(), self.view.view_local)

    def handle_memory_click(self, event):
        """Handles inspection of library matches found in memory."""
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
                
                # 1. Heatmap Preview (Cached)
                hm_path = os.path.join(self.model.active_library.heatmaps_dir, meta.get('heatmap_cache', ''))
                if os.path.exists(hm_path):
                    self.view.label_source_preview.setPixmap(QPixmap(hm_path).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                # 2. Raw Source Image
                raw_path = os.path.join(self.model.active_library.images_dir, meta['image_name'])
                if os.path.exists(raw_path):
                    self.view.label_source_clean.setPixmap(QPixmap(raw_path).scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
                self.view.label_source_info.setText(f"📄 filename: {meta['image_name'][:10]}...")
                self.view.btn_delete_patch.setVisible(True)

    def next_image(self):
        """Saves current patch if selected and moves to the next image."""
        if self.last_click_vector is not None and self.model.active_library:
            img_p = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
            hm_cache = self.create_heatmap_pixmap(self.last_click_vector, img_p, self.max_size)
            self.model.active_library.add_patch(self.last_click_vector, img_p, self.last_click_coords, self.model_size, self.max_size, hm_cache)
        
        if self.model.image_list:
            self.model.current_image_idx = (self.model.current_image_idx + 1) % len(self.model.image_list)
            self.last_click_vector = None
            self.load_current_image()
            self.update_status_info()

    def populate_edit_grid(self):
        """Fills the library edit grid with cached heatmap previews."""
        while self.view.patch_grid.count():
            w = self.view.patch_grid.takeAt(0).widget()
            if w: w.deleteLater()
        if not self.model.active_library: return
        
        from view import PatchCard
        for i, meta in enumerate(self.model.active_library.metadata):
            hm_path = os.path.join(self.model.active_library.heatmaps_dir, meta.get('heatmap_cache', ''))
            if os.path.exists(hm_path):
                self.view.patch_grid.addWidget(PatchCard(QPixmap(hm_path), f"#{i} {meta['image_name'][:12]}", i, self.delete_patch_from_grid), i//4, i%4)

    # --- UTILS ---

    def set_scaled_pixmap(self, pixmap, target_label):
        """Safely scales and sets a pixmap to a target label."""
        if pixmap:
            s = pixmap.scaled(target_label.width(), target_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            target_label.setPixmap(s)

    def load_current_image(self):
        """Loads and processes the currently selected image index."""
        if not self.model.image_list: return
        img_p = os.path.join(self.model.image_folder, self.model.image_list[self.model.current_image_idx])
        self.current_features, (tw, th) = self.model.dino.get_features(Image.open(img_p).convert('RGB'), self.max_size)
        self.view_local_base = QPixmap(img_p).scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.set_scaled_pixmap(self.view_local_base, self.view.view_local.label_image)
        self.set_scaled_pixmap(self.view_local_base, self.view.view_memory.label_image)
        self.update_memory_view()

    def update_memory_view(self):
        """Performs a global search in the active library for matches on the current image."""
        if self.current_features is None:
            # Si on a une base d'image mais pas de features, on affiche l'image brute
            if self.view_local_base:
                self.set_scaled_pixmap(self.view_local_base, self.view.view_memory.label_image)
            return

        # Si une bibliothèque est active, on procède au calcul
        if not self.model.active_library or self.model.active_library.vectors is None:
            if self.view_local_base: 
                self.set_scaled_pixmap(self.view_local_base, self.view.view_memory.label_image)
            return
        with torch.no_grad():
            f = self.current_features.reshape(-1, self.model.dino.current_config['dim'])
            s = torch.matmul(f, self.model.active_library.vectors.to(self.model.dino.device).T)
            best_s, best_i = torch.max(s, dim=1)
            hp, wp = self.current_features.shape[:2]
            self.memory_scores, self.memory_match_indices = best_s.reshape(hp, wp).cpu().numpy(), best_i.reshape(hp, wp).cpu().numpy()
            self.display_dynamic_heatmap(self.memory_scores, self.view.view_memory)

    def update_threshold(self, v):
        """Updates the match threshold and refreshes memory view."""
        self.threshold = v/100.0
        self.view.label_thresh_val.setText(f"🎯 Threshold: {self.threshold:.2f}")
        self.update_memory_view()

    def update_status_info(self):
        """Updates the status bar information."""
        name = os.path.basename(self.model.active_library.lib_path) if self.model.active_library else "-"
        count = len(self.model.active_library.metadata) if self.model.active_library else 0
        self.view.label_lib_name.setText(f"📂 Lib: {name}")
        self.view.label_mem_count.setText(f"📦 {count} patches")

    def open_image_folder(self):
        """Opens a file dialog to select the working image folder."""
        f = QFileDialog.getExistingDirectory(self.view, "Select Images Folder")
        if f: 
            self.model.image_folder = f
            self.model.image_list = sorted([x for x in os.listdir(f) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.model.current_image_idx = 0
            self.load_current_image()

    def delete_patch_from_grid(self, idx):
        """Deletes a patch from the edit grid after confirmation."""
        if QMessageBox.question(self.view, "Delete", "Remove this patch from library?") == QMessageBox.Yes:
            self.model.active_library.remove_patch(idx)
            self.populate_edit_grid()
            self.update_status_info()

    def delete_inspected_patch(self):
        """Deletes the currently inspected patch from the active library."""
        if self.current_inspected_idx is not None: 
            self.model.active_library.remove_patch(self.current_inspected_idx)
            self.view.btn_delete_patch.setVisible(False)
            self._refresh_ui()

    def open_library(self):
        """Opens an existing Patch Library."""
        p = QFileDialog.getExistingDirectory(self.view, "Open Library")
        if p: 
            self.model.active_library = PatchLibrary(p)
            self._refresh_ui()

    def create_new_library(self):
        """Creates a new Patch Library folder."""
        p = QFileDialog.getExistingDirectory(self.view, "New Library")
        if p: 
            self.model.active_library = PatchLibrary(p)
            self._refresh_ui()

    def cancel_last_click(self):
        """Resets the last user selection on the local image."""
        self.last_click_vector = None
        self.set_scaled_pixmap(self.view_local_base, self.view.view_local.label_image)

    def prev_image(self):
        """Navigates to the previous image in the folder."""
        if self.model.image_list:
            self.model.current_image_idx = (self.model.current_image_idx - 1) % len(self.model.image_list)
            self.load_current_image()

    def delete_current_library(self):
        """Completely deletes the active library folder from disk."""
        if self.model.active_library and QMessageBox.critical(self.view, "⚠️ Delete Library", "Destroy the entire library folder?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            import shutil
            shutil.rmtree(self.model.active_library.lib_path)
            self.model.active_library = None
            self._refresh_ui()

    def merge_libraries_dialog(self):
        """Handles the merging of two libraries into a third one."""
        a = QFileDialog.getExistingDirectory(self.view, "Select Library A")
        b = QFileDialog.getExistingDirectory(self.view, "Select Library B")
        o = QFileDialog.getExistingDirectory(self.view, "Select Output Destination")
        if a and b and o: 
            PatchLibrary.merge(a, b, o)
            QMessageBox.information(self.view, "Success", "Libraries merged successfully.")

    def show_about_page(self):
        """Populates and displays the About/Information page."""
        
        # 1. Technical Info
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        tech_info = (
            f"Model Architecture: {self.model_size.upper()}\n"
            f"Inference Resolution: {self.max_size}px\n"
            f"Processing Unit: {device}"
        )
        self.view.lbl_about_software.setText(tech_info)
        
        # 2. Author Info
        self.view.lbl_author_name.setText("Name: Rémi Pérenne")
        self.view.lbl_author_email.setText("Email: remi.perenne@etu.minesparis.psl.eu")
        bio_text = (
            "Hi! I am currently a student at Mines Paris - PSL, specializing in AI and applied math, "
            "I built this tool primarily to help biologists quilckly annotate and extract meaningful information from huge amount of camera trap images without having to train a yolo model on thousands of annotated samples but feel free to use it for other purposes.\n\n"
            "If you have any questions, suggestions or want to collaborate on improving the software, please reach out to me!\n\n"
            "This software is under license GPLv3 and the source code is available on GitHub: https://github.com/theDude29/dinopatch_manager"
        )
        self.view.lbl_author_bio_content.setText(bio_text)
        
        # Switch to About View
        self.view.central_stack.setCurrentIndex(2)
        self.view.action_edit_lib.setText("🖼️ Back to Explorer")

    def show_how_to_page(self):
        """Populates the help section with detailed usage instructions."""
        
        help_content = """
        <h2 style='color: #58a6ff;'> Quick Start Guide</h2>
        
        <h3 style='color: #a5d6ff;'>1. Configuration (config.json)</h3>
        Before launching, ensure your <b>config.json</b> is set correctly:
        <ul>
            <li><b>model_size</b>: "small", "base", or "large" (DINOv3 architecture).</li>
            <li><b>max_size</b>: The processing resolution (e.g., 672 or 1024).</li>
            <li><b>path_repo_dino</b>: Absolute path to your local DINOv3 repository.</li>
        </ul>

        <h3 style='color: #a5d6ff;'>2. Interface Overview</h3>
        <ul>
            <li><b>Local View (Left)</b>: Your workspace to select new reference patches.</li>
            <li><b>Memory View (Center)</b>: Displays real-time similarity between the current image and your entire saved library.</li>
            <li><b>Inspector (Right)</b>: Shows the specific patch from the library that matches the current selection.</li>
        </ul>

        <h3 style='color: #a5d6ff;'>3. How to Click & Select</h3>
        <ul>
            <li><b>Define a Reference</b>: In the <b>Local View</b>, click on any object. A heatmap will appear showing all similar areas in the image.</li>
            <li><b>Save to Library</b>: Click <b>'Next'</b> to save this patch's signature and move to the next image.</li>
            <li><b>Inspect Matches</b>: In the <b>Memory View</b>, click on a highlighted area (heatmap) to see which library image it corresponds to in the Inspector.</li>
        </ul>

        <h3 style='color: #a5d6ff;'>4. Managing your Library</h3>
        Use the <b>'Manage Library'</b> menu to delete unwanted patches or merge two different libraries into a single master collection.
        
        <h3 style='color: #a5d6ff;'>5. File & Library Management</h3>
        All file operations are located in the <b>'Library'</b> top menu:
        <ul>
            <li><b>New Library</b>: Creates a new workspace. Select an <b>empty folder</b>. The software will automatically create the internal structure (images, heatmaps, and vector files).</li>
            <li><b>Open Library</b>: Loads an existing collection. Select the folder you previously created or downloaded.</li>
            <li><b>Open Images Folder</b>: Loads the dataset you want to analyze. Select the directory containing your JPG or PNG images. Use the navigation arrows to browse through them.</li>
        </ul>

        <h3 style='color: #a5d6ff;'>6. RaptorScan (Processing)</h3>
        This section allows for the automatic analysis of thousands of images based on your saved library patches. RaptorVision scans each image, calculates the highest similarity scores, and exports the data without manual intervention.

        <h4 style='color: #a5d6ff;'>6.1 Directory Configuration</h4>
        <ul>
            <li><b>Input Folder</b>: The directory containing the images to be analyzed (supported: .jpg, .png, .jpeg).</li>
            <li><b>Output Folder</b>: The destination for the log file (<b>matching_results.txt</b>) and all exported visual data.</li>
        </ul>

        <h4 style='color: #a5d6ff;'>6.2 Engine Parameters</h4>
        <ul>
            <li><b>Min Score (Threshold)</b>: Sets the confidence level (0.0 to 1.0). Only results higher than this score will trigger a heatmap export or file copy. A score of <b>0.70</b> is a recommended baseline for strict detection.</li>
            <li><b>Batch Size</b>: The number of images processed simultaneously by the GPU.
                <ul>
                    <li><b>Value 1 (Default)</b>: Mandatory if your dataset contains images with different aspect ratios (mixed Portrait/Landscape).</li>
                    <li><b>Value > 1</b>: Use only if all images have identical dimensions. This significantly increases speed on high-end NVIDIA GPUs.</li>
                </ul>
            </li>
        </ul>

        <h4 style='color: #a5d6ff;'>6.3 Export & Automation Options</h4>
        <ul>
            <li><b>Export Heatmaps</b>: Generates a thermal image showing the precise location of the detection for each "Top Match". Filenames are saved as <i>heatmap_filename_score.png</i>.</li>
            <li><b>Copy Top Matches</b>: Automatically copies the original image to the output folder if it exceeds the threshold. Files are prefixed with the score for easy sorting in your file explorer.</li>
        </ul>

        <h4 style='color: #a5d6ff;'>6.4 Interpreting Results</h4>
        At the end of each session, a <b>matching_results.txt</b> file is generated. It contains a full list of scanned images and their best matching scores. This file can be imported into Excel or Python for statistical analysis.

        <h3 style='color: #a5d6ff;'>7. Performance & Hardware</h3>
        RaptorVision adapts its power based on your hardware configuration:
        <ul>
            <li><b>GPU Mode (CUDA)</b>: Uses NVIDIA Tensor cores. This is the fastest mode, recommended for datasets larger than 100 images. Processing usually takes 50-200ms per image.</li>
            <li><b>CPU Mode</b>: If no GPU is detected, the software switches to the processor. Analysis will be <b>10x to 50x slower</b>.</li>
        </ul>
        
        """
        
        self.view.lbl_how_to_text.setText(help_content)
        self.view.central_stack.setCurrentIndex(3)
        self.view.action_edit_lib.setText("🖼️ Back to Explorer")