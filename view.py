from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QMenuBar, QMenu, QFrame, QLabel, 
                             QStackedWidget, QGridLayout, QScrollArea, QSizePolicy, QStatusBar, QSlider)
from PySide6.QtCore import Qt

class PatchImageWidget(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain) # Plain pour éviter les ombres inégales
        self.setLineWidth(1)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Supprime les marges internes
        layout.setSpacing(5)

        self.label_title = QLabel(title)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-weight: bold; background-color: #333; color: white; padding: 5px;")
        
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        # On force le label à ne pas s'étendre tout seul pour garder le contrôle
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_image.setStyleSheet("background-color: #1a1a1a;")
        
        layout.addWidget(self.label_title)
        layout.addWidget(self.label_image)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINO Patch Manager Pro")
        self.resize(1400, 850) # Légèrement plus large pour l'inspecteur

        # 1. Barre de Navigation (Menu)
        self._create_menubar()

        # 2. Widget Central Empilé (Explorer / Editer)
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # --- MODE EXPLORATEUR (Index 0) ---
        self.explorer_view = QWidget()
        self.explorer_layout = QVBoxLayout(self.explorer_view)
        
        # A. Zone de contenu principal (Images + Inspecteur)
        main_content_layout = QHBoxLayout()
        
        # Sous-zone : Les deux images (Local / Mémoire)
        self.view_local = PatchImageWidget("VUE LOCALE (CLIC)")
        self.view_memory = PatchImageWidget("VUE MÉMOIRE (GLOBAL)")
        
        images_layout = QHBoxLayout()
        images_layout.addWidget(self.view_local, 1) 
        images_layout.addWidget(self.view_memory, 1)
        
        # Sous-zone : Panneau de l'Inspecteur (à droite)
        self.inspector_panel = QFrame()
        self.inspector_panel.setFixedWidth(280)
        self.inspector_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.inspector_panel.setStyleSheet("background-color: #252525; border-left: 1px solid #444;")
        
        inspector_layout = QVBoxLayout(self.inspector_panel)
        
        label_inspect_title = QLabel("🔍 INSPECTEUR DE MATCH")
        label_inspect_title.setStyleSheet("font-weight: bold; color: #aaa; margin-bottom: 10px;")
        label_inspect_title.setAlignment(Qt.AlignCenter)
        
        self.label_source_preview = QLabel("Cliquez sur un patch\nde la mémoire")
        self.label_source_preview.setFixedSize(256, 256) # Taille carrée pour la vignette
        self.label_source_preview.setAlignment(Qt.AlignCenter)
        self.label_source_preview.setStyleSheet("border: 2px dashed #444; color: #666; background-color: #1a1a1a;")
        
        self.label_source_info = QLabel("")
        self.label_source_info.setWordWrap(True)
        self.label_source_info.setStyleSheet("font-size: 11px; color: #888; margin-top: 10px;")

        inspector_layout.addWidget(label_inspect_title)
        inspector_layout.addWidget(self.label_source_preview)
        inspector_layout.addWidget(self.label_source_info)
        inspector_layout.addStretch() # Pousse tout vers le haut

        # Assemblage Images + Inspecteur
        main_content_layout.addLayout(images_layout, 4) # 80% de largeur
        main_content_layout.addWidget(self.inspector_panel, 1) # 20% de largeur

        # B. Barre de contrôle inférieure (Navigation + Slider)
        control_layout = QHBoxLayout()
        self.btn_prev = QPushButton("⬅️ Précédent")
        self.btn_cancel = QPushButton("🔄 Annuler")
        self.btn_next = QPushButton("Suivant ➡️")
        self.btn_next.setStyleSheet("background-color: #2d5a27; font-weight: bold;")

        # Slider de Threshold
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(60)
        self.slider_threshold.setFixedWidth(150)
        
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_cancel)
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Sensibilité :"))
        control_layout.addWidget(self.slider_threshold)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_next)

        # Finalisation Explorer Layout
        self.explorer_layout.addLayout(main_content_layout, 1)
        self.explorer_layout.addLayout(control_layout)
        
        # --- MODE ÉDITION (Index 1) ---
        self.edit_view = QWidget()
        self.edit_layout = QVBoxLayout(self.edit_view)
        
        edit_actions = QHBoxLayout()
        self.btn_merge = QPushButton("🔗 Fusionner Bibliothèques")
        self.btn_delete_lib = QPushButton("🗑️ Supprimer cette Librairie")
        edit_actions.addWidget(self.btn_merge)
        edit_actions.addWidget(self.btn_delete_lib)
        
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.patch_grid = QGridLayout(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.edit_layout.addLayout(edit_actions)
        self.edit_layout.addWidget(self.scroll_area)

        # Ajout des vues au Stack
        self.central_stack.addWidget(self.explorer_view)
        self.central_stack.addWidget(self.edit_view)

        # 3. Barre d'État (Status Bar)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.label_mem_count = QLabel("📦 Mémoire : 0 patchs")
        self.label_thresh_val = QLabel("🎯 Seuil : 0.60")
        
        self.status_bar.addPermanentWidget(self.label_mem_count)
        self.status_bar.addPermanentWidget(self.label_thresh_val)

    def _create_menubar(self):
        menubar = self.menuBar()
        
        # Menu Librairie
        lib_menu = menubar.addMenu("📁 Librairie")
        self.action_new_lib = lib_menu.addAction("Créer une librairie")
        self.action_open_lib = lib_menu.addAction("Ouvrir une librairie")
        self.action_edit_lib = lib_menu.addAction("Modifier une librairie")
        lib_menu.addSeparator()
        self.action_save = lib_menu.addAction("Enregistrer")
        self.action_save_as = lib_menu.addAction("Enregistrer sous...")

        # Menu Images
        img_menu = menubar.addMenu("🖼️ Images")
        self.action_open_folder = img_menu.addAction("Ouvrir dossier image")

    def show_explorer(self):
        self.central_stack.setCurrentIndex(0)

    def show_editor(self):
        self.central_stack.setCurrentIndex(1)
