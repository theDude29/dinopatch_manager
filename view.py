from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFrame, QLabel, QStackedWidget, 
                             QGridLayout, QScrollArea, QSizePolicy, QStatusBar, 
                             QSlider, QSplitter)
from PySide6.QtCore import Qt

# À ajouter dans view.py
from PySide6.QtWidgets import QGroupBox

class PatchCard(QGroupBox):
    def __init__(self, heatmap_pixmap, info_text, index, delete_callback):
        super().__init__()
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #444; color: white;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # En-tête avec index et bouton supprimer
        header = QHBoxLayout()
        lbl_info = QLabel(info_text)
        lbl_info.setStyleSheet("font-size: 10px; font-weight: bold; border: none;")
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(20, 20)
        btn_del.setStyleSheet("background-color: #500; border: none; font-weight: bold;")
        btn_del.clicked.connect(lambda: delete_callback(index))
        
        header.addWidget(lbl_info)
        header.addStretch()
        header.addWidget(btn_del)
        
        # Image unique avec Heatmap
        self.lbl_img = QLabel()
        self.lbl_img.setPixmap(heatmap_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setStyleSheet("border: 1px solid #111; background: #000;")

        layout.addLayout(header)
        layout.addWidget(self.lbl_img)

# REMPLACEZ la classe PatchImageWidget entière :
class PatchImageWidget(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(1)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label_title = QLabel(title)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-weight: bold; background-color: #333; color: white; padding: 5px;")
        
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        
        # --- CRUCIAL : Autorise le label à être plus petit que l'image ---
        self.label_image.setMinimumSize(1, 1) 
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_image.setStyleSheet("background-color: #1a1a1a;")
        
        layout.addWidget(self.label_title)
        layout.addWidget(self.label_image, 1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINO Patch Manager Pro")
        self.setMinimumSize(1100, 750)
        self.resize(1400, 850)

        # 1. Barre de Navigation (Menu)
        self._create_menubar()

        # 2. Widget Central Empilé (pour switcher entre Explorer et Editer)
        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # ==========================================
        # --- MODE EXPLORATEUR (Index 0) ---
        # ==========================================
        self.explorer_view = QWidget()
        self.explorer_layout = QVBoxLayout(self.explorer_view)
        self.explorer_layout.setContentsMargins(5, 5, 5, 5)

        # SPLITTER PRINCIPAL (Ratio 40/40/20)
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Vues Images
        self.view_local = PatchImageWidget("VUE LOCALE (CLIC)")
        self.view_memory = PatchImageWidget("VUE MÉMOIRE (GLOBAL)")
        
        # Panneau Inspecteur (À droite)
        self.inspector_panel = QFrame()
        self.inspector_panel.setStyleSheet("background-color: #222; border-left: 1px solid #444;")
        inspect_layout = QVBoxLayout(self.inspector_panel)
        
        self.label_inspect_title = QLabel("🔍 INSPECTEUR")
        self.label_inspect_title.setStyleSheet("font-weight: bold; color: #aaa; padding: 5px;")
        self.label_inspect_title.setAlignment(Qt.AlignCenter)

        self.label_source_preview = QLabel("Pas de sélection")
        self.label_source_preview.setFixedSize(224, 224) 
        self.label_source_preview.setStyleSheet("border: 1px solid #444; background: #111;")
        self.label_source_preview.setAlignment(Qt.AlignCenter)
        
        self.label_source_info = QLabel("")
        self.label_source_info.setStyleSheet("font-size: 10px; color: #999; margin-top: 10px;")
        self.label_source_info.setWordWrap(True)

        self.btn_delete_patch = QPushButton("🗑️ Supprimer ce Patch")
        self.btn_delete_patch.setVisible(False) # Caché par défaut
        self.btn_delete_patch.setStyleSheet("background-color: #500; color: white; padding: 8px; font-weight: bold;")

        inspect_layout.addWidget(self.label_inspect_title)
        inspect_layout.addWidget(self.label_source_preview)
        inspect_layout.addWidget(self.label_source_info)
        inspect_layout.addWidget(self.btn_delete_patch)
        inspect_layout.addStretch()

        # Ajout au Splitter
        self.main_splitter.addWidget(self.view_local)      # Index 0
        self.main_splitter.addWidget(self.view_memory)     # Index 1
        self.main_splitter.addWidget(self.inspector_panel) # Index 2
        
        # Configuration des proportions (40% / 40% / 20%)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setStretchFactor(2, 1)

        # Barre de contrôle inférieure (Boutons + Slider)
        control_bar = QFrame()
        control_bar.setFixedHeight(60)
        control_bar.setStyleSheet("background: #282828; border-top: 1px solid #444;")
        control_layout = QHBoxLayout(control_bar)
        
        self.btn_prev = QPushButton("⬅️ Précédent")
        self.btn_cancel = QPushButton("🔄 Annuler")
        self.btn_next = QPushButton("Suivant ➡️")
        self.btn_next.setFixedWidth(120)
        self.btn_next.setStyleSheet("background-color: #2d5a27; font-weight: bold; color: white;")
        
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(60)
        self.slider_threshold.setFixedWidth(150)

        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_cancel)
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Sensibilité (Seuil) :"))
        control_layout.addWidget(self.slider_threshold)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_next)

        # Assemblage Explorer
        self.explorer_layout.addWidget(self.main_splitter, 1)
        self.explorer_layout.addWidget(control_bar)

        # ==========================================
        # --- MODE ÉDITION (Index 1) ---
        # ==========================================
        self.edit_view = QWidget()
        self.edit_layout = QVBoxLayout(self.edit_view)
        
        # Actions de bibliothèque
        edit_actions_layout = QHBoxLayout()
        self.btn_merge = QPushButton("🔗 Fusionner deux Librairies")
        self.btn_delete_lib = QPushButton("🗑️ Supprimer la Librairie active")
        
        # Styles des boutons (On les applique APRÈS la création)
        self.btn_merge.setStyleSheet("background-color: #155724; color: white; font-weight: bold; padding: 10px;")
        self.btn_delete_lib.setStyleSheet("background-color: #721c24; color: white; font-weight: bold; padding: 10px;")
        
        edit_actions_layout.addWidget(self.btn_merge)
        edit_actions_layout.addWidget(self.btn_delete_lib)
        
        # Grille de visualisation des patchs
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.patch_grid = QGridLayout(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.edit_layout.addLayout(edit_actions_layout)
        self.edit_layout.addWidget(self.scroll_area)

        # Ajout des vues au Stack
        self.central_stack.addWidget(self.explorer_view) # Index 0
        self.central_stack.addWidget(self.edit_view)     # Index 1

        # ==========================================
        # --- BARRE D'ÉTAT (Status Bar) ---
        # ==========================================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.label_mem_count = QLabel("📦 Mémoire : 0 patchs")
        self.label_thresh_val = QLabel("🎯 Seuil : 0.60")
        
        self.status_bar.addPermanentWidget(self.label_mem_count)
        self.status_bar.addPermanentWidget(self.label_thresh_val)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Nouveau label pour le nom de la librairie
        self.label_lib_name = QLabel("📂 Lib : Aucune")
        self.label_lib_name.setStyleSheet("font-weight: bold; color: #3498db; margin-right: 20px;")

        self.label_mem_count = QLabel("📦 Mémoire : 0 patchs")
        self.label_thresh_val = QLabel("🎯 Seuil : 0.60")

        # On ajoute le nom de la lib en premier (à gauche des infos permanentes)
        self.status_bar.addPermanentWidget(self.label_lib_name)
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
