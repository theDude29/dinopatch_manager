from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFrame, QLabel, QStackedWidget, 
                             QGridLayout, QScrollArea, QSizePolicy, QStatusBar, 
                             QSlider, QSplitter, QGroupBox)
from PySide6.QtCore import Qt

class PatchCard(QGroupBox):
    def __init__(self, heatmap_pixmap, info_text, index, delete_callback):
        super().__init__()
        self.setStyleSheet("background-color: #2a2a2a; border: 1px solid #444; border-radius: 4px;")
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        lbl_info = QLabel(info_text)
        lbl_info.setStyleSheet("font-size: 10px; font-weight: bold; color: #aaa; border: none;")
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(22, 22)
        btn_del.setStyleSheet("background-color: #721c24; color: white; border: none;")
        btn_del.clicked.connect(lambda: delete_callback(index))
        header.addWidget(lbl_info); header.addStretch(); header.addWidget(btn_del)
        lbl_img = QLabel()
        lbl_img.setPixmap(heatmap_pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl_img.setAlignment(Qt.AlignCenter)
        lbl_img.setStyleSheet("background: #000; border: 1px solid #111;")
        layout.addLayout(header); layout.addWidget(lbl_img)

class PatchImageWidget(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label_title = QLabel(title)
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("font-weight: bold; background-color: #333; color: white; padding: 6px;")
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(1, 1)
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_image.setStyleSheet("background-color: #121212;")
        layout.addWidget(self.label_title); layout.addWidget(self.label_image, 1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINO Patch Manager Pro")
        self.resize(1450, 900)
        self._create_menubar()

        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        # --- EXPLORATEUR ---
        self.explorer_view = QWidget()
        exp_layout = QVBoxLayout(self.explorer_view)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.view_local, self.view_memory = PatchImageWidget("VUE LOCALE"), PatchImageWidget("VUE MÉMOIRE")
        
        # PANNEAU INSPECTEUR (À DROITE)
        self.inspector_panel = QFrame()
        self.inspector_panel.setStyleSheet("background-color: #1e1e1e; border-left: 1px solid #333;")
        ins_layout = QVBoxLayout(self.inspector_panel)
        
        # Section Heatmap
        ins_layout.addWidget(QLabel("🎯 HEATMAP (MATCH)"), alignment=Qt.AlignCenter)
        self.label_source_preview = QLabel("Sélectionnez un point")
        self.label_source_preview.setFixedSize(256, 256)
        self.label_source_preview.setStyleSheet("border: 1px solid #444; background: #000;")
        self.label_source_preview.setAlignment(Qt.AlignCenter)
        ins_layout.addWidget(self.label_source_preview)
        
        # Section Image Brute
        ins_layout.addWidget(QLabel("🖼️ IMAGE SOURCE"), alignment=Qt.AlignCenter)
        self.label_source_clean = QLabel("Original")
        self.label_source_clean.setFixedSize(256, 256)
        self.label_source_clean.setStyleSheet("border: 1px solid #444; background: #000;")
        self.label_source_clean.setAlignment(Qt.AlignCenter)
        ins_layout.addWidget(self.label_source_clean)

        self.label_source_info = QLabel("")
        self.label_source_info.setStyleSheet("color: #999; font-size: 11px;")
        self.btn_delete_patch = QPushButton("🗑️ Supprimer Patch")
        self.btn_delete_patch.setVisible(False)
        self.btn_delete_patch.setStyleSheet("background-color: #500; color: white; padding: 10px; font-weight: bold;")

        ins_layout.addWidget(self.label_source_info)
        ins_layout.addWidget(self.btn_delete_patch); ins_layout.addStretch()

        self.main_splitter.addWidget(self.view_local); self.main_splitter.addWidget(self.view_memory); self.main_splitter.addWidget(self.inspector_panel)
        self.main_splitter.setStretchFactor(0, 2); self.main_splitter.setStretchFactor(1, 2); self.main_splitter.setStretchFactor(2, 1)

        nav_bar = QFrame()
        nav_bar.setFixedHeight(60); nav_bar.setStyleSheet("background: #252525; border-top: 1px solid #333;")
        nav_layout = QHBoxLayout(nav_bar)
        self.btn_prev, self.btn_cancel, self.btn_next = QPushButton("⬅️ Précédent"), QPushButton("🔄 Annuler"), QPushButton("Suivant ➡️")
        self.btn_next.setStyleSheet("background-color: #2d5a27; font-weight: bold; padding: 8px 25px;")
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100); self.slider_threshold.setValue(60); self.slider_threshold.setFixedWidth(160)
        nav_layout.addWidget(self.btn_prev); nav_layout.addWidget(self.btn_cancel); nav_layout.addStretch()
        nav_layout.addWidget(QLabel("Seuil :")); nav_layout.addWidget(self.slider_threshold); nav_layout.addStretch()
        nav_layout.addWidget(self.btn_next)
        exp_layout.addWidget(self.main_splitter, 1); exp_layout.addWidget(nav_bar)

        # --- ÉDITION ---
        self.edit_view = QWidget()
        edi_layout = QVBoxLayout(self.edit_view)
        act_layout = QHBoxLayout()
        self.btn_merge, self.btn_delete_lib = QPushButton("🔗 Fusionner Librairies"), QPushButton("🗑️ Supprimer Librairie")
        self.btn_merge.setStyleSheet("background-color: #155724; color: white; padding: 12px;")
        self.btn_delete_lib.setStyleSheet("background-color: #721c24; color: white; padding: 12px;")
        act_layout.addWidget(self.btn_merge); act_layout.addWidget(self.btn_delete_lib)
        self.scroll_area = QScrollArea()
        self.scroll_content = QWidget()
        self.patch_grid = QGridLayout(self.scroll_content)
        self.scroll_area.setWidgetResizable(True); self.scroll_area.setWidget(self.scroll_content)
        edi_layout.addLayout(act_layout); edi_layout.addWidget(self.scroll_area)

        self.central_stack.addWidget(self.explorer_view); self.central_stack.addWidget(self.edit_view)

        # Barre d'état
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.label_lib_name, self.label_mem_count, self.label_thresh_val = QLabel("📂 Lib : -"), QLabel("📦 0"), QLabel("🎯 0.60")
        for w in [self.label_lib_name, self.label_mem_count, self.label_thresh_val]:
            w.setStyleSheet("margin-right: 20px; font-weight: bold; color: #ddd;")
            self.status_bar.addPermanentWidget(w)

    def _create_menubar(self):
        menubar = self.menuBar()
        lib_menu = menubar.addMenu("📁 Librairie")
        self.action_open_folder = lib_menu.addAction("Ouvrir dossier images")
        lib_menu.addSeparator()
        self.action_new_lib = lib_menu.addAction("Nouvelle librairie")
        self.action_open_lib = lib_menu.addAction("Ouvrir librairie")
        lib_menu.addSeparator()
        self.action_edit_lib = lib_menu.addAction("⚙️ Gérer la librairie")