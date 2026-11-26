import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QPushButton, QSlider, QLabel, QGroupBox, QComboBox,
    QFrame, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg


SAMPLE_SCENARIOS = {
    "Laut Banda (Tuna Sirip Kuning)": {
        "max_depth": 150.0, "seabed_base": 125.0, "surface_temp": 29.2,
        "thermocline_depth": 65.0, "deep_temp": 14.5, "salinity": 34.6,
        "species": "Thunnus albacares (Tuna)", "mean_weight_kg": 3.8, "ts_ref": -33.5,
    },
    "Selat Makassar (Ikan Layang - Pelagis)": {
        "max_depth": 100.0, "seabed_base": 78.0, "surface_temp": 28.8,
        "thermocline_depth": 45.0, "deep_temp": 18.0, "salinity": 33.8,
        "species": "Decapterus spp. (Layang)", "mean_weight_kg": 0.14, "ts_ref": -44.2,
    },
    "Laut Arafura (Kakap Merah - Demersal)": {
        "max_depth": 80.0, "seabed_base": 62.0, "surface_temp": 28.1,
        "thermocline_depth": 35.0, "deep_temp": 21.5, "salinity": 34.2,
        "species": "Lutjanus spp. (Kakap Merah)", "mean_weight_kg": 1.6, "ts_ref": -36.8,
    }
}


class InstructorSonarStation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Instructor Workstation - Simulasi Echosounder, Radar 360° & Navigasi Kapal")
        self.resize(1520, 860)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0b131e; color: #d8e6f3; font-family: 'Segoe UI', Arial; }
            QGroupBox { border: 1px solid #1e354d; border-radius: 6px; margin-top: 8px; font-weight: bold; color: #5bc0eb; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #16324f; border: 1px solid #2a5a8a; padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #1f4870; }
            QPushButton#danger { background-color: #5c1a24; border: 1px solid #9e2a3b; }
            QPushButton#danger:hover { background-color: #7d2230; }
            QComboBox, QDoubleSpinBox { background-color: #13283e; border: 1px solid #2a5a8a; padding: 3px 6px; border-radius: 3px; color: #00f5d4; }
            QLabel { font-size: 12px; }
        """)

        # State Kapal (Koordinat Dunia dalam Meter)
        self.ship_x = 0.0
        self.ship_y = -150.0
        self.ship_heading = 0.0     # 0° = Utara (+Y), 90° = Timur (+X)
        self.ship_speed_kts = 6.0   # Kecepatan kapal (knot)
        self.ship_track_x = [self.ship_x]
        self.ship_track_y = [self.ship_y]

        # Parameter Akustik & Simulasi
        self.current_scenario_name = list(SAMPLE_SCENARIOS.keys())[0]
        self.scenario = SAMPLE_SCENARIOS[self.current_scenario_name]
        self.depth_bins = 320
        self.history_len = 380
        self.radar_range_m = 250.0
        self.sweep_angle = 0.0
        self.time_step = 0
        self.dt = 0.05  # 50 ms per tick
        self.is_running = True

        self.echogram_data = np.full((self.history_len, self.depth_bins), -85.0)
        self.schools = []

        self.init_ui()
        self.load_scenario(self.current_scenario_name)

        # Penempatan skenario awal: 2 gerombolan di jalur lintasan kapal
        self.add_fish_school(x=0.0, y=-40.0, depth=45.0, heading=90.0, speed_kts=1.5, density=0.75)
        self.add_fish_school(x=35.0, y=90.0, depth=70.0, heading=240.0, speed_kts=2.0, density=0.80)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system)
        self.timer.start(int(self.dt * 1000))

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # =================================================================
        # PANEL KIRI: KONTROL INSTRUKTUR (KAPAL & EDITOR SKENARIO IKAN)
        # =================================================================
        instructor_panel = QVBoxLayout()
        instructor_panel.setSpacing(8)

        # 1. Dataset Oseanografi & Target Spesies
        grp_env = QGroupBox("A. Parameter Perairan & Kontrol")
        env_layout = QVBoxLayout(grp_env)
        
        self.combo_sample = QComboBox()
        self.combo_sample.addItems(list(SAMPLE_SCENARIOS.keys()))
        self.combo_sample.currentTextChanged.connect(self.load_scenario)
        env_layout.addWidget(QLabel("Dataset Oseanografi:"))
        env_layout.addWidget(self.combo_sample)

        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("Pause Simulasi")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_reset_ship = QPushButton("Reset Kapal")
        self.btn_reset_ship.clicked.connect(self.reset_ship_position)
        btn_row.addWidget(self.btn_pause)
        btn_row.addWidget(self.btn_reset_ship)
        env_layout.addLayout(btn_row)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Sonar Gain:"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-15, 15)
        self.slider_gain.setValue(0)
        self.lbl_gain = QLabel("0 dB")
        self.slider_gain.valueChanged.connect(lambda v: self.lbl_gain.setText(f"{v} dB"))
        gain_row.addWidget(self.slider_gain)
        gain_row.addWidget(self.lbl_gain)
        env_layout.addLayout(gain_row)
        instructor_panel.addWidget(grp_env)

        # 2. Kendali Navigasi Kapal (Helm)
        grp_ship = QGroupBox("B. Kendali Gerak Kapal (Helm)")
        ship_layout = QGridLayout(grp_ship)

        ship_layout.addWidget(QLabel("Haluan (Heading °):"), 0, 0)
        self.spin_ship_hdg = QDoubleSpinBox()
        self.spin_ship_hdg.setRange(0.0, 359.0)
        self.spin_ship_hdg.setWrapping(True)
        self.spin_ship_hdg.setValue(self.ship_heading)
        self.spin_ship_hdg.valueChanged.connect(self.on_ship_control_changed)
        ship_layout.addWidget(self.spin_ship_hdg, 0, 1)

        self.slider_ship_hdg = QSlider(Qt.Orientation.Horizontal)
        self.slider_ship_hdg.setRange(0, 359)
        self.slider_ship_hdg.setValue(int(self.ship_heading))
        self.slider_ship_hdg.valueChanged.connect(lambda v: self.spin_ship_hdg.setValue(float(v)))
        ship_layout.addWidget(self.slider_ship_hdg, 1, 0, 1, 2)

        ship_layout.addWidget(QLabel("Kecepatan (Knot):"), 2, 0)
        self.spin_ship_spd = QDoubleSpinBox()
        self.spin_ship_spd.setRange(0.0, 20.0)
        self.spin_ship_spd.setSingleStep(0.5)
        self.spin_ship_spd.setValue(self.ship_speed_kts)
        self.spin_ship_spd.valueChanged.connect(self.on_ship_control_changed)
        ship_layout.addWidget(self.spin_ship_spd, 2, 1)

        self.lbl_ship_pos = QLabel("Posisi: X=0.0m, Y=-150.0m")
        self.lbl_ship_pos.setStyleSheet("color: #ffbe0b; font-weight: bold;")
        ship_layout.addWidget(self.lbl_ship_pos, 3, 0, 1, 2)
        instructor_panel.addWidget(grp_ship)

        # 3. Editor Penempatan Gerombolan Ikan
        grp_fish = QGroupBox("C. Penempatan Gerombolan Ikan")
        fish_layout = QGridLayout(grp_fish)

        self.chk_click_spawn = QCheckBox("Klik pada Peta Taktis untuk menempatkan ikan")
        self.chk_click_spawn.setChecked(True)
        self.chk_click_spawn.setStyleSheet("color: #00f5d4;")
        fish_layout.addWidget(self.chk_click_spawn, 0, 0, 1, 2)

        fish_layout.addWidget(QLabel("Koordinat X (m):"), 1, 0)
        self.spin_fish_x = QDoubleSpinBox()
        self.spin_fish_x.setRange(-500.0, 500.0)
        self.spin_fish_x.setValue(0.0)
        fish_layout.addWidget(self.spin_fish_x, 1, 1)

        fish_layout.addWidget(QLabel("Koordinat Y (m):"), 2, 0)
        self.spin_fish_y = QDoubleSpinBox()
        self.spin_fish_y.setRange(-500.0, 500.0)
        self.spin_fish_y.setValue(20.0)
        fish_layout.addWidget(self.spin_fish_y, 2, 1)

        fish_layout.addWidget(QLabel("Kedalaman (m):"), 3, 0)
        self.spin_fish_depth = QDoubleSpinBox()
        self.spin_fish_depth.setRange(5.0, 140.0)
        self.spin_fish_depth.setValue(45.0)
        fish_layout.addWidget(self.spin_fish_depth, 3, 1)

        fish_layout.addWidget(QLabel("Arah Gerak Ikan (°):"), 4, 0)
        self.spin_fish_hdg = QDoubleSpinBox()
        self.spin_fish_hdg.setRange(0.0, 359.0)
        self.spin_fish_hdg.setWrapping(True)
        self.spin_fish_hdg.setValue(90.0)
        fish_layout.addWidget(self.spin_fish_hdg, 4, 1)

        fish_layout.addWidget(QLabel("Kecepatan Ikan (Knot):"), 5, 0)
        self.spin_fish_spd = QDoubleSpinBox()
        self.spin_fish_spd.setRange(0.0, 10.0)
        self.spin_fish_spd.setSingleStep(0.5)
        self.spin_fish_spd.setValue(1.5)
        fish_layout.addWidget(self.spin_fish_spd, 5, 1)

        fish_layout.addWidget(QLabel("Kepadatan (0.1 - 1.0):"), 6, 0)
        self.spin_fish_dens = QDoubleSpinBox()
        self.spin_fish_dens.setRange(0.1, 1.0)
        self.spin_fish_dens.setSingleStep(0.1)
        self.spin_fish_dens.setValue(0.75)
        fish_layout.addWidget(self.spin_fish_dens, 6, 1)

        self.btn_add_fish = QPushButton("+ Tempatkan Ikan di Koordinat")
        self.btn_add_fish.clicked.connect(self.on_manual_add_fish)
        fish_layout.addWidget(self.btn_add_fish, 7, 0, 1, 2)

        self.btn_clear_fish = QPushButton("Hapus Semua Ikan")
        self.btn_clear_fish.setObjectName("danger")
        self.btn_clear_fish.clicked.connect(self.clear_all_fish)
        fish_layout.addWidget(self.btn_clear_fish, 8, 0, 1, 2)

        instructor_panel.addWidget(grp_fish)
        instructor_panel.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(instructor_panel)
        left_widget.setFixedWidth(295)
        main_layout.addWidget(left_widget)

        # =================================================================
        # PANEL KANAN: TAMPILAN PETA TAKTIS, RADAR 360°, ECHOGRAM & CTD
        # =================================================================
        display_grid = QGridLayout()

        # 1. PETA TAKTIS INSTRUKTUR (WORLD VIEW - ATAS KIRI)
        tactical_group = QGroupBox("1. Peta Taktis Dunia (Instructor View - Klik untuk Taruh Ikan)")
        tactical_layout = QVBoxLayout(tactical_group)
        self.tactical_widget = pg.GraphicsLayoutWidget()
        self.tactical_plot = self.tactical_widget.addPlot()
        self.tactical_plot.setAspectLocked(True)
        self.tactical_plot.setLabel('left', 'Utara - Selatan (Y)', units='m')
        self.tactical_plot.setLabel('bottom', 'Barat - Timur (X)', units='m')
        self.tactical_plot.setRange(xRange=(-350, 350), yRange=(-350, 350))
        self.tactical_plot.showGrid(x=True, y=True, alpha=0.25)

        # Elemen grafis pada Peta Taktis
        self.world_track_curve = self.tactical_plot.plot(pen=pg.mkPen('#ffbe0b', width=1.5, style=Qt.PenStyle.DotLine))
        self.world_beam_circle = pg.QtWidgets.QGraphicsEllipseItem(-45, -45, 90, 90)
        self.world_beam_circle.setPen(pg.mkPen('#00f5d4', width=1, style=Qt.PenStyle.DashLine))
        self.tactical_plot.addItem(self.world_beam_circle)

        self.world_fish_scatter = pg.ScatterPlotItem(size=8, brush='#00f5d4', pen='w')
        self.tactical_plot.addItem(self.world_fish_scatter)
        self.world_fish_vectors = self.tactical_plot.plot(pen=pg.mkPen('#ff595e', width=2))

        self.world_ship_marker = pg.ScatterPlotItem(size=16, symbol='t1', brush='#ffbe0b', pen='w')
        self.tactical_plot.addItem(self.world_ship_marker)
        self.world_ship_heading_line = self.tactical_plot.plot(pen=pg.mkPen('#ffbe0b', width=2))

        # Event klik mouse pada peta taktis untuk spawn ikan
        self.tactical_plot.scene().sigMouseClicked.connect(self.on_tactical_map_clicked)

        tactical_layout.addWidget(self.tactical_widget)
        display_grid.addWidget(tactical_group, 0, 0)

        # 2. OMNI-SONAR 360° (RELATIF TERHADAP KAPAL - BAWAH KIRI)
        radar_group = QGroupBox("2. Omni-Sonar 360° (Tampilan Radar Relatif Kapal)")
        radar_layout = QVBoxLayout(radar_group)
        self.radar_widget = pg.GraphicsLayoutWidget()
        self.radar_plot = self.radar_widget.addPlot()
        self.radar_plot.setAspectLocked(True)
        self.radar_plot.hideAxis('bottom')
        self.radar_plot.hideAxis('left')
        self.radar_plot.setRange(xRange=(-270, 270), yRange=(-270, 270))
        self.setup_radar_overlay()
        radar_layout.addWidget(self.radar_widget)
        display_grid.addWidget(radar_group, 1, 0)

        # 3. SCIENTIFIC ECHOGRAM (VERTICAL WATERFALL - ATAS TENGAH)
        echo_group = QGroupBox("3. Scientific Echogram (Kolom Air Tepat di Bawah Kapal)")
        echo_layout = QVBoxLayout(echo_group)
        self.echo_widget = pg.GraphicsLayoutWidget()
        self.echo_plot = self.echo_widget.addPlot()
        self.echo_plot.setLabel('left', 'Kedalaman', units='m')
        self.echo_plot.setLabel('bottom', 'Riwayat Ping')
        self.echo_plot.invertY(True)

        self.img_item = pg.ImageItem()
        self.echo_plot.addItem(self.img_item)
        colormap = pg.colormap.get('turbo')
        self.img_item.setColorMap(colormap)
        bar = pg.ColorBarItem(values=(-85, -15), colorMap=colormap, label='Sv (dB)')
        bar.setImageItem(self.img_item)
        echo_layout.addWidget(self.echo_widget)
        display_grid.addWidget(echo_group, 0, 1)

        # 4. PROFIL OSEANOGRAFI CTD (ATAS KANAN)
        ocean_group = QGroupBox("4. Profil Oseanografi (CTD)")
        ocean_layout = QVBoxLayout(ocean_group)
        self.ocean_widget = pg.GraphicsLayoutWidget()
        self.temp_plot = self.ocean_widget.addPlot()
        self.temp_plot.setLabel('left', 'Kedalaman', units='m')
        self.temp_plot.setLabel('bottom', 'Suhu °C (Merah) | Sound/50 (Cyan)')
        self.temp_plot.invertY(True)
        self.temp_plot.showGrid(x=True, y=True, alpha=0.25)
        self.curve_temp = self.temp_plot.plot(pen=pg.mkPen('#ff595e', width=2.5))
        self.curve_sound = self.temp_plot.plot(pen=pg.mkPen('#00f5d4', width=2, style=Qt.PenStyle.DashLine))
        ocean_layout.addWidget(self.ocean_widget)
        display_grid.addWidget(ocean_group, 0, 2)

        # 5. TELEMETRI STOK IKAN & OSEANOGRAFI (BAWAH TENGAH-KANAN)
        metrics_group = QGroupBox("5. Estimasi Stok Ikan (Hidroakustik) & Telemetri Oseanografi")
        metrics_layout = QGridLayout(metrics_group)

        self.lbl_species = self.create_metric_card(metrics_layout, 0, 0, "Spesies Target", "-")
        self.lbl_nasc = self.create_metric_card(metrics_layout, 0, 1, "NASC (Indeks Biomassa)", "0 m²/nmi²")
        self.lbl_density = self.create_metric_card(metrics_layout, 0, 2, "Densitas Volumetrik", "0.000 ekor/m³")
        self.lbl_biomass = self.create_metric_card(metrics_layout, 0, 3, "Estimasi Biomassa (250m)", "0.00 Ton")

        self.lbl_sst = self.create_metric_card(metrics_layout, 1, 0, "Suhu Permukaan (SST)", "0.0 °C")
        self.lbl_salinity = self.create_metric_card(metrics_layout, 1, 1, "Salinitas", "0.0 PSU")
        self.lbl_soundspeed = self.create_metric_card(metrics_layout, 1, 2, "Kecepatan Suara", "1500 m/s")
        self.lbl_seabed = self.create_metric_card(metrics_layout, 1, 3, "Kedalaman Dasar Laut", "0.0 m")

        display_grid.addWidget(metrics_group, 1, 1, 1, 2)

        display_grid.setColumnStretch(0, 4)
        display_grid.setColumnStretch(1, 5)
        display_grid.setColumnStretch(2, 3)
        display_grid.setRowStretch(0, 5)
        display_grid.setRowStretch(1, 4)

        main_layout.addLayout(display_grid)

    def create_metric_card(self, layout, row, col, title, default_val):
        frame = QFrame()
        frame.setStyleSheet("background-color: #112031; border-radius: 5px; padding: 5px;")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 4, 6, 4)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #88a4c0; font-size: 11px;")
        lbl_v = QLabel(default_val)
        lbl_v.setStyleSheet("color: #00f5d4; font-size: 14px; font-weight: bold;")
        vbox.addWidget(lbl_t)
        vbox.addWidget(lbl_v)
        layout.addWidget(frame, row, col)
        return lbl_v

    def setup_radar_overlay(self):
        for r in [50, 100, 150, 200, 250]:
            circle = pg.QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pg.mkPen(color=(0, 200, 150, 70), width=1))
            self.radar_plot.addItem(circle)
            txt = pg.TextItem(f"{r}m", color=(0, 200, 150, 130), anchor=(0, 1))
            txt.setPos(0, r)
            self.radar_plot.addItem(txt)

        self.radar_plot.plot([-250, 250], [0, 0], pen=pg.mkPen((0, 200, 150, 60)))
        self.radar_plot.plot([0, 0], [-250, 250], pen=pg.mkPen((0, 200, 150, 60)))
        for label, pos in [("HDG 0°", (0, 265)), ("90°", (263, 0)), ("180°", (0, -265)), ("270°", (-263, 0))]:
            t = pg.TextItem(label, color="#5bc0eb", anchor=(0.5, 0.5))
            t.setPos(*pos)
            self.radar_plot.addItem(t)

        self.radar_echoes = pg.ScatterPlotItem(size=5, pen=None)
        self.radar_plot.addItem(self.radar_echoes)

        ship = pg.ScatterPlotItem(pos=[(0, 0)], symbol='t1', size=14, brush='#ffbe0b', pen='w')
        self.radar_plot.addItem(ship)
        self.sweep_line = self.radar_plot.plot([0, 0], [0, 250], pen=pg.mkPen('#00f5d4', width=2.5))

    # =====================================================================
    # LOGIKA KONTROL INSTRUKTUR & PENEMPATAN SKENARIO
    # =====================================================================
    def load_scenario(self, scenario_name):
        self.current_scenario_name = scenario_name
        self.scenario = SAMPLE_SCENARIOS[scenario_name]
        self.max_depth = self.scenario["max_depth"]
        self.spin_fish_depth.setMaximum(self.max_depth - 10.0)

        self.echogram_data = np.full((self.history_len, self.depth_bins), -85.0)
        self.img_item.setRect(0, 0, self.history_len, self.max_depth)
        self.echo_plot.setLimits(xMin=0, xMax=self.history_len, yMin=0, yMax=self.max_depth)

        # Update profil CTD
        depths = np.linspace(0, self.max_depth, 100)
        T_surf, T_deep = self.scenario["surface_temp"], self.scenario["deep_temp"]
        z_therm, S = self.scenario["thermocline_depth"], self.scenario["salinity"]
        temps = T_deep + (T_surf - T_deep) / (1.0 + np.exp((depths - z_therm) / 10.0))
        c = (
            1448.96 + 4.591 * temps - 5.304e-2 * (temps ** 2) + 2.374e-4 * (temps ** 3)
            + 1.340 * (S - 35.0) + 1.630e-2 * depths + 1.675e-7 * (depths ** 2)
            - 1.025e-2 * temps * (S - 35.0) - 7.139e-13 * temps * (depths ** 3)
        )
        self.curve_temp.setData(temps, depths)
        self.curve_sound.setData(c / 50.0, depths)

        self.lbl_species.setText(self.scenario["species"])
        self.lbl_sst.setText(f"{T_surf:.1f} °C")
        self.lbl_salinity.setText(f"{S:.1f} PSU")
        self.lbl_soundspeed.setText(f"{np.mean(c):.1f} m/s")

    def on_ship_control_changed(self):
        self.ship_heading = self.spin_ship_hdg.value()
        self.slider_ship_hdg.blockSignals(True)
        self.slider_ship_hdg.setValue(int(self.ship_heading))
        self.slider_ship_hdg.blockSignals(False)
        self.ship_speed_kts = self.spin_ship_spd.value()

    def reset_ship_position(self):
        self.ship_x = 0.0
        self.ship_y = -150.0
        self.ship_track_x = [self.ship_x]
        self.ship_track_y = [self.ship_y]

    def on_tactical_map_clicked(self, event):
        if not self.chk_click_spawn.isChecked():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            if self.tactical_plot.sceneBoundingRect().contains(pos):
                mouse_point = self.tactical_plot.vb.mapSceneToView(pos)
                x, y = mouse_point.x(), mouse_point.y()
                self.spin_fish_x.setValue(round(x, 1))
                self.spin_fish_y.setValue(round(y, 1))
                self.on_manual_add_fish()

    def on_manual_add_fish(self):
        self.add_fish_school(
            x=self.spin_fish_x.value(),
            y=self.spin_fish_y.value(),
            depth=self.spin_fish_depth.value(),
            heading=self.spin_fish_hdg.value(),
            speed_kts=self.spin_fish_spd.value(),
            density=self.spin_fish_dens.value()
        )

    def add_fish_school(self, x, y, depth, heading, speed_kts, density):
        """Menambahkan gerombolan ikan pada koordinat & vektor gerak tertentu."""
        # Konversi knot ke m/s (1 knot ≈ 0.5144 m/s), dipercepat 3x untuk visualisasi simulasi
        speed_ms = speed_kts * 0.5144 * 3.0
        rad = np.deg2rad(heading)
        vx = speed_ms * np.sin(rad)
        vy = speed_ms * np.cos(rad)

        num_points = int(40 + density * 60)
        spread = 16.0
        school = {
            'x': float(x),
            'y': float(y),
            'vx': vx,
            'vy': vy,
            'heading': heading,
            'speed_ms': speed_ms,
            'depth': float(depth),
            'thickness': 5.0 + density * 6.0,
            'points_offset': np.random.normal(0, spread * 0.45, (num_points, 2)),
            'intensities': np.zeros(num_points),
            'density': float(density)
        }
        self.schools.append(school)

    def clear_all_fish(self):
        self.schools.clear()
        self.radar_echoes.clear()

    def toggle_pause(self):
        if self.is_running:
            self.timer.stop()
            self.btn_pause.setText("Resume Simulasi")
        else:
            self.timer.start()
            self.btn_pause.setText("Pause Simulasi")
        self.is_running = not self.is_running

    # =====================================================================
    # LOOP UTAMA SIMULASI (NAVIGASI + RADAR 360° + ECHOGRAM + BIOMASSA)
    # =====================================================================
    def update_system(self):
        self.time_step += 1

        # 1. UPDATE POSISI KAPAL DI KOORDINAT DUNIA
        ship_v_ms = self.ship_speed_kts * 0.5144 * 3.0  # Skala waktu simulasi 3x
        hdg_rad = np.deg2rad(self.ship_heading)
        self.ship_x += ship_v_ms * np.sin(hdg_rad) * self.dt
        self.ship_y += ship_v_ms * np.cos(hdg_rad) * self.dt

        if self.time_step % 4 == 0:
            self.ship_track_x.append(self.ship_x)
            self.ship_track_y.append(self.ship_y)
            if len(self.ship_track_x) > 400:
                self.ship_track_x.pop(0)
                self.ship_track_y.pop(0)

        self.lbl_ship_pos.setText(f"Posisi: X={self.ship_x:.1f}m, Y={self.ship_y:.1f}m")

        # 2. UPDATE POSISI GEROMBOLAN IKAN & TRANSFORMASI KE KOORDINAT RELATIF KAPAL
        self.sweep_angle = (self.sweep_angle + 5.0) % 360.0
        sweep_rad = np.deg2rad(self.sweep_angle)
        self.sweep_line.setData([0, self.radar_range_m * np.sin(sweep_rad)], [0, self.radar_range_m * np.cos(sweep_rad)])

        world_fish_centers = []
        vec_x, vec_y = [], []
        radar_pts_list = []
        radar_brushes = []
        under_ship_schools = []

        # Matriks rotasi untuk mengubah koordinat dunia ke orientasi haluan kapal (Head-Up Radar)
        cos_h = np.cos(-hdg_rad)
        sin_h = np.sin(-hdg_rad)

        for s in self.schools:
            s['x'] += s['vx'] * self.dt
            s['y'] += s['vy'] * self.dt
            world_fish_centers.append((s['x'], s['y']))

            # Garis vektor arah gerak ikan pada Peta Taktis
            vec_len = 25.0 + s['speed_ms'] * 5.0
            f_rad = np.deg2rad(s['heading'])
            vec_x.extend([s['x'], s['x'] + vec_len * np.sin(f_rad), np.nan])
            vec_y.extend([s['y'], s['y'] + vec_len * np.cos(f_rad), np.nan])

            # Posisi relatif gerombolan terhadap kapal
            dx = s['x'] - self.ship_x
            dy = s['y'] - self.ship_y
            dist_to_ship = np.hypot(dx, dy)

            # Cek apakah kapal melintas di atas gerombolan (beam echosounder vertikal radius 45m)
            if dist_to_ship < 45.0:
                under_ship_schools.append((s, dist_to_ship))

            # Proyeksi titik-titik ikan ke layar Omni-Sonar 360° (Head-Up)
            if dist_to_ship < self.radar_range_m + 30.0:
                world_pts = s['points_offset'] + np.array([dx, dy])
                # Rotasi sesuai haluan kapal agar atas radar selalu arah haluan (HDG 0°)
                rel_x = world_pts[:, 0] * cos_h - world_pts[:, 1] * sin_h
                rel_y = world_pts[:, 0] * sin_h + world_pts[:, 1] * cos_h
                rel_pts = np.column_stack((rel_x, rel_y))

                pt_angles = (np.rad2deg(np.arctan2(rel_x, rel_y)) + 360.0) % 360.0
                s['intensities'] *= 0.93
                angle_diff = np.abs((pt_angles - self.sweep_angle + 180.0) % 360.0 - 180.0)
                s['intensities'][angle_diff < 6.5] = 1.0

                vis = (s['intensities'] > 0.08) & (np.hypot(rel_x, rel_y) <= self.radar_range_m)
                if np.any(vis):
                    radar_pts_list.append(rel_pts[vis])
                    for val in s['intensities'][vis]:
                        radar_brushes.append(pg.mkBrush(int(255 * val), int(255 * min(1.0, val * 1.4)), 40, int(255 * val)))

        # Render Peta Taktis Dunia
        self.world_track_curve.setData(self.ship_track_x, self.ship_track_y)
        self.world_ship_marker.setData(pos=[(self.ship_x, self.ship_y)])
        self.world_ship_heading_line.setData(
            [self.ship_x, self.ship_x + 40.0 * np.sin(hdg_rad)],
            [self.ship_y, self.ship_y + 40.0 * np.cos(hdg_rad)]
        )
        self.world_beam_circle.setRect(self.ship_x - 45.0, self.ship_y - 45.0, 90.0, 90.0)

        if world_fish_centers:
            self.world_fish_scatter.setData(pos=world_fish_centers)
            self.world_fish_vectors.setData(vec_x, vec_y)
        else:
            self.world_fish_scatter.clear()
            self.world_fish_vectors.clear()

        # Render Radar 360°
        if radar_pts_list:
            self.radar_echoes.setData(pos=np.vstack(radar_pts_list), brush=radar_brushes)
        else:
            self.radar_echoes.clear()

        # 3. GENERATE SCIENTIFIC ECHOGRAM (BERDASARKAN GEROMBOLAN DI BAWAH KAPAL)
        depths = np.linspace(0, self.max_depth, self.depth_bins)
        ping = np.random.normal(-83.0, 2.5, self.depth_bins)

        surf = depths < 3.0
        ping[surf] += np.random.uniform(20, 38, np.sum(surf)) * np.exp(-depths[surf])

        # Batimetri dasar laut bergantung pada posisi X, Y kapal di dunia nyata
        seabed_depth = (
            self.scenario["seabed_base"]
            + 6.0 * np.sin(self.ship_x * 0.015)
            + 4.5 * np.cos(self.ship_y * 0.015)
        )
        seabed_idx = int((seabed_depth / self.max_depth) * self.depth_bins)
        if 0 <= seabed_idx < self.depth_bins:
            end_sb = min(self.depth_bins, seabed_idx + 7)
            ping[seabed_idx:end_sb] = np.random.uniform(-20.0, -11.0, end_sb - seabed_idx)
            if end_sb < self.depth_bins:
                ping[end_sb:] = np.linspace(-28.0, -72.0, self.depth_bins - end_sb)

        water_column_sv = np.full(max(1, seabed_idx - 5), -85.0)
        for s, dist in under_ship_schools:
            # Semakin dekat kapal ke pusat gerombolan, ketebalan & kekuatan gema maksimal
            beam_attenuation = max(0.2, 1.0 - (dist / 45.0))
            eff_thickness = s['thickness'] * beam_attenuation
            in_z = np.abs(depths[:len(water_column_sv)] - s['depth']) < eff_thickness
            for idx in np.where(in_z)[0]:
                if np.random.rand() < s['density']:
                    sv_val = self.scenario["ts_ref"] + 13.0 * beam_attenuation + np.random.normal(0, 3.0)
                    ping[idx] = max(ping[idx], sv_val)
                    water_column_sv[idx] = max(water_column_sv[idx], sv_val)

        ping = np.clip(ping + self.slider_gain.value(), -90.0, -10.0)
        self.echogram_data = np.roll(self.echogram_data, -1, axis=0)
        self.echogram_data[-1, :] = ping
        self.img_item.setImage(self.echogram_data, autoLevels=False, levels=(-85, -15))

        # 4. KALKULASI STOK IKAN & TELEMETRI
        sv_linear = 10.0 ** (water_column_sv / 10.0)
        dz = self.max_depth / self.depth_bins
        sa = np.sum(sv_linear) * dz
        nasc = 4.0 * np.pi * (1852.0 ** 2) * sa

        sigma_bs = 10.0 ** (self.scenario["ts_ref"] / 10.0)
        area_density = sa / sigma_bs
        vol_density = area_density / max(1.0, seabed_depth)
        survey_area_m2 = np.pi * (self.radar_range_m ** 2)
        total_fish = area_density * survey_area_m2
        biomass_ton = (total_fish * self.scenario["mean_weight_kg"]) / 1000.0

        if self.time_step % 4 == 0:
            self.lbl_nasc.setText(f"{nasc:,.1f} m²/nmi²")
            self.lbl_density.setText(f"{vol_density:.4f} ekor/m³")
            self.lbl_biomass.setText(f"{biomass_ton:.2f} Ton")
            self.lbl_seabed.setText(f"{seabed_depth:.1f} m")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InstructorSonarStation()
    window.show()
    sys.exit(app.exec())