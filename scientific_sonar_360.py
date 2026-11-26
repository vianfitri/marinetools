import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QPushButton, QSlider, QLabel, QGroupBox, QComboBox, QFrame
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg


# =====================================================================
# DATASET SAMPEL OSEANOGRAFI & STOK IKAN PERAIRAN INDONESIA
# =====================================================================
SAMPLE_SCENARIOS = {
    "Laut Banda (Tuna Sirip Kuning & Cakalang)": {
        "max_depth": 150.0,
        "seabed_base": 125.0,
        "surface_temp": 29.2,
        "thermocline_depth": 65.0,
        "deep_temp": 14.5,
        "salinity": 34.6,
        "species": "Thunnus albacares (Tuna)",
        "mean_length_cm": 58.0,
        "mean_weight_kg": 3.8,
        "ts_ref": -33.5,  # Target Strength individu (dB)
        "school_count": 4,
    },
    "Selat Makassar (Layang & Kembung - Pelagis)": {
        "max_depth": 100.0,
        "seabed_base": 78.0,
        "surface_temp": 28.8,
        "thermocline_depth": 45.0,
        "deep_temp": 18.0,
        "salinity": 33.8,
        "species": "Decapterus spp. (Ikan Layang)",
        "mean_length_cm": 21.0,
        "mean_weight_kg": 0.14,
        "ts_ref": -44.2,
        "school_count": 6,
    },
    "Laut Arafura (Kakap Merah & Demersal)": {
        "max_depth": 80.0,
        "seabed_base": 62.0,
        "surface_temp": 28.1,
        "thermocline_depth": 35.0,
        "deep_temp": 21.5,
        "salinity": 34.2,
        "species": "Lutjanus spp. (Kakap Merah)",
        "mean_length_cm": 42.0,
        "mean_weight_kg": 1.6,
        "ts_ref": -36.8,
        "school_count": 5,
    }
}


class ScientificSonarSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Echosounder 360° & Oceanography Workstation")
        self.resize(1380, 780)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0b131e; color: #d8e6f3; font-family: 'Segoe UI', Arial; }
            QGroupBox { border: 1px solid #1e354d; border-radius: 6px; margin-top: 10px; font-weight: bold; color: #5bc0eb; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #16324f; border: 1px solid #2a5a8a; padding: 6px 12px; border-radius: 4px; color: white; }
            QPushButton:hover { background-color: #1f4870; }
            QComboBox { background-color: #16324f; border: 1px solid #2a5a8a; padding: 5px; border-radius: 4px; color: white; }
        """)

        # Parameter Sistem
        self.current_scenario_name = list(SAMPLE_SCENARIOS.keys())[0]
        self.scenario = SAMPLE_SCENARIOS[self.current_scenario_name]
        
        self.depth_bins = 350
        self.history_len = 400
        self.radar_range_m = 250.0  # Radius jangkauan sonar 360 derajat (meter)
        self.sweep_angle = 0.0      # Sudut sapuan radar saat ini (derajat)
        self.time_step = 0
        self.is_running = True

        # Matriks Echogram (Sv dalam dB)
        self.echogram_data = np.full((self.history_len, self.depth_bins), -85.0)
        
        # Daftar objek kerumunan ikan (dipakai bersama oleh Radar 360 dan Echogram)
        self.schools = []
        
        self.init_ui()
        self.load_scenario(self.current_scenario_name)

        # Timer utama simulasi (30 FPS / ~33 ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system)
        self.timer.start(40)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)

        # --- BARIS ATAS: Kontrol & Pemilihan Sampel Data ---
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("<b>Dataset Sampel Perairan:</b>"))
        
        self.combo_sample = QComboBox()
        self.combo_sample.addItems(list(SAMPLE_SCENARIOS.keys()))
        self.combo_sample.currentTextChanged.connect(self.load_scenario)
        top_bar.addWidget(self.combo_sample)

        self.btn_spawn = QPushButton("+ Tambah Kerumunan Ikan")
        self.btn_spawn.clicked.connect(self.spawn_fish_school)
        top_bar.addWidget(self.btn_spawn)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        top_bar.addWidget(self.btn_pause)

        top_bar.addWidget(QLabel("Gain (dB):"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-15, 15)
        self.slider_gain.setValue(0)
        self.slider_gain.setMaximumWidth(140)
        self.lbl_gain = QLabel("0 dB")
        self.slider_gain.valueChanged.connect(lambda v: self.lbl_gain.setText(f"{v} dB"))
        top_bar.addWidget(self.slider_gain)
        top_bar.addWidget(self.lbl_gain)
        top_bar.addStretch()

        root_layout.addLayout(top_bar)

        # --- AREA VISUALISASI UTAMA (3 KOLOM) ---
        grid = QGridLayout()

        # 1. KOLOM KIRI: RADAR SONAR 360 DERAJAT (OMNI-SONAR PPI)
        radar_group = QGroupBox("1. Omni-Sonar 360° (Plan Position Indicator)")
        radar_layout = QVBoxLayout(radar_group)
        self.radar_widget = pg.GraphicsLayoutWidget()
        self.radar_plot = self.radar_widget.addPlot()
        self.radar_plot.setAspectLocked(True)
        self.radar_plot.hideAxis('bottom')
        self.radar_plot.hideAxis('left')
        self.radar_plot.setRange(xRange=(-275, 275), yRange=(-275, 275))
        self.setup_radar_overlay()
        radar_layout.addWidget(self.radar_widget)
        grid.addWidget(radar_group, 0, 0, 2, 1)

        # 2. KOLOM TENGAH: SCIENTIFIC ECHOGRAM (VERTICAL WATERFALL)
        echo_group = QGroupBox("2. Scientific Echogram (Sv - Volume Backscattering)")
        echo_layout = QVBoxLayout(echo_group)
        self.echo_widget = pg.GraphicsLayoutWidget()
        self.echo_plot = self.echo_widget.addPlot()
        self.echo_plot.setLabel('left', 'Kedalaman', units='m')
        self.echo_plot.setLabel('bottom', 'Ping History (Waktu)')
        self.echo_plot.invertY(True)

        self.img_item = pg.ImageItem()
        self.echo_plot.addItem(self.img_item)
        colormap = pg.colormap.get('turbo')
        self.img_item.setColorMap(colormap)
        
        bar = pg.ColorBarItem(values=(-85, -15), colorMap=colormap, label='Sv (dB)')
        bar.setImageItem(self.img_item)
        echo_layout.addWidget(self.echo_widget)
        grid.addWidget(echo_group, 0, 1, 1, 1)

        # 3. KOLOM KANAN: PROFIL OSEANOGRAFI (CTD & SOUND SPEED)
        ocean_group = QGroupBox("3. Profil Oseanografi (CTD & Akustik)")
        ocean_layout = QVBoxLayout(ocean_group)
        self.ocean_widget = pg.GraphicsLayoutWidget()
        
        self.temp_plot = self.ocean_widget.addPlot(title="Suhu (°C) & Kecepatan Suara")
        self.temp_plot.setLabel('left', 'Kedalaman', units='m')
        self.temp_plot.setLabel('bottom', 'Suhu (°C) [Merah] | Sound Speed/50 (m/s) [Cyan]')
        self.temp_plot.invertY(True)
        self.temp_plot.showGrid(x=True, y=True, alpha=0.25)

        self.curve_temp = self.temp_plot.plot(pen=pg.mkPen('#ff595e', width=2.5))
        self.curve_sound = self.temp_plot.plot(pen=pg.mkPen('#00f5d4', width=2, style=Qt.PenStyle.DashLine))
        ocean_layout.addWidget(self.ocean_widget)
        grid.addWidget(ocean_group, 0, 2, 1, 1)

        # 4. PANEL BAWAH (KANAN-TENGAH): ESTIMASI STOK IKAN & TELEMETRI
        metrics_group = QGroupBox("4. Analisis Hidroakustik Stok Ikan & Parameter Oseanografi")
        metrics_layout = QGridLayout(metrics_group)

        self.lbl_species = self.create_metric_card(metrics_layout, 0, 0, "Target Spesies Dominan", "-")
        self.lbl_nasc = self.create_metric_card(metrics_layout, 0, 1, "NASC (Indeks Biomassa)", "0 m²/nmi²")
        self.lbl_density = self.create_metric_card(metrics_layout, 0, 2, "Densitas Ikan Rata-rata", "0.000 ekor/m³")
        self.lbl_biomass = self.create_metric_card(metrics_layout, 0, 3, "Estimasi Stok (Radius 250m)", "0.00 Ton")

        self.lbl_sst = self.create_metric_card(metrics_layout, 1, 0, "Suhu Permukaan (SST)", "0.0 °C")
        self.lbl_salinity = self.create_metric_card(metrics_layout, 1, 1, "Salinitas Rata-rata", "0.0 PSU")
        self.lbl_soundspeed = self.create_metric_card(metrics_layout, 1, 2, "Kecepatan Suara (Mackenzie)", "1500 m/s")
        self.lbl_seabed = self.create_metric_card(metrics_layout, 1, 3, "Kedalaman Dasar Laut", "0.0 m")

        grid.addWidget(metrics_group, 1, 1, 1, 2)

        # Proporsi lebar kolom
        grid.setColumnStretch(0, 4)  # Radar 360
        grid.setColumnStretch(1, 5)  # Echogram
        grid.setColumnStretch(2, 3)  # Oseanografi
        root_layout.addLayout(grid)

    def create_metric_card(self, layout, row, col, title, default_val):
        frame = QFrame()
        frame.setStyleSheet("background-color: #112031; border-radius: 5px; padding: 6px;")
        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(6, 4, 6, 4)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #88a4c0; font-size: 11px;")
        lbl_v = QLabel(default_val)
        lbl_v.setStyleSheet("color: #00f5d4; font-size: 15px; font-weight: bold;")
        vbox.addWidget(lbl_t)
        vbox.addWidget(lbl_v)
        layout.addWidget(frame, row, col)
        return lbl_v

    def setup_radar_overlay(self):
        """Menggambar lingkaran jarak, garis kompas, dan sapuan radar 360 derajat."""
        # Lingkaran jarak (50m, 100m, 150m, 200m, 250m)
        for r in [50, 100, 150, 200, 250]:
            circle = pg.QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pg.mkPen(color=(0, 200, 150, 70), width=1))
            self.radar_plot.addItem(circle)
            txt = pg.TextItem(f"{r}m", color=(0, 200, 150, 130), anchor=(0, 1))
            txt.setPos(0, r)
            self.radar_plot.addItem(txt)

        # Garis silang kompas (N, E, S, W)
        self.radar_plot.plot([-250, 250], [0, 0], pen=pg.mkPen((0, 200, 150, 60)))
        self.radar_plot.plot([0, 0], [-250, 250], pen=pg.mkPen((0, 200, 150, 60)))
        for label, pos in [("N (0°)", (0, 268)), ("E (90°)", (265, 0)), ("S (180°)", (0, -268)), ("W (270°)", (-265, 0))]:
            t = pg.TextItem(label, color="#5bc0eb", anchor=(0.5, 0.5))
            t.setPos(*pos)
            self.radar_plot.addItem(t)

        # Scatter plot untuk titik gema ikan di Radar 360
        self.radar_echoes = pg.ScatterPlotItem(size=5, pen=None)
        self.radar_plot.addItem(self.radar_echoes)

        # Posisi kapal di pusat (0,0)
        ship = pg.ScatterPlotItem(pos=[(0, 0)], symbol='t1', size=14, brush='#ffbe0b', pen='w')
        self.radar_plot.addItem(ship)

        # Garis sapuan (Sweep Line)
        self.sweep_line = self.radar_plot.plot([0, 0], [0, 250], pen=pg.mkPen('#00f5d4', width=2.5))

    def load_scenario(self, scenario_name):
        """Memuat dataset sampel perairan dan mereset simulasi."""
        self.current_scenario_name = scenario_name
        self.scenario = SAMPLE_SCENARIOS[scenario_name]
        self.max_depth = self.scenario["max_depth"]

        self.echogram_data = np.full((self.history_len, self.depth_bins), -85.0)
        self.img_item.setRect(0, 0, self.history_len, self.max_depth)
        self.echo_plot.setLimits(xMin=0, xMax=self.history_len, yMin=0, yMax=self.max_depth)

        # Reset & spawn kerumunan ikan sesuai karakteristik skenario
        self.schools.clear()
        for _ in range(self.scenario["school_count"]):
            self.spawn_fish_school()

        self.update_oceanography_profile()

    def update_oceanography_profile(self):
        """Menghitung profil CTD vertikal & kecepatan suara menggunakan Rumus Mackenzie."""
        depths = np.linspace(0, self.max_depth, 100)
        T_surf = self.scenario["surface_temp"]
        T_deep = self.scenario["deep_temp"]
        z_therm = self.scenario["thermocline_depth"]
        S = self.scenario["salinity"]

        # Kurva termoklin (penurunan suhu terhadap kedalaman)
        temps = T_deep + (T_surf - T_deep) / (1.0 + np.exp((depths - z_therm) / 10.0))

        # Persamaan Mackenzie (1981) untuk kecepatan rambat suara di air laut (m/s)
        c = (
            1448.96 + 4.591 * temps - 5.304e-2 * (temps ** 2) + 2.374e-4 * (temps ** 3)
            + 1.340 * (S - 35.0) + 1.630e-2 * depths + 1.675e-7 * (depths ** 2)
            - 1.025e-2 * temps * (S - 35.0) - 7.139e-13 * temps * (depths ** 3)
        )

        # Plot suhu & kecepatan suara (diskalakan dibagi 50 agar muat di sumbu X yang sama)
        self.curve_temp.setData(temps, depths)
        self.curve_sound.setData(c / 50.0, depths)

        self.lbl_species.setText(self.scenario["species"])
        self.lbl_sst.setText(f"{T_surf:.1f} °C")
        self.lbl_salinity.setText(f"{S:.1f} PSU")
        self.lbl_soundspeed.setText(f"{np.mean(c):.1f} m/s")

    def spawn_fish_school(self):
        """Membuat gerombolan ikan dengan koordinat 3D (x, y horizontal di radar & z kedalaman)."""
        angle_rad = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(25.0, 210.0)
        num_points = np.random.randint(35, 90)
        spread = np.random.uniform(10.0, 25.0)

        school = {
            'x': dist * np.cos(angle_rad),
            'y': dist * np.sin(angle_rad),
            'vx': np.random.uniform(-0.35, 0.35),
            'vy': np.random.uniform(-0.35, 0.35),
            'depth': np.random.uniform(15.0, self.scenario["seabed_base"] - 12.0),
            'thickness': np.random.uniform(4.0, 11.0),
            'spread': spread,
            'points_offset': np.random.normal(0, spread * 0.45, (num_points, 2)),
            'intensities': np.zeros(num_points),  # Kecerahan gema di radar (efek afterglow)
            'density_factor': np.random.uniform(0.4, 0.9)
        }
        self.schools.append(school)

    def toggle_pause(self):
        if self.is_running:
            self.timer.stop()
            self.btn_pause.setText("Resume")
        else:
            self.timer.start()
            self.btn_pause.setText("Pause")
        self.is_running = not self.is_running

    def update_system(self):
        self.time_step += 1

        # 1. UPDATE SAPUAN RADAR 360 DERAJAT
        self.sweep_angle = (self.sweep_angle + 4.5) % 360.0
        rad = np.deg2rad(self.sweep_angle)
        self.sweep_line.setData([0, self.radar_range_m * np.sin(rad)], [0, self.radar_range_m * np.cos(rad)])

        # Gerakkan kerumunan ikan & hitung deteksi sapuan radar
        all_pts = []
        all_brushes = []
        near_nadir_schools = []  # Kerumunan yang melintas tepat di bawah kapal (beam vertikal)

        for s in self.schools:
            s['x'] += s['vx']
            s['y'] += s['vy']

            # Pantulkan jika keluar radius 230m
            r_center = np.hypot(s['x'], s['y'])
            if r_center > 230.0:
                s['vx'] *= -1
                s['vy'] *= -1

            # Cek apakah kerumunan melintas di bawah sorotan vertikal kapal (radius < 45m)
            if r_center < 45.0:
                near_nadir_schools.append(s)

            # Hitung posisi tiap bintik ikan dalam gerombolan
            pts = s['points_offset'] + np.array([s['x'], s['y']])
            pt_angles = (np.rad2deg(np.arctan2(pts[:, 0], pts[:, 1])) + 360.0) % 360.0

            # Efek fosfor radar: menyala terang saat tersapu garis sonar, lalu memudar perlahan
            s['intensities'] *= 0.94
            angle_diff = np.abs((pt_angles - self.sweep_angle + 180.0) % 360.0 - 180.0)
            hit_mask = angle_diff < 6.0
            s['intensities'][hit_mask] = 1.0

            visible = s['intensities'] > 0.08
            if np.any(visible):
                all_pts.append(pts[visible])
                for val in s['intensities'][visible]:
                    # Warna dari hijau redup ke kuning-merah terang saat baru tersapu
                    r_col = int(255 * val)
                    g_col = int(255 * min(1.0, val * 1.4))
                    alpha = int(255 * val)
                    all_brushes.append(pg.mkBrush(r_col, g_col, 40, alpha))

        if all_pts:
            merged_pts = np.vstack(all_pts)
            self.radar_echoes.setData(pos=merged_pts, brush=all_brushes)
        else:
            self.radar_echoes.clear()

        # 2. GENERATE PING VERTICAL SCIENTIFIC ECHOGRAM
        depths = np.linspace(0, self.max_depth, self.depth_bins)
        ping = np.random.normal(-83.0, 2.5, self.depth_bins)

        # Surface clutter
        surf = depths < 3.0
        ping[surf] += np.random.uniform(20, 38, np.sum(surf)) * np.exp(-depths[surf])

        # Dasar Laut (Seabed)
        seabed_depth = (
            self.scenario["seabed_base"]
            + 4.5 * np.sin(self.time_step * 0.025)
            + 1.8 * np.cos(self.time_step * 0.06)
        )
        seabed_idx = int((seabed_depth / self.max_depth) * self.depth_bins)
        if 0 <= seabed_idx < self.depth_bins:
            end_sb = min(self.depth_bins, seabed_idx + 7)
            ping[seabed_idx:end_sb] = np.random.uniform(-20.0, -11.0, end_sb - seabed_idx)
            if end_sb < self.depth_bins:
                ping[end_sb:] = np.linspace(-28.0, -72.0, self.depth_bins - end_sb)

        # Masukkan gema ikan ke kolom air (kolom di atas dasar laut)
        water_column_sv = np.full(max(1, seabed_idx - 5), -85.0)
        for s in near_nadir_schools:
            in_z = np.abs(depths[:len(water_column_sv)] - s['depth']) < s['thickness']
            for idx in np.where(in_z)[0]:
                if np.random.rand() < s['density_factor']:
                    sv_val = self.scenario["ts_ref"] + 12.0 + np.random.normal(0, 3.5)
                    ping[idx] = max(ping[idx], sv_val)
                    water_column_sv[idx] = max(water_column_sv[idx], sv_val)

        gain = self.slider_gain.value()
        ping = np.clip(ping + gain, -90.0, -10.0)

        self.echogram_data = np.roll(self.echogram_data, -1, axis=0)
        self.echogram_data[-1, :] = ping
        self.img_item.setImage(self.echogram_data, autoLevels=False, levels=(-85, -15))

        # 3. KALKULASI ESTIMASI STOK IKAN (HIDROAKUSTIK)
        # Konversi Sv (dB) -> sv linear (m^-1) -> sA / NASC (m^2 / nmi^2)
        sv_linear = 10.0 ** (water_column_sv / 10.0)
        dz = self.max_depth / self.depth_bins
        sa = np.sum(sv_linear) * dz
        nasc = 4.0 * np.pi * (1852.0 ** 2) * sa  # Rumus standar NASC

        # Konversi TS individu (dB) -> sigma_bs (backscattering cross-section)
        sigma_bs = 10.0 ** (self.scenario["ts_ref"] / 10.0)
        # Densitas volumetrik (ekor / m^3) & Biomassa dalam radius survei 250m
        area_density = sa / sigma_bs  # ekor per m^2 kolom air
        vol_density = area_density / max(1.0, seabed_depth)
        survey_area_m2 = np.pi * (self.radar_range_m ** 2)
        total_fish = area_density * survey_area_m2 * (len(self.schools) / max(1, len(near_nadir_schools) + 1))
        biomass_ton = (total_fish * self.scenario["mean_weight_kg"]) / 1000.0

        # Update kartu telemetri setiap 5 frame agar angka mudah dibaca
        if self.time_step % 5 == 0:
            self.lbl_nasc.setText(f"{nasc:,.1f} m²/nmi²")
            self.lbl_density.setText(f"{vol_density:.4f} ekor/m³")
            self.lbl_biomass.setText(f"{biomass_ton:.2f} Ton")
            self.lbl_seabed.setText(f"{seabed_depth:.1f} m")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScientificSonarSystem()
    window.show()
    sys.exit(app.exec())