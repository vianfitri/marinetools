import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QSlider, QLabel, QGroupBox
)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg


class EchosounderSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulasi Echosounder - Deteksi Kerumunan Ikan")
        self.resize(1000, 600)

        # Parameter Simulasi Sonar
        self.max_depth = 100.0       # Kedalaman maksimum (meter)
        self.depth_bins = 400        # Resolusi vertikal (piksel)
        self.history_len = 500       # Lebar riwayat ping horizontal
        self.time_step = 0
        self.is_running = True

        # Matriks data echogram (dB: -90 dB noise dasar hingga -10 dB pantulan kuat)
        self.echogram_data = np.full((self.history_len, self.depth_bins), -85.0)

        # Parameter Kerumunan Ikan (Fish Schools)
        self.schools = []
        self.spawn_fish_school()

        self.init_ui()

        # Timer untuk memicu "Ping" sonar (20 Hz / setiap 50 ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ping)
        self.timer.start(50)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Area Grafik Echogram (PyQtGraph) ---
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.plot = self.graphics_widget.addPlot(title="Echogram (Real-Time Sonar Waterfall)")
        
        # Konfigurasi sumbu
        self.plot.setLabel('left', 'Kedalaman', units='m')
        self.plot.setLabel('bottom', 'Riwayat Ping (Waktu)')
        self.plot.invertY(True)  # Kedalaman 0m berada di atas
        self.plot.setLimits(xMin=0, xMax=self.history_len, yMin=0, yMax=self.max_depth)

        # Item gambar untuk menampilkan matriks sonar
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)

        # Mengatur skala koordinat gambar agar sesuai dengan meter
        self.img_item.setRect(0, 0, self.history_len, self.max_depth)

        # Skema warna standar Echosounder (Biru tua -> Hijau -> Kuning -> Merah)
        colormap = pg.colormap.get('turbo') 
        self.img_item.setColorMap(colormap)

        # ColorBar untuk indikator kekuatan sinyal (dB)
        bar = pg.ColorBarItem(values=(-85, -15), colorMap=colormap, label='Target Strength (dB)')
        bar.setImageItem(self.img_item)

        main_layout.addWidget(self.graphics_widget)

        # --- Panel Kontrol ---
        control_group = QGroupBox("Kontrol Sonar & Simulasi")
        control_layout = QHBoxLayout()

        # Tombol Pause / Resume
        self.btn_toggle = QPushButton("Pause")
        self.btn_toggle.clicked.connect(self.toggle_simulation)
        control_layout.addWidget(self.btn_toggle)

        # Tombol Munculkan Kerumunan Ikan Baru
        self.btn_spawn = QPushButton("Tambah Kerumunan Ikan")
        self.btn_spawn.clicked.connect(self.spawn_fish_school)
        control_layout.addWidget(self.btn_spawn)

        # Slider Gain (Sensitivitas Penerima)
        control_layout.addWidget(QLabel("Gain (dB):"))
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(-20, 20)
        self.slider_gain.setValue(0)
        self.lbl_gain = QLabel("0 dB")
        self.slider_gain.valueChanged.connect(lambda v: self.lbl_gain.setText(f"{v} dB"))
        control_layout.addWidget(self.slider_gain)
        control_layout.addWidget(self.lbl_gain)

        # Slider Kecepatan Kapal (Ping Rate)
        control_layout.addWidget(QLabel("Ping Interval (ms):"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(15, 150)
        self.slider_speed.setValue(50)
        self.slider_speed.valueChanged.connect(self.change_speed)
        control_layout.addWidget(self.slider_speed)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

    def toggle_simulation(self):
        if self.is_running:
            self.timer.stop()
            self.btn_toggle.setText("Resume")
        else:
            self.timer.start()
            self.btn_toggle.setText("Pause")
        self.is_running = not self.is_running

    def change_speed(self, value):
        self.timer.setInterval(value)

    def spawn_fish_school(self):
        """Membuat objek kerumunan ikan baru yang akan melewati sorotan sonar."""
        school = {
            'center_depth': np.random.uniform(20.0, 65.0),  # Kedalaman (m)
            'thickness': np.random.uniform(4.0, 12.0),      # Ketebalan vertikal (m)
            'duration': np.random.randint(30, 90),          # Lebar horizontal (jumlah ping)
            'age': 0,                                       # Umur kemunculan
            'density': np.random.uniform(0.3, 0.8),         # Kepadatan ikan
            'strength': np.random.uniform(-45.0, -25.0)     # Kekuatan pantulan (dB)
        }
        self.schools.append(school)

    def generate_ping(self):
        """Menghasilkan 1 kolom data vertikal (1 kali pancaran gelombang suara)."""
        depths = np.linspace(0, self.max_depth, self.depth_bins)
        
        # 1. Background noise laut
        ping = np.random.normal(-82.0, 3.0, self.depth_bins)

        # 2. Surface clutter (pantulan permukaan & gelembung lunas kapal)
        surface_mask = depths < 3.5
        ping[surface_mask] += np.random.uniform(25, 45, np.sum(surface_mask)) * np.exp(-depths[surface_mask])

        # 3. Kontur Dasar Laut (Seabed) bergelombang menggunakan kombinasi sinus
        seabed_depth = (
            78.0 
            + 6.0 * np.sin(self.time_step * 0.02) 
            + 2.5 * np.cos(self.time_step * 0.07)
            + np.random.normal(0, 0.3)
        )
        
        # Pantulan keras pada batas dasar laut (garis merah tebal)
        seabed_idx = int((seabed_depth / self.max_depth) * self.depth_bins)
        if 0 <= seabed_idx < self.depth_bins:
            # Lapisan sedimen keras
            end_hard = min(self.depth_bins, seabed_idx + 8)
            ping[seabed_idx:end_hard] = np.random.uniform(-22.0, -12.0, end_hard - seabed_idx)
            # Sub-bottom (pantulan sekunder di bawah dasar laut)
            if end_hard < self.depth_bins:
                decay = np.linspace(-30.0, -75.0, self.depth_bins - end_hard)
                ping[end_hard:] = decay + np.random.normal(0, 4.0, len(decay))

        # 4. Simulasi Kerumunan Ikan (Fish Schools)
        active_schools = []
        for s in self.schools:
            if s['age'] < s['duration']:
                # Bentuk elips/organik untuk kerumunan ikan
                progress = s['age'] / s['duration']
                envelope = np.sin(progress * np.pi)  # Mengecil di awal & akhir kerumunan
                current_thickness = s['thickness'] * envelope

                # Gerakan naik-turun alami ikan
                current_depth = s['center_depth'] + np.sin(self.time_step * 0.1) * 1.5

                # Cari indeks kedalaman yang terkena kerumunan
                in_school = np.abs(depths - current_depth) < current_thickness
                indices = np.where(in_school)[0]

                for idx in indices:
                    # Tidak semua titik di dalam kerumunan terisi (efek bintik-bintik individu ikan)
                    if np.random.rand() < s['density']:
                        # Bagian tengah kerumunan memantulkan sinyal lebih kuat
                        dist_ratio = 1.0 - (abs(depths[idx] - current_depth) / (current_thickness + 1e-5))
                        echo_val = s['strength'] + (dist_ratio * 12.0) + np.random.normal(0, 4.0)
                        ping[idx] = max(ping[idx], echo_val)

                s['age'] += 1
                active_schools.append(s)
        
        self.schools = active_schools

        # Munculkan kerumunan baru secara otomatis sesekali
        if np.random.rand() < 0.015 and len(self.schools) < 4:
            self.spawn_fish_school()

        # Terapkan pengaturan Gain dari slider
        gain = self.slider_gain.value()
        ping = np.clip(ping + gain, -90.0, -10.0)

        return ping

    def update_ping(self):
        self.time_step += 1
        new_ping = self.generate_ping()

        # Geser data lama ke kiri, masukkan ping baru di sisi paling kanan
        self.echogram_data = np.roll(self.echogram_data, -1, axis=0)
        self.echogram_data[-1, :] = new_ping

        # Perbarui tampilan grafik
        self.img_item.setImage(self.echogram_data, autoLevels=False, levels=(-85, -15))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EchosounderSimulator()
    window.show()
    sys.exit(app.exec())