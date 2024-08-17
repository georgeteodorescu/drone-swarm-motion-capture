# ------------------------------------------------------------
# -------------------- main_window.py ----------------------
# ------------------------------------------------------------
import sys

import numpy as np
import qdarktheme
from PyQt6 import uic
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import *
from PyQt6.QtWidgets import QApplication

from camera_manager import CameraManager
from camera_widget import CameraWidget
from logger import Logger
from plot_widget import PlotWidget


class DroneSwarmControlAppWindow(QMainWindow):
    camera_manager: CameraManager

    isCollectingPoints: bool

    fpsLabel: QLabel
    calculate_camera_pose_label: QLabel

    exposureSlider: QSlider
    gainSlider: QSlider

    collectPointsBttn: QPushButton
    saveTrackedPointsBttn: QPushButton
    loadTrackedPointsBttn: QPushButton

    calculateCameraPoseBttn: QPushButton
    saveCameraPosesBttn: QPushButton
    loadCameraPosesBttn: QPushButton

    camera_manager_widget: CameraWidget
    plot_widget: PlotWidget
    plot_placeholder: QWidget

    def __init__(self):
        super().__init__()
        # incarcarea UI-ului din fisierul .ui
        uic.loadUi("src/main_window.ui", self)

        self.setWindowTitle("Drone Swarm Control GUI")
        self.working_directory = "/home/george/Projects/drone-swarm-motion-capture"

        # initializarea logger-ului
        self.logger = Logger(log_to_file=True)
        self.logger.info("Initializare Drone Swarm Control GUI...")

        # initializarea managerului de camere
        self.camera_manager = CameraManager()

        self.isCollectingPoints = False
        self.isTriangulatingObjects = False

        # gasirea elementelor din UI pentru folosirea in metode
        self.fpsLabel = self.findChild(QLabel, "fpsLabel")
        self.calculate_camera_pose_label = self.findChild(QLabel, "calculateCameraPoseLabel")

        self.startCameraBttn = self.findChild(QPushButton, "startCameraBttn")

        self.exposureSlider = self.findChild(QSlider, "exposureSlider")
        self.gainSlider = self.findChild(QSlider, "gainSlider")

        self.collectPointsBttn = self.findChild(QPushButton, "collectPointsBttn")
        self.saveTrackedPointsBttn = self.findChild(QPushButton, "saveTrackedPointsBttn")
        self.loadTrackedPointsBttn = self.findChild(QPushButton, "loadTrackedPointsBttn")

        self.calculateCameraPoseBttn = self.findChild(QPushButton, "calculateCameraPoseBttn")
        self.saveCameraPosesBttn = self.findChild(QPushButton, "saveCameraPosesBttn")
        self.loadCameraPosesBttn = self.findChild(QPushButton, "loadCameraPosesBttn")

        self.loadCameraParams = self.findChild(QPushButton, "loadCameraParams")

        self.startTriangBttn = self.findChild(QPushButton, "startTriangBttn")
        self.startObjectLocalization = self.findChild(QPushButton, "startObjectLocalization")
        self.setOrigin = self.findChild(QPushButton, "setOrigin")

        # conectarea slider-elor la functiile de actualizare a setarilor camerelor
        self.exposureSlider.valueChanged.connect(self.update_camera_settings)
        self.gainSlider.valueChanged.connect(self.update_camera_settings)

        # setarea butoanelor la functiile corespunzatoare
        self.startCameraBttn.clicked.connect(self.start_camera)

        self.collectPointsBttn.clicked.connect(self.toggle_point_collection)
        self.saveTrackedPointsBttn.clicked.connect(self.save_tracked_points)
        self.loadTrackedPointsBttn.clicked.connect(self.load_tracked_points)

        self.calculateCameraPoseBttn.clicked.connect(self.estimate_camera_poses)
        self.saveCameraPosesBttn.clicked.connect(self.save_camera_poses)
        self.loadCameraPosesBttn.clicked.connect(self.load_camera_poses)
        self.loadCameraParams.clicked.connect(self.load_camera_params)

        self.startTriangBttn.clicked.connect(self.toggle_triangulation)

        # dezactivarea butoanelor pana cand exista date de salvat
        self.saveCameraPosesBttn.setEnabled(False)
        self.saveTrackedPointsBttn.setEnabled(False)

        # init widget-ului pentru gestionarea camerelor
        self.camera_manager_widget = CameraWidget(camera_manager=self.camera_manager, fpsLabel=self.fpsLabel,
                                                  parent=self)

        self.video_placeholder = self.findChild(QWidget, "videoPlaceholder")
        video_layout = QVBoxLayout(self.video_placeholder)
        video_layout.addWidget(self.camera_manager_widget)
        self.camera_manager_widget.setGeometry(0, 0, self.video_placeholder.width(), self.video_placeholder.height())
        self.camera_manager_widget.thread.points_updated_signal.connect(self.update_points_label)

        # init widget-ului pentru grafic
        self.plot_widget = PlotWidget(parent=self)
        self.plot_placeholder = self.findChild(QWidget, "plotPlaceholder")
        self.plot_placeholder_layout = QVBoxLayout(self.plot_placeholder)
        self.plot_placeholder_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_placeholder_layout.addWidget(self.plot_widget)

        # conectarea semnalului pentru actualizarea pozitiilor camerelor
        self.camera_manager.camera_poses_updated.connect(self.update_plot)

    # metoda pentru pornirea camerelor
    def start_camera(self):
        self.camera_manager.initialize_cameras()

    # metoda pentru activarea/dezactivarea butonului de salvare a pozitiilor camerelor
    def enable_save_camera_poses_button(self, camera_poses):
        if camera_poses:
            self.saveCameraPosesBttn.setEnabled(True)
        else:
            self.saveCameraPosesBttn.setEnabled(False)

    # metoda pentru activarea/dezactivarea butonului de salvare a punctelor urmarite
    def enable_save_tracked_points_button(self, points_count):
        if points_count > 10:
            self.saveTrackedPointsBttn.setEnabled(True)
        else:
            self.saveTrackedPointsBttn.setEnabled(False)

    # metoda pentru salvarea punctelor urmarite intr-un fisier
    def save_tracked_points(self):
        if self.camera_manager.triang_processor.tracked_points_for_camera_pose_list:
            filePath, _ = QFileDialog.getSaveFileName(self, "Salvare Puncte Urmarite", self.working_directory,
                                                      "NPZ Files (*.npz);;All Files (*)")
            if filePath:
                if not filePath.endswith('.npz'):
                    filePath += '.npz'
                np.savez_compressed(filePath, tracked_points=np.array(
                        self.camera_manager.triang_processor.tracked_points_for_camera_pose_list
                    )
                )
                self.logger.info(f'Puncte Urmarite salvate in {filePath}')

    # metoda pentru incarcarea punctelor urmarite dintr-un fisier
    def load_tracked_points(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Incarcare Puncte Urmarite", self.working_directory,
                                                  "NPZ Files (*.npz);;All Files (*)")
        if filePath:
            data = np.load(filePath, allow_pickle=True)
            self.camera_manager.triang_processor.tracked_points_for_camera_pose_list = data['tracked_points'].tolist()
            self.update_points_label(len(data['tracked_points'].tolist()))
            self.logger.info(f'Puncte Urmarite incarcate din {filePath}')

    # metoda pentru salvarea pozitiilor camerelor intr-un fisier
    def save_camera_poses(self):
        if self.camera_manager.triang_processor.camera_poses:
            filePath, _ = QFileDialog.getSaveFileName(self, "Salvare Pozitii Camere", self.working_directory,
                                                      "NPZ Files (*.npz);;All Files (*)")
            if filePath:
                if not filePath.endswith('.npz'):
                    filePath += '.npz'
                np.savez_compressed(filePath, camera_poses=np.array(self.camera_manager.triang_processor.camera_poses))
                self.logger.info(f'Pozitia Camerelor salvate in {filePath}')

    # metoda pentru incarcarea pozitiilor camerelor dintr-un fisier
    def load_camera_poses(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Incarcare Pozitii Camere", self.working_directory,
                                                  "NPZ Files (*.npz);;All Files (*)")
        if filePath:
            data = np.load(filePath, allow_pickle=True)
            camera_poses = data['camera_poses'].tolist()

            self.camera_manager.triang_processor.update_camera_poses(camera_poses)
            self.logger.info(f'Pozitii Camere incarcate din {filePath}')

    # metoda pentru incarcarea parametrilor camerelor dintr-un director
    def load_camera_params(self):
        base_path = QFileDialog.getExistingDirectory(self, "Selectare Director cu Parametrii Camerelor",
                                                     self.working_directory)
        if base_path:
            self.camera_manager.load_camera_parameters(base_path + "/")

    # pyqtslot pentru actualizarea graficului cu pozitii noi
    @pyqtSlot(list)
    def update_plot(self, camera_poses):
        self.enable_save_camera_poses_button(camera_poses)
        self.plot_widget.update_plot(camera_poses)

    # pyqtslot pentru actualizarea etichetei cu numarul de puncte colectate
    @pyqtSlot(int)
    def update_points_label(self, points_count):
        self.calculate_camera_pose_label.setText(f"{points_count} puncte colectate")
        self.enable_save_tracked_points_button(points_count)

    def update_camera_settings(self):
        exposure = self.exposureSlider.value()
        gain = self.gainSlider.value()

        # mapped_value = (new_range_max - new_range_min) / (old_range_max - old_range_min) * (value - old_range_min)
        # + new_range_min

        # map exposure de la 0-100 la 0-255 intervalul acceptat de camere
        mapped_exposure = ((255 - 1) / (100 - 1)) * (exposure - 1) + 1
        clamped_exposure = max(0, min(255, int(mapped_exposure)))

        # map gain de la 0-100 la 0-63 intervalul acceptat de camere
        mapped_gain = ((63 - 1) / (100 - 1)) * (gain - 1) + 1
        clamped_gain = max(0, min(63, int(mapped_gain)))

        self.camera_manager.camera_devices.exposure = [clamped_exposure for _ in
                                                       self.camera_manager.camera_devices.exposure]
        self.camera_manager.camera_devices.gain = [clamped_gain for _ in self.camera_manager.camera_devices.gain]

    def toggle_point_collection(self):
        if self.isCollectingPoints:
            self.isCollectingPoints = False
            self.collectPointsBttn.setText("Start")
            self.collectPointsBttn.setStyleSheet("background-color: none")

            self.camera_manager.stop_capture()
        else:
            self.isCollectingPoints = True
            self.collectPointsBttn.setText("Stop")
            self.collectPointsBttn.setStyleSheet("background-color: red")

            self.camera_manager.start_capture()

    def toggle_triangulation(self):
        if self.isTriangulatingObjects:
            self.isCollectingPoints = False
            self.isTriangulatingObjects = False
            self.startTriangBttn.setText("Start Triangulare")
            self.startTriangBttn.setStyleSheet("background-color: none")

            self.camera_manager.stop_capture()
            self.camera_manager.stop_triangulating()
        else:
            self.isCollectingPoints = True
            self.isTriangulatingObjects = True
            self.startTriangBttn.setText("Stop Triangulare")
            self.startTriangBttn.setStyleSheet("background-color: red")

            self.camera_manager.start_capture()
            self.camera_manager.start_triangulating()

    def estimate_camera_poses(self):
        self.logger.info(f'Pornirea calcularii pozitiei camerelor...')
        if self.isCollectingPoints:
            self.toggle_point_collection()

        self.camera_manager.triang_processor.camera_poses = np.nan
        # calcularea pozitiilor camerelor se poate face intr-un thread separat sau in thread-ul principal
        self.camera_manager.triang_processor.start_estimate_camera_poses_in_thread()
        # self.camera_manager.triang_processor.estimate_camera_poses()

    def closeEvent(self, event):
        if hasattr(self.camera_manager_widget, 'thread'):
            if self.camera_manager_widget.thread.isRunning():
                self.camera_manager_widget.thread.stop()
                self.camera_manager_widget.thread.quit()
                self.camera_manager_widget.thread.wait()

            del self.camera_manager_widget.thread

        event.accept()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        qdarktheme.setup_theme(theme="dark", corner_shape="rounded")
        window = DroneSwarmControlAppWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print("Eroare la initializarea aplicatiei:", e)
        sys.exit(1)
