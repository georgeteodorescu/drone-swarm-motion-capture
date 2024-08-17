# -----------------------------------------------------------
# -------------------- camera_widget.py ---------------------
# -----------------------------------------------------------
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from camera_manager import CameraManager
from logger import Logger


class VideoThread(QThread):
    _run_flag: bool
    camera_manager: CameraManager

    change_pixmap_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(int)
    points_updated_signal = pyqtSignal(int)  # semnal pentru emiterea numarului actualizat de puncte

    def __init__(self, camera_manager: CameraManager):
        super().__init__()
        self._run_flag = True
        self.camera_manager = camera_manager

    def run(self):
        frequency = 100
        loop_interval = 1.0 / frequency
        next_run_time = time.perf_counter()
        frame_count = 0
        fps_start_time = time.perf_counter()

        while self._run_flag:
            current_time = time.perf_counter()
            if current_time >= next_run_time:
                frame_count += 1
                next_run_time = current_time + loop_interval

                if self.camera_manager.camera_devices is not None:
                    frames = self.camera_manager.get_combined_frames()
                    if frames is not None:
                        self.change_pixmap_signal.emit(frames)
                        self.points_updated_signal.emit(
                            len(self.camera_manager.triang_processor.tracked_points_for_camera_pose_list))
                else:
                    # emite o imagine alba in cazul in care nu sunt detectate dispozitive camera_devices
                    white_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    self.change_pixmap_signal.emit(white_frame)

                # calcularea FPS in fiecare secunda
                if current_time - fps_start_time >= 1.0:
                    self.fps_signal.emit(frame_count)
                    frame_count = 0
                    fps_start_time = current_time

            # sleep pana la urmatorul interval de timp asteptat sau cedarea executiei catre altor fire de executie
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # daca firul este in urma, sare peste sleep ci continua imediat, dar ajutam timpul de executie urmator
                next_run_time = current_time + loop_interval

    def stop(self):
        """Seteaza flag-ul de executie la False pentru a semnala firului de executie sa termine fara a mai astepta"""
        self._run_flag = False


class CameraWidget(QWidget):
    logger: Logger
    camera_manager: CameraManager

    def __init__(self, camera_manager: CameraManager, fpsLabel: QLabel = None, parent=None):
        QWidget.__init__(self, parent)

        self.logger = Logger(log_to_file=True)
        self.logger.info("Initializare Camera Widget...")

        self.camera_manager = camera_manager

        self.disply_width: int = 600
        self.display_height: int = 500
        self.image_label: QLabel = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        vbox: QVBoxLayout = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        self.thread: VideoThread = VideoThread(camera_manager=self.camera_manager)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.fpsLabel: QLabel = fpsLabel

        self.thread.fps_signal.connect(self.update_fps)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updateaza image_label cu o noua imagine opencv"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(int)
    def update_fps(self, fps):
        """Updateaza label-ul FPS cu o noua valoare"""
        if self.fpsLabel:
            self.fpsLabel.setText(f"FPS {fps}")

    def convert_cv_qt(self, cv_img):
        """ https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
            Converteste dintr-o imagine opencv la QPixmap
        """

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        """Gestioneaza evenimentul de inchidere a widget-ului pentru a opri in mod corespunzator firul de executie al
        videoclipului"""
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.quit()
            self.thread.wait()
        event.accept()
