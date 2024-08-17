# ------------------------------------------------------------
# -------------------- camera_manager.py ----------------------
# ------------------------------------------------------------
import time

import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from scipy.io import loadmat

from logger import Logger
from pseyepy import Camera
from triangulation_processor import TriangulationProcessor
from utils import detect_and_annotate_points


def convert_to_bgr(frame):
    """
    Converteste o imagine din format RBG in format BGR.

    Args:
        frame (numpy.ndarray): Imaginea in format RBG.

    Returns:
        numpy.ndarray: Imaginea convertita in format BGR.
    """
    return cv.cvtColor(frame, cv.COLOR_RGB2BGR)


class CameraManager(QObject):
    camera_poses_updated = pyqtSignal(list)

    logger: Logger
    camera_devices: Camera
    triang_processor: TriangulationProcessor

    active_camera_count: int
    camera_calibration_params: dict

    collecting_points_active: bool

    def __init__(self):
        super().__init__()
        self.logger = Logger(log_to_file=True)
        self.logger.info("Initializare Camera Manager...")

        self.camera_devices = None
        self.active_camera_count = 4  # pentru debugging far camere atasate
        self.camera_calibration_params = {}
        self.triang_processor = TriangulationProcessor(self)
        self.last_append_time = time.perf_counter()

        self.collecting_points_active = False
        self.triangulating_points_active = False
        self.locating_objects_active = False

        self.drones_num = 1

        self.load_camera_parameters('src/camera_params/')  # pentru debugging

    def initialize_cameras(self):
        """Initializeaza camerele, setand parametrii initiali si incarcand parametrii de calibrare."""
        if self.camera_devices is None:
            self.camera_devices = Camera(fps=100, resolution=Camera.RES_SMALL, gain=31, exposure=126)
            self.active_camera_count = len(self.camera_devices.gain)
            self.logger.info(f'Nr. de Camere: {self.active_camera_count}')
            self.load_camera_parameters('src/camera_params/')

    def load_camera_parameters(self, base_path):
        """
        Incarca parametrii de calibrare pentru camere din fisiere .mat.

        Args:
            base_path (str): Calea catre directorul unde sunt stocate fisierele de calibrare.
        """
        for i in range(1, self.active_camera_count + 1):
            filepath = f'{base_path}cameraParams_cam{i}.mat'
            try:
                data = loadmat(filepath)
                self.camera_calibration_params[f'camera_{i}'] = {
                    'camera_matrix': data['cameraMatrix'],
                    'dist_coeff': data['distCoeffs'],
                    'focal_distance': data['focal_distance']
                }
                self.logger.debug(
                    f"Parametri de Calibrare pentru Camera {i}: {self.camera_calibration_params[f'camera_{i}']}")
            except FileNotFoundError:
                self.logger.error(f'Fisierul {filepath} nu a fost gasit.')

    def get_camera_calibration_params(self, camera_num):
        """
        Returneaza parametrii de calibrare pentru o anumita camera.

        Args:
            camera_num (int): Numarul camerei.

        Returns:
            dict: Parametrii de calibrare pentru camera specificata.
        """
        params = self.camera_calibration_params.get(f'camera_{camera_num + 1}', {})
        return {
            'intrinsic_matrix': np.array(params.get('camera_matrix', np.eye(3))),
            'distortion_coef': np.array(params.get('dist_coeff', np.zeros(5))),
            'focal_distance': params.get('focal_distance', 0),
            'rotation': params.get('rotation', 0),
        }

    def process_frames(self, frames):
        """
        Proceseaza cadrele primite de la camere pentru a detecta si adnota punctele de interes.

        Args:
            frames (list): Lista cu cadrele capturate de camere.

        Returns:
            list: Lista cu cadrele procesate.
        """
        tracked_points = []
        current_time = time.perf_counter()
        if self.collecting_points_active:
            for i in range(0, self.active_camera_count):
                frames[i], single_camera_tracked_points = detect_and_annotate_points(frames[i])
                # [[CiP1x, CiP1y], [CiP2x, CiP2y]]  lista de coordonate pentru fiecare cadru al camerei, [[np.nan,
                # np.nan]] daca nu exista puncte
                tracked_points.append(single_camera_tracked_points)

        # [ [[C1P1x, C1P1y], [C1P2x, C1P2y]], [[C2P1x, C2P1y], [C2P2x, C2P2y]] ]  lista cu coordonatele tuturor
        # camerelor
        if all(np.all(point[0] != [np.nan, np.nan]) for point in tracked_points):
            if self.collecting_points_active and (
                    current_time - self.last_append_time >= 0.25) and not self.triangulating_points_active:
                self.triang_processor.tracked_points_for_camera_pose_list.append(
                    [point[0] for point in tracked_points]
                )
                self.last_append_time = current_time
            elif self.triangulating_points_active:
                errors, object_points, frames = self.triang_processor.determine_object_points(tracked_points, frames)

        return frames

    def undistort_frame(self, frame, camera_num):
        """
        Corecteaza distorsiunile dintr-un cadru capturat de o camera specificata.

        Args:
            frame (numpy.ndarray): Cadrul care trebuie corectat.
            camera_num (int): Numarul camerei.

        Returns:
            numpy.ndarray: Cadrul corectat.
        """
        h, w = frame.shape[:2]

        params = self.get_camera_calibration_params(camera_num)
        intrinsic_matrix = params['intrinsic_matrix']
        distortion_coef = params['distortion_coef']

        newcameramtx, _ = cv.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coef, (w, h), 1, (w, h))
        undistorted_frame = cv.undistort(frame, intrinsic_matrix, distortion_coef, None, newcameramtx)

        return undistorted_frame

    def apply_marker_enhancement(self, frame):
        """
        Aplica imbunatatiri asupra markerelor dintr-un cadru dat pentru a imbunatati detectia acestora.

        Args:
            frame (numpy.ndarray): Cadrul care trebuie imbunatatit.

        Returns:
            numpy.ndarray: Cadrul imbunatatit.
        """
        # convertire la grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # aplicare GaussianBlur pentru a reduce zgomotul
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # aplicare treshold adaptiv pentru a crea o masca binara
        adaptive_thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        # aplicare operatii morfologice pentru a rafina masca
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        morphed = cv.morphologyEx(adaptive_thresh, cv.MORPH_CLOSE, kernel)

        # utilizarea mastii pentru a imbunatati detectia markerelor IR in imaginea grayscale originala
        enhanced = cv.bitwise_and(gray, gray, mask=morphed)

        return cv.cvtColor(enhanced, cv.COLOR_GRAY2BGR)

    def get_combined_frames(self):
        """
        Obtine cadrele combinate de la toate camerele active, aplicand corectii si imbunatatiri.

        Returns:
            numpy.ndarray: Cadrele combinate intr-o singura imagine.
        """
        try:
            frames, _ = self.camera_devices.read()

            if frames is None:
                self.logger.error("Eroare in obtinerea cadrelor combinate: 'frames' este None")
                return None

            for i in range(0, self.active_camera_count):
                frames[i] = self.undistort_frame(frames[i], i)
                frames[i] = self.apply_marker_enhancement(frames[i])
                frames[i] = convert_to_bgr(frames[i])

            frames = self.process_frames(frames)

            if frames is None:
                self.logger.error("Eroare in obtinerea cadrelor combinate: 'frames' este None")
                return None

            for i in range(0, self.active_camera_count):
                frames[i] = self.undistort_frame(frames[i], i)
                frames[i] = self.apply_marker_enhancement(frames[i])
                frames[i] = convert_to_bgr(frames[i])

                text = f'Camera {i + 1}'
                font_scale = 0.5
                thickness = 1
                color = (255, 255, 255)
                text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, font_scale, thickness)
                text_position = (10, text_size[1] + 10)

                cv.putText(frames[i], text, text_position, cv.FONT_HERSHEY_COMPLEX, font_scale, color, thickness,
                           cv.LINE_AA)

            if self.active_camera_count <= 2:
                rows, cols = 1, self.active_camera_count
            else:
                rows = (self.active_camera_count + 1) // 2
                cols = 2

            frame_height, frame_width, _ = frames[0].shape
            grid_height = rows * frame_height
            grid_width = cols * frame_width

            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

            for i, frame in enumerate(frames):
                row = i // cols
                col = i % cols
                y_start = row * frame_height
                y_end = (row + 1) * frame_height
                x_start = col * frame_width
                x_end = (col + 1) * frame_width
                grid[y_start:y_end, x_start:x_end, :] = frame

            return grid
        except Exception as e:
            self.logger.error(f"Eroare in obtinerea cadrelor combinate: {str(e)}")
            return None

    def start_capture(self):
        """Porneste capturarea punctelor de interes de la camere."""
        self.collecting_points_active = True
        self.triang_processor.tracked_points_for_camera_pose_list = []

    def stop_capture(self):
        """Opreste capturarea punctelor de interes de la camere."""
        self.collecting_points_active = False

    def start_triangulating(self):
        """Porneste procesul de triangulare a punctelor de interes."""
        self.triangulating_points_active = True

    def stop_triangulating(self):
        """Opreste procesul de triangulare a punctelor de interes."""
        self.triangulating_points_active = False
