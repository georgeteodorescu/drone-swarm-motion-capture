# ---------------------------------------------------------------------
# -------------------- triangulation_processor.py ---------------------
# ---------------------------------------------------------------------
import copy
import traceback

import cv2 as cv
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from logger import Logger
from utils import DLT, draw_epipolar_lines


class TriangulationWorker(QObject):
    """
    Clasa care gestioneaza optimizarea pozitiilor camerelor intr-un thread separat.
    """

    finished = pyqtSignal()
    updateCameraPoses = pyqtSignal(list)

    def __init__(self, triang_processor):
        super().__init__()

        self.triang_processor = triang_processor

    def estimate_camera_poses_worker(self):
        """
        Metoda care apeleaza functia de estimare a pozitiei camerei si emite semnale la finalizare.
        """
        self.triang_processor.estimate_camera_poses()
        self.updateCameraPoses.emit(self.triang_processor.camera_poses)
        self.finished.emit()


def filter_tracked_points(tracked_points):
    """
    Filtreaza punctele urmarite pentru a elimina valorile NaN.

    Args:
        tracked_points (list): Lista de puncte urmarite.

    Returns:
        list: Lista de puncte urmarite fara NaN.
    """
    filtered_tracked_points = []
    for camera_points in tracked_points:
        filtered_camera_points = [
            point for point in camera_points if not np.isnan(np.array(point, dtype=np.float64)).any()
        ]
        filtered_tracked_points.append(filtered_camera_points)
    return filtered_tracked_points


def initialize_correspondences_and_root_points(filtered_tracked_points):
    """
    Initializeaza corespondentele si punctele radacina.

    Args:
        filtered_tracked_points (list): Lista de puncte urmarite filtrate.

    Returns:
        tuple: Corespondentele si punctele radacina.
    """
    correspondences = [[[i]] for i in filtered_tracked_points[0]]
    root_image_points = [{"camera": 0, "point": point} for point in filtered_tracked_points[0]]
    return correspondences, root_image_points


def compute_epipolar_lines(projection_matrices, root_image_points, current_camera_index, frames):
    """
    Calculeaza liniile epipolare pentru punctele radacina.

    Args:
        projection_matrices (list): Listele de matrice de proiectie.
        root_image_points (list): Lista de puncte radacina.
        current_camera_index (int): Indexul camerei curente.
        frames (list): Listele de cadre.

    Returns:
        list: Liniile epipolare.
    """
    epipolar_lines = []
    for root_image_point in root_image_points:
        F = cv.sfm.fundamentalFromProjections(projection_matrices[root_image_point["camera"]],
                                              projection_matrices[current_camera_index])
        line = cv.computeCorrespondEpilines(np.array([root_image_point["point"]], dtype=np.float32), 1, F)
        epipolar_lines.append(line[0, 0].tolist())
        frames[current_camera_index] = draw_epipolar_lines(frames[current_camera_index], line[0])
    return epipolar_lines


def find_potential_matches(points, epipolar_line):
    """
    Gaseste punctele potentiale care corespund unei linii epipolare.

    Args:
        points (numpy.ndarray): Punctele de verificat.
        epipolar_line (list): Linia epipolara.

    Returns:
        numpy.ndarray: Punctele potentiale care corespund.
    """
    a, b, c = epipolar_line
    distances_to_line = np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a ** 2 + b ** 2)
    potential_matches = points[distances_to_line < 5]
    distances_to_line = distances_to_line[distances_to_line < 5]
    potential_matches_sorter = distances_to_line.argsort()
    return potential_matches[potential_matches_sorter]


def update_correspondences_with_matches(correspondences, unmatched_image_points, potential_matches, index):
    """
    Actualizeaza corespondentele cu punctele potentiale gasite.

    Args:
        correspondences (list): Lista de corespondente.
        unmatched_image_points (numpy.ndarray): Punctele de imagine necorespunzatoare.
        potential_matches (numpy.ndarray): Punctele potentiale care corespund.
        index (int): Indexul curent.

    Returns:
        numpy.ndarray: Punctele de imagine necorespunzatoare actualizate.
    """
    if len(potential_matches) == 0:
        for group in correspondences[index]:
            group.append([None, None])
    else:
        unmatched_image_points = [
            row for row in unmatched_image_points.tolist() if row != potential_matches.tolist()[0]
        ]
        unmatched_image_points = np.array(unmatched_image_points)
        new_correspondences = []
        for match in potential_matches:
            temp = copy.deepcopy(correspondences[index])
            for group in temp:
                group.append(match.tolist())
            new_correspondences += temp
        correspondences[index] = new_correspondences
    return unmatched_image_points


class TriangulationProcessor:
    """
    Clasa pentru procesarea triangulatiei, inclusiv estimarea pozitiei camerei si triangularea punctelor.
    """
    logger: Logger

    def __init__(self, camera_manager):
        """
        Initializeaza procesorul de triangulatie.

        Args:
            camera_manager (CameraManager): Managerul camerelor.
        """
        self.logger = Logger(log_to_file=True)
        self.logger.info("Initializare Triangulation Processor...")

        self.camera_manager = camera_manager
        self.camera_poses = np.nan
        self.reprojection_errors = np.nan

        self.tracked_points_for_camera_pose_list = []
        self.errors_list = []

    def update_camera_poses(self, camera_poses):
        """
        Actualizeaza pozitiile camerelor si emite semnalul de actualizare.

        Args:
            camera_poses (list): Lista de pozitii ale camerelor.
        """
        self.camera_poses = camera_poses
        self.camera_manager.camera_poses_updated.emit(self.camera_poses)

    def start_estimate_camera_poses_in_thread(self):
        """
        Porneste estimarea pozitiei camerei intr-un thread separat.
        """
        self.thread = QThread()
        self.worker = TriangulationWorker(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.estimate_camera_poses_worker)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # conexiune pentru a gestiona in siguranta firul de executie daca se incheie in mod neasteptat
        self.thread.finished.connect(lambda: self.thread is None or self.thread.deleteLater())

        self.worker.updateCameraPoses.connect(self.update_camera_poses)

        self.thread.start()

    def estimate_camera_poses(self):
        """
        Estimeaza pozitiile camerelor folosind punctele urmarite.
        """
        tracked_points = np.array(self.tracked_points_for_camera_pose_list)
        self.logger.debug(f"Dimensiunea tracked_points: {tracked_points.shape}")
        # self.logger.debug(f"tracked_points: \n{tracked_points}")

        nan_count = np.isnan(tracked_points).sum()
        self.logger.debug(f"Numar initial de valori NaN in tracked_points: {nan_count}")

        tracked_points_T = tracked_points.transpose((1, 0, 2))
        self.logger.debug(f"Dimensiunea tracked_points_T: {tracked_points_T.shape}")
        # self.logger.debug(f"tracked_points_T: \n{tracked_points_T}")

        nan_count_T = np.isnan(tracked_points_T).sum()
        self.logger.debug(f"Numar initial de valori NaN in tracked_points_T dupa transpunere: {nan_count_T}")

        camera_poses = [{"R": np.eye(3), "t": np.zeros((3, 1))}]

        self.logger.debug("==== Inceperea Calcularii Pozitiei Camerelor ====")
        for camera_index in range(0, self.camera_manager.active_camera_count - 1):
            self.logger.debug(f"Perechea de Camere: {camera_index + 1} & {camera_index + 2}")

            current_camera_points, next_camera_points = self.get_valid_camera_points(tracked_points_T, camera_index)
            if current_camera_points is None or next_camera_points is None:
                continue

            # utilizare o valoare de incredere ridicata pentru RANSAC 99.999%
            F, mask = cv.findFundamentalMat(current_camera_points, next_camera_points, cv.FM_RANSAC, 0.1, 0.9999)

            if F is None or F.shape != (3, 3):
                self.logger.critical(
                    f"Nu s-a reusit calcularea Matricei Fundamentale pentru perechea de Camere: {camera_index + 1}, {camera_index + 2}.")
                continue

            self.logger.debug(f"Matricea fundamentala pentru Camerele {camera_index + 1} si {camera_index + 2}:\n{F}")

            K1 = self.camera_manager.get_camera_calibration_params(camera_index)["intrinsic_matrix"].astype(np.float64)
            K2 = self.camera_manager.get_camera_calibration_params(camera_index + 1)["intrinsic_matrix"].astype(
                np.float64)

            E = cv.sfm.essentialFromFundamental(F, K1, K2)
            self.logger.debug(f"\t Matricea Esentiala:\n{E}")

            R, t = self.select_best_pose(E, current_camera_points, next_camera_points, camera_poses)

            if R is np.nan or t is np.nan:
                self.logger.critical(
                    f"Nu s-a reusit selectarea unei pozitii valide pentru perechea de Camere: {camera_index + 1}, {camera_index + 2}."
                )
                continue

            camera_poses.append({
                "R": R @ camera_poses[-1]["R"],
                "t": camera_poses[-1]["t"] + (camera_poses[-1]["R"] @ t)
            })

            self.logger.debug(f"Actualizare Pozitie pentru Camera {camera_index + 1}:")
            self.logger.debug(f"\t R:\n{R}")
            self.logger.debug(f"\t t:\n{t}")

        camera_poses = self.optimize_camera_poses(tracked_points, camera_poses)
        object_points = self.triangulate_multiple_points(tracked_points, camera_poses)
        errors = np.mean(self.calculate_reprojection_errors(tracked_points, object_points, camera_poses))
        # camera_poses = [
        #     {'R': np.array([[1., 0., 0.],
        #                     [0., 1., 0.],
        #                     [0., 0., 1.]]), 't': np.array([0., 0., 0.])},
        #     {'R': np.array([[-0.0008290000610233772, -0.7947131755287576, 0.6069845808584402],
        #                     [0.7624444396180684, 0.3922492478955913, 0.5146056781855716],
        #                     [-0.6470531579819294, 0.46321862674804054, 0.6055994671226776]]),
        #      't': np.array([-2.6049886186449047, -2.173986915510569, 0.7303458563542193])},
        #     {'R': np.array([[-0.9985541623963866, -0.028079891357569067, -0.045837806036037466],
        #                     [-0.043210651917521686, -0.08793122558361385, 0.9951888962042462],
        #                     [-0.03197537054848707, 0.995730696156702, 0.0865907408997996]]),
        #      't': np.array([0.8953888630067902, -3.4302652822708373, 3.70967106300893])},
        #     {'R': np.array([[-0.4499864100408215, 0.6855400696798954, -0.5723172578577878],
        #                     [-0.7145273934510732, 0.10804105689305427, 0.6912146801345055],
        #                     [0.5356891214002657, 0.7199735709654319, 0.4412201517663212]]),
        #      't': np.array([2.50141072072536, -2.313616767292231, 1.8529907514099284])}
        # ]
        self.update_camera_poses(camera_poses)
        self.reprojection_errors = errors

        self.logger.debug(f"Pozitii Finale ale Camerelor: \n{camera_poses}")
        self.logger.debug(f'Erori de Reproiectie Finale: \n{self.reprojection_errors}')
        self.logger.debug("==== Finalizarea Calculararii Pozitiilor Finale a Camerelor ====")

    def get_valid_camera_points(self, tracked_points_T, camera_index):
        """
        Obtine punctele valide pentru camerele curente si urmatoare.

        Args:
            tracked_points_T (numpy.ndarray): Punctele urmarite transpuse.
            camera_index (int): Indexul camerei curente.

        Returns:
            tuple: Punctele valide pentru camerele curente si urmatoare.
        """
        current_camera_points = tracked_points_T[camera_index]
        next_camera_points = tracked_points_T[camera_index + 1]

        # self.logger.debug(f"Raw current_camera_points (camera {camera_index + 1}):\n{current_camera_points}")
        # self.logger.debug(f"Raw next_camera_points (camera {camera_index + 2}):\n{next_camera_points}")

        # valid_indices = np.where(~np.isnan(current_camera_points).any(axis=1) & ~np.isnan(next_camera_points).any(axis=1))[0]
        valid_indices = np.where(np.all(~np.isnan(current_camera_points), axis=1) & np.all(~np.isnan(next_camera_points), axis=1))[0]

        if valid_indices.size == 0:
            self.logger.critical(
                f"Nu s-au gasit puncte valide pentru perechea de Camere: {camera_index + 1}, {camera_index + 2}."
            )
            return None, None

        current_camera_points = np.take(current_camera_points, valid_indices, axis=0).astype(np.float64)
        next_camera_points = np.take(next_camera_points, valid_indices, axis=0).astype(np.float64)

        self.logger.debug(f"\t Puncte Valide: {len(valid_indices)}")
        self.logger.debug(f"\t Dimensiunea current_camera_points: {current_camera_points.shape}")
        # self.logger.debug(f"current_camera_points: \n{current_camera_points}")
        self.logger.debug(f"\t Dimensiunea next_camera_points: {next_camera_points.shape}")
        # self.logger.debug(f"next_camera_points: \n{next_camera_points}")

        return current_camera_points, next_camera_points

    def select_best_pose(self, E, current_camera_points, next_camera_points, camera_poses):
        """
        Selecteaza cea mai buna rotatie si translatie pentru camerele curente si urmatoare.

        Args:
            E (numpy.ndarray): Matricea esentiala.
            current_camera_points (numpy.ndarray): Punctele camerei curente.
            next_camera_points (numpy.ndarray): Punctele camerei urmatoare.
            camera_poses (list): Pozitiile camerelor.

        Returns:
            tuple: Cea mai buna rotatie si translatie.
        """
        # functie pentru a calcula punctele obiect folosind rotatia si translatia data
        def compute_object_points(R, t, current_points, next_points, prev_pose):
            # combianrea punctelor curente si urmatoare intr-o matrice
            stacked_points = np.hstack([
                np.expand_dims(current_points, axis=1),
                np.expand_dims(next_points, axis=1)
            ])
            # self.logger.debug(f"[select_best_pose] stacked_points: {stacked_points}")

            # combinarea pozitiilor anterioare ale camerei cu noua rotatie si translatie
            poses = np.concatenate([prev_pose, [{"R": R, "t": t}]])
            # self.logger.debug(f"[select_best_pose] poses: {poses}")

            # triangularea punctelor obiectului
            object_points = self.triangulate_multiple_points(stacked_points, poses)
            # self.logger.debug(f"object_points: {object_points}")

            return object_points

        def count_points_in_front(object_points, R):
            # transformarea punctelor obiectului in cadrul de coordonate al camerei
            transformed_points = np.array([R.T @ point for point in object_points])
            # numararea punctelor aflate in fata camerei
            points_in_front = np.sum(object_points[:, 2] > 0) + np.sum(transformed_points[:, 2] > 0)
            return points_in_front

        # extragerea posibilelor rotatii si translatii din matricea esentiala
        possible_rotations, possible_translations = cv.sfm.motionFromEssential(E)
        best_R, best_t = np.nan, np.nan  # initializare rotatii si translatii
        max_points_in_front = 0  # nr. max de puncte in fata camerei

        # iterare prin toate combinatiile de rotatii si translatii posibile
        for i in range(4):
            R, t = possible_rotations[i], possible_translations[i]  # extragere rotatii si translatii
            # calcularea punctelor obiectului folosind rotatia si translatia curenta
            object_points = compute_object_points(R, t, current_camera_points, next_camera_points, [camera_poses[-1]])
            # numara punctele obiectului aflate in fata camerei
            points_in_front = count_points_in_front(object_points, R)

            # daca numarul de puncte in fata camerei este mai mare decat cel maxim gasit pana acum, actualizeaza
            # valorile
            if points_in_front > max_points_in_front:
                max_points_in_front = points_in_front
                best_R, best_t = R, t

        # returneaza cea mai buna rotatie si translatie gasita
        return best_R, best_t

    def determine_object_points(self, tracked_points, frames):
        """
        Determina punctele obiectului din punctele urmarite si cadrele camerelor.

        Args:
            tracked_points (list): Lista de puncte urmarite.
            frames (list): Lista de cadre ale camerelor.

        Returns:
            tuple: Erori, puncte obiect si frame-uri actualizate.
        """
        try:
            filtered_tracked_points = filter_tracked_points(tracked_points)
            self.logger.debug(f"Filtered tracked points: {filtered_tracked_points}")

            correspondences, root_image_points = initialize_correspondences_and_root_points(filtered_tracked_points)

            projection_matrices = [
                self.camera_manager.get_camera_calibration_params(i)["intrinsic_matrix"] @ np.c_[pose["R"], pose["t"]]
                for i, pose in enumerate(self.camera_poses)
            ]

            for i in range(1, len(self.camera_poses)):
                epipolar_lines = compute_epipolar_lines(projection_matrices, root_image_points, i, frames)

                unmatched_image_points = np.array(tracked_points[i], dtype=np.float64)
                points = np.array(tracked_points[i], dtype=np.float64)

                for j, epipolar_line in enumerate(epipolar_lines):
                    potential_matches = find_potential_matches(points, epipolar_line)
                    unmatched_image_points = update_correspondences_with_matches(
                        correspondences, unmatched_image_points, potential_matches, j
                    )

                for unmatched_image_point in unmatched_image_points:
                    root_image_points.append({"camera": i, "point": unmatched_image_point})
                    temp = [[[None, None]] * i]
                    temp[0].append(unmatched_image_point.tolist())
                    correspondences.append(temp)

            object_points = []
            errors = []
            for image_points in correspondences:
                object_points_i = self.triangulate_multiple_points(image_points, self.camera_poses)

                if object_points_i.size == 0:
                    self.logger.debug(f"Skipped empty object_points_i for image_points: {image_points}")
                    continue

                errors_i = self.calculate_reprojection_errors(image_points, object_points_i, self.camera_poses)

                if errors_i.size == 0:
                    self.logger.debug(f"Skipped empty errors_i for object_points_i: {object_points_i}")
                    continue

                object_points.append(object_points_i[np.argmin(errors_i)])
                errors.append(np.min(errors_i))

            return np.array(errors), np.array(object_points), frames
        except Exception as e:
            self.logger.error(f"Error in determine_point_correspondences_and_object_points: {str(e)}")
            self.logger.error(f"Stack Trace: {traceback.format_exc()}")
            raise e

    def calculate_reprojection_errors(self, tracked_points, object_points, camera_poses):
        """
        Calculeaza erorile de reproiectie pentru punctele urmarite si punctele obiectului.

        Args:
            tracked_points (list): Lista de puncte urmarite.
            object_points (numpy.ndarray): Punctele obiectului.
            camera_poses (list): Pozitiile camerelor.

        Returns:
            numpy.ndarray: Erorile de reproiectie.
        """
        errors = np.array([])
        for tracked_point, object_point in zip(tracked_points, object_points):
            # if np.any(np.isnan(object_point)) or np.any(np.isinf(object_point)):
            #     self.logger.error(f"[calculate_reprojection_errors] Punect obiect nu este valid: {object_point}")
            #     continue
            error = self.calculate_reprojection_error(tracked_point, object_point, camera_poses)
            if error is not np.nan:
                errors = np.concatenate([errors, [error]])

        if errors.size > 0:  # verificare daca matricea de erori are elemente
            return np.array(errors)
        else:
            self.logger.error(f"[calculate_reprojection_errors] Erori de reproiectie 0")
            return np.array([0])

    def calculate_reprojection_error(self, tracked_point, object_point, camera_poses):
        """
        Calculeaza eroarea de reproiectie pentru un punct urmarit si un punct obiect.

        Args:
            tracked_point (numpy.ndarray): Punctul urmarit.
            object_point (numpy.ndarray): Punctul obiect.
            camera_poses (list): Pozitiile camerelor.

        Returns:
            float: Eroarea de reproiectie.
        """
        tracked_point = np.array(tracked_point)

        none_indices = np.where(np.all(tracked_point == [np.nan, np.nan], axis=1))[0]

        tracked_point = np.delete(tracked_point, none_indices, axis=0)
        camera_poses = np.delete(camera_poses, none_indices, axis=0)

        if not isinstance(tracked_point, np.ndarray) or tracked_point.ndim != 2 or tracked_point.size <= 1:
            self.logger.critical(f"[calculate_reprojection_error] Tracked_point trebuie sa fie o matrice 2D: {tracked_point}")
            return np.nan

        tracked_point_T = tracked_point.transpose((0, 1))

        errors = np.array([])
        for i, camera_pose in enumerate(camera_poses):
            if np.all(tracked_point[i] is np.nan, axis=0):
                continue

            # obtinerea parametrilor intrinseci ai camerei
            params = self.camera_manager.get_camera_calibration_params(i)
            intrinsic_matrix = params['intrinsic_matrix']
            distortion_coef = params['distortion_coef']
            # proiectarea punctelor 3D pe planul imaginii folosind modelul de camera cu orificiu stenopeic
            projected_img_points, _ = cv.projectPoints(
                np.expand_dims(object_point, axis=0).astype(np.float64),
                np.array(camera_pose["R"], dtype=np.float64),
                np.array(camera_pose["t"], dtype=np.float64),
                intrinsic_matrix,
                distortion_coef
            )
            projected_img_point = projected_img_points[:, 0, :][0]

            if tracked_point_T[i] is None or np.any(tracked_point_T[i] == [None, None]):
                self.logger.error(
                    f"tracked_point_T[{i}] or projected_img_point is None: {tracked_point_T[i]}, {projected_img_point}")
                continue

            errors = np.concatenate([errors, (tracked_point_T[i] - projected_img_point).flatten() ** 2])

        return errors.mean()

    def optimize_camera_poses(self, tracked_points, camera_poses):
        """
        Optimizeaza pozitiile camerelor folosind punctele urmarite.

        Args:
            tracked_points (list): Lista de puncte urmarite.
            camera_poses (list): Pozitiile camerelor.

        Returns:
            list: Pozitiile optimizate ale camerelor.
        """
        self.logger.debug("---- INITIALIZARE OPTIMIZARE POZITII CAMERE ----")
        self.logger.debug(f"Pozitii initiale ale camerelor: {camera_poses}")

        # functie pentru a extrage pozitii ale camerei din parametri
        def extract_camera_poses_from_params(params):
            num_cameras = (params.size - 1) // 7 + 1
            # matricea identitatii pentru rotatie si vectorul de translatie initial
            ecpfp_camera_poses = [{"R": np.eye(3), "t": np.zeros(3)}]

            for i in range(0, num_cameras - 1):
                start = i * 7 + 1
                rot_vec = params[start + 1:start + 4]
                t_vec = params[start + 4:start + 7]
                ecpfp_camera_poses.append({"R": Rotation.from_rotvec(rot_vec).as_matrix(), "t": t_vec})

            return ecpfp_camera_poses

        def initialize_params(ip_camera_poses):
            """
            Initializare parametri pentru optimizarea pozitiilor camerelor.
            """
            # obtinerea distantei focale initiale
            focal_distance = self.camera_manager.get_camera_calibration_params(0)['focal_distance']
            if isinstance(focal_distance, np.ndarray):
                focal_distance = focal_distance.item()

            # initializarea parametrilor initiali
            ip_init_params = np.array([focal_distance])
            for i, pose in enumerate(ip_camera_poses[1:]):
                params = self.camera_manager.get_camera_calibration_params(i)
                focal_distance = params['focal_distance']
                if isinstance(focal_distance, np.ndarray):
                    focal_distance = np.mean(focal_distance)
                # daca distanta focala este NaN, o inlocuim cu prima valoare din matricea intrinseca
                focal_distance = np.nan_to_num(focal_distance, nan=params['intrinsic_matrix'][0, 0])

                R = Rotation.from_matrix(pose['R']).as_rotvec().flatten()
                t = pose['t'].flatten()
                ip_init_params = np.concatenate([ip_init_params, [focal_distance], R, t])

                self.logger.debug(f"Distanta focala (camera {i + 1}): {focal_distance}")
                self.logger.debug(f"Vectorul de rotatie (camera {i + 1}): {R}")
                self.logger.debug(f"Vectorul de translatie (camera {i + 1}): {t}")

            return np.array(ip_init_params, dtype=np.float64)

        # functie pentru a calcula valorile erorilor
        def compute_error_values(params, cev_tracked_points):
            cev_camera_poses = extract_camera_poses_from_params(params)
            object_points = self.triangulate_multiple_points(cev_tracked_points, cev_camera_poses)
            errors = self.calculate_reprojection_errors(cev_tracked_points, object_points, cev_camera_poses)
            if not np.all(np.isfinite(errors)):
                self.logger.critical(f"Reziduuri nenule finite detectate {errors}")
                return np.full_like(errors, 1000)  # penalizare mare pentru reziduuri nenule finite
            return errors

        # verificare daca parametrii initiali sunt finiti
        init_params = initialize_params(camera_poses)
        if not np.all(np.isfinite(init_params)):
            self.logger.error(f"Valori nenule finite gasite in parametrii initiali:: {init_params}")
            return camera_poses

        # optimizare folosind least_squares
        res = least_squares(
            lambda params: compute_error_values(params, tracked_points)
            , init_params, method='trf', loss="cauchy", ftol=1e-5, xtol=1e-5, gtol=1e-5, verbose=2
        )

        camera_poses = extract_camera_poses_from_params(res.x)

        self.logger.debug(f"Rezultatul optimizarii: {res}")
        self.logger.debug("Pozitii finale ale camerelor dupa optimizare:")
        for idx, pose in enumerate(camera_poses):
            self.logger.debug(f"Camera {idx}: R={pose['R']}, t={pose['t']}")

        self.logger.debug("---- FINALIZARE OPTIMIZARE POZITII CAMERE ----")
        return camera_poses

    def triangulate_single_point(self, tracked_point, camera_poses):
        """
        Trianguleaza un singur punct folosind pozitiile camerelor.

        Args:
            tracked_point (list): Lista de puncte urmarite.
            camera_poses (list): Pozitiile camerelor.

        Returns:
            numpy.ndarray: Punctul triangulat.
        """
        tracked_point = np.array(tracked_point)
        # if tracked_point.ndim != 2:
        #     self.logger.warning("[triangulate_single_point] Forma tracked_point detectata ca fiind nevalida. Se "
        #                         "asteapta o matrice 2D.")
        #     return [np.nan, np.nan, np.nan]

        nan_indices = np.where(np.all(np.isnan(tracked_point), axis=1))[0]
        # self.logger.debug(f"[triangulate_single_point] nan_indices: {nan_indices}")

        if np.any(nan_indices >= len(camera_poses)):
            self.logger.critical(f"[triangulate_single_point] nan_indices out of bounds: {nan_indices}")
            return [np.nan, np.nan, np.nan]

        tracked_point = np.delete(tracked_point, nan_indices, axis=0)
        camera_poses = np.delete(camera_poses, nan_indices, axis=0)

        if len(tracked_point) <= 1:
            self.logger.critical("[triangulate_single_point] Puncte de urmarire valide insuficiente dupa eliminarea NaNs.")
            return [np.nan, np.nan, np.nan]

        Ps = [self.camera_manager.get_camera_calibration_params(i)["intrinsic_matrix"] @ np.c_[pose["R"], pose["t"]] for i, pose in enumerate(camera_poses)]

        if len(Ps) != tracked_point.shape[0]:
            self.logger.critical(
                f"[triangulate_single_point] Dimensiuni incompatibile: len(Ps)={len(Ps)}, tracked_point.shape={tracked_point.shape}")
            return [np.nan, np.nan, np.nan]

        return DLT(Ps, tracked_point)

    def triangulate_multiple_points(self, tracked_points, camera_poses):
        """
        Trianguleaza multiple puncte folosind pozitiile camerelor.

        Args:
            tracked_points (list): Lista de puncte urmarite.
            camera_poses (list): Pozitiile camerelor.

        Returns:
            numpy.ndarray: Punctele triangulate.
        """
        object_points = []
        for tracked_point in tracked_points:
            tracked_point = np.array(tracked_point, dtype=np.float64)
            if np.isnan(tracked_point).all():
                self.logger.critical(f"[triangulate_multiple_points] Omitere punct urmarit care nu este valid din cauza Nans: {tracked_point}")
                continue
            object_point = self.triangulate_single_point(tracked_point, camera_poses)
            if not isinstance(object_point, np.ndarray):
                try:
                    object_point = np.array(object_point, dtype=np.float64)
                except ValueError:
                    self.logger.error(f"[triangulate_multiple_points] S-a esuat conversia punctului obiectului in matrice float: {object_point}")

            if np.any(np.isnan(object_point)) or np.any(np.isinf(object_point)):
                self.logger.critical(f"[triangulate_multiple_points] Punct obiect detectat nu este valid: {object_point}")
                continue

            object_points.append(object_point)

        return np.array(object_points)
