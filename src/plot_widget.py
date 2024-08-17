# -----------------------------------------------------------
# -------------------- plot_widget.py ------------------------
# -----------------------------------------------------------
import numpy as np
import matplotlib
from logger import Logger
from utils import create_cone_vertices, create_flip_matrix, create_scale_matrix

matplotlib.use('QtAgg')
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation as R


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, edgecolor='w', facecolor='w')
        self.axes = fig.add_subplot(111, projection='3d')
        self.axes.set_position([0, 0, 1, 1])
        super().__init__(fig)


def default_camera_poses(n_cameras=6, radius=4, z=2):
    default_poses = []
    for i in range(n_cameras):
        angle_rad = 2 * np.pi * i / n_cameras
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)

        azimuth = np.degrees(angle_rad) + 180

        distance_to_center = np.sqrt(x ** 2 + y ** 2)
        elevation = -np.degrees(np.arctan2(z, distance_to_center))

        rotation = R.from_euler('zy', [azimuth, elevation], degrees=True)
        rotation_matrix = rotation.as_matrix()

        default_poses.append({
            "R": rotation_matrix,
            "t": [x, y, z]
        })

    return default_poses


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.logger = Logger(log_to_file=True)
        self.logger.info("Initializare Plot Widget...")

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.update_plot(default_camera_poses(4))

    def update_plot(self, camera_poses):
        self.canvas.axes.clear()
        for pose in camera_poses:
            self.plot_cone(pose['t'], pose['R'])
        self.canvas.draw()

    def plot_cone(self, position, rotation_matrix, height=0.25, base=0.25):
        def apply_transformations(verts, trans_matrix):
            verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
            transformed_verts_h = verts_h @ trans_matrix.T
            transformed_verts = transformed_verts_h[:, :3] / transformed_verts_h[:, 3].reshape(-1, 1)
            return transformed_verts

        def plot_edges(vert, i):
            for start, end in i:
                self.canvas.axes.plot(
                    [vert[start, 0], vert[end, 0]],
                    [vert[start, 1], vert[end, 1]],
                    [vert[start, 2], vert[end, 2]],
                    color="b"
                )

        vertices = create_cone_vertices(height, base)
        scale_matrix = create_scale_matrix(0.5)
        flip_matrix = create_flip_matrix()

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        # Apply transformations: flip -> scale -> rotation -> translation
        transformation_matrix = flip_matrix @ scale_matrix @ transformation_matrix

        transformed_vertices = apply_transformations(vertices, transformation_matrix)

        # Create indices for the cone edges
        indices = [(0, i) for i in range(1, len(vertices))]
        indices += [(i, i + 1) for i in range(1, len(vertices) - 1)]
        indices.append((len(vertices) - 1, 1))

        plot_edges(transformed_vertices, indices)
