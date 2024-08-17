# ---------------------------------------------------------------------
# --------------------------- utils.py --------------------------------
# ---------------------------------------------------------------------
import glob
import sys

import serial
import cv2 as cv
import numpy as np
from scipy import linalg

from logger import Logger

logger = Logger(log_to_file=True)


def serial_ports():
    """ https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
        Listeaza porturile seriale disponibile pe sistem.
        :raises EnvironmentError:
            Daca platforma este necunoscuta sau nesuportata
        :returns:
            O lista cu porturile seriale disponibile pe sistem.
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial-*')
    else:
        logger.critical('Unsupported platform')
        quit()

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def draw_epipolar_lines(image, epipolar_lines):
    """
    Deseneaza linii epipolare pe o imagine data.

    Parameters:
        image (numpy.ndarray): Imaginea pe care se deseneaza liniile.
        epipolar_lines (numpy.ndarray): Liniile epipolare care trebuie desenate.

    Returns:
        numpy.ndarray: Imaginea cu liniile epipolare desenate.
    """
    rows, cols, _ = image.shape
    for line in epipolar_lines:
        x_start, y_start = map(int, [0, -line[2] / line[1]])
        x_end, y_end = map(int, [cols, -(line[2] + line[0] * cols) / line[1]])
        image = cv.line(image, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)
    return image


def detect_and_annotate_points(frame):
    """
    Detecteaza si anoteaza punctele de interes intr-un cadru dat.

    Args:
        frame (numpy.ndarray): Cadrul in care se detecteaza punctele.

    Returns:
        tuple:
            numpy.ndarray: Cadrul cu punctele adnotate.
            list: Lista cu coordonatele centroizilor detectati.
    """
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    _, binary = cv.threshold(gray, 255 * 0.2, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # calculare memente invariante pentru toate contururile, apoi filtrare pe baza 'm00'
    moments = [cv.moments(cnt) for cnt in contours]
    valid_moments = [m for m in moments if m['m00'] != 0]  # filtrarea momentelor in care 'm00' este zero

    # calcularea centroizilor pentru momentele valide
    centroids = np.array([[int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])] for m in valid_moments])

    # # daca sunt gasite mai multe centroide, se pastreaza cel mai apropiat de centru
    # if len(centroids) > 1:
    #     height, width = frame.shape[:2]
    #     center = np.array([width // 2, height // 2])
    #
    #     distances = np.linalg.norm(centroids - center, axis=1)
    #     closest_point_index = np.argmin(distances)
    #     centroids = np.array([centroids[closest_point_index]])

    # trasare toate contururile si cel mai apropiat centroid sau singurul centroid daca se gaseste doar unul
    for (x, y) in centroids:
        cv.circle(frame, (x, y), 2, (255, 255, 255), -1)
        text = f'({x}, {y})'
        font_scale = 0.4
        thickness = 1
        color = (255, 255, 255)
        text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, font_scale, thickness)
        text_width, text_height = text_size
        text_x = min(x + 5, frame.shape[1] - text_width - 5)
        text_y = min(y + text_height + 5, frame.shape[0] - 5)

        cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_COMPLEX, font_scale, color, thickness, cv.LINE_AA)

    frame = cv.drawContours(frame, contours, -1, (255, 0, 0), 1)

    if len(centroids) == 0:
        return frame, [[np.nan, np.nan]]
    else:
        return frame, centroids.tolist()


# https://stackoverflow.com/questions/76980602/direct-linear-transformation-in-c-from-python
def DLT(Ps, tracked_point):
    """
    Trianguleaza un punct 3D folosind metoda DLT (Direct Linear Transformation).

    Args:
        Ps (list): Lista de matrici de proiectie ale camerelor.
        tracked_point (numpy.ndarray): Coordonatele 2D ale punctului urmarit in imaginile camerelor.

    Returns:
        numpy.ndarray: Coordonatele 3D ale punctului triangulat.
    """
    A = []
    for P, point in zip(Ps, tracked_point):
        row1 = point[1] * P[2, :] - P[1, :]
        row2 = P[0, :] - point[0] * P[2, :]
        A.append(row1)
        A.append(row2)

        if np.any(np.isnan(row1)) or np.any(np.isnan(row2)):
            logger.error("[DLT] Valori NaN detectate in formarea matricei DLT.")

    A = np.array(A)
    if A.shape[0] != len(Ps) * 2 or A.shape[1] != 4:
        logger.error(f"[DLT] Dimensiuni incompatibile pentru reshaping: A.shape={A.shape}, len(Ps)={len(Ps)}")
        return [np.nan, np.nan, np.nan]

    A = A.reshape((len(Ps) * 2, 4))
    if np.any(np.isnan(A)):
        logger.error("[DLT] Valori NaN prezente in matricea A.")

    B = A.transpose() @ A
    if np.any(np.isnan(B)):
        logger.critical("[DLT] B contine inf sau valori NaN.")
        return [np.nan, np.nan, np.nan]

    U, s, V = np.linalg.svd(B, full_matrices=False)
    if np.any(np.isnan(V)):
        logger.critical("[DLT] Calculul SVD a rezultat in valori NaN.")

    if V[-1, -1] != 0:
        return V[3, 0:3] / V[3, 3]
    else:
        logger.critical("[DLT] Calculul DLT a rezultat in valori NaN.")
        return [np.nan, np.nan, np.nan]


def create_cone_vertices(height, radius, num_base_points=10):
    """
    Creeaza varfurile unui con cu o inaltime si o raza specificata.

    Args:
        height (float): Inaltimea conului.
        radius (float): Raza bazei conului.
        num_base_points (int, optional): Numarul de puncte de pe baza conului. Default este 10.

    Returns:
        numpy.ndarray: Varfurile conului.
    """
    vertices = [[0, 0, 0]]
    angle_step = 2 * np.pi / num_base_points
    for i in range(num_base_points):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        vertices.append([x, y, z])
    return np.array(vertices)


def create_scale_matrix(scale_factor):
    """
    Creeaza o matrice de scalare.

    Args:
        scale_factor (float): Factorul de scalare.

    Returns:
        numpy.ndarray: Matricea de scalare.
    """
    return np.array([
        [scale_factor, 0, 0, 0],
        [0, scale_factor, 0, 0],
        [0, 0, scale_factor, 0],
        [0, 0, 0, 1]
    ])


def create_flip_matrix():
    """
    Creeaza o matrice pentru inversarea axelor.

    Returns:
        numpy.ndarray: Matricea de inversare a axelor.
    """
    return np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
