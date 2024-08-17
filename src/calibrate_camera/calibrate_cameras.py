import numpy as np
import cv2
import glob
import yaml

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

def calibrate_camera(images):
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    calibration_images = []
    successful_image_paths = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to read image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
            calibration_images.append(img)
            successful_image_paths.append(fname)  # Add the path to the list
        else:
            print(f"Chessboard corners not found in {fname}")

    return objpoints, imgpoints, calibration_images, successful_image_paths

def process_cameras():
    base_path = 'src/calibrate_camera/images/'
    camera_folders = [f'{base_path}cam{i}/' for i in range(1, 5)]
    calibration_results = {}

    rotation_values = {1: 1, 2: 1, 3: 1, 4: 1}

    for idx, folder in enumerate(camera_folders, start=1):
        images = glob.glob(f'{folder}*.jpg')
        objpoints, imgpoints, calibration_images, successful_image_paths = calibrate_camera(images)

        for img_path, img in zip(successful_image_paths, calibration_images):
            print(f"Displaying {img_path}")
            cv2.imshow(f'Camera {idx} Calibration', img)
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):
                    break

        if len(objpoints) > 0 and len(imgpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

            calibration_results[f'camera_{idx}'] = {
                'camera_matrix': np.asarray(mtx).tolist(),
                'dist_coeff': np.asarray(dist).tolist(),
                'rotation': rotation_values[idx]
            }

    cv2.destroyAllWindows()
    return calibration_results

def main():
    calibration_results = process_cameras()

    yaml_file = 'src/calibrate_camera/calibration_matrices.yaml'

    with open(yaml_file, "w") as f:
        yaml.dump(calibration_results, f)

    print(f'Calibration results saved to {yaml_file}.')

    # with open(yaml_file, "r") as file:
    #     calibration_data = yaml.safe_load(file)

    #     print(f'Calibration Data from {yaml_file}')
    #     pretty_data = json.dumps(calibration_data, indent=4)
    #     print(pretty_data)

if __name__ == "__main__":
    main()