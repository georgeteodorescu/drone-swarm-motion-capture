import numpy as np
import cv2 as cv
import yaml
import sys
sys.path.append('/home/george/Projects/drone-swarm-motion-capture')
from src.pseyepy import Camera

def load_camera_params(filepath="src/calibrate_camera/calibration_matrices.yaml"):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def init_frame(img, camera_num, camera_params): 
    h, w = img.shape[:2]
    cam_key = f'camera_{camera_num+1}'
    camera_matrix = np.array(camera_params[cam_key]['camera_matrix'])
    dist_coeff = np.array(camera_params[cam_key]['dist_coeff'])

    newcameramtx, _ = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w, h), 1, (w, h))
    img = cv.undistort(img, camera_matrix, dist_coeff, None, newcameramtx)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img

def combine_frames(frames, dots, rows=2, cols=2, border=1, size=(320, 240)):
    processed_frames = [process_frame(frame, dot, i, size, border) for i, (frame, dot) in enumerate(zip(frames, dots))]
    return stitch_frames(processed_frames, rows, cols, border, size)

def process_frame(frame, dot, index, size, border):
    resized = cv.resize(frame, size)
    if dot[0] is not None and dot[1] is not None:
        cv.circle(resized, (dot[0], dot[1]), 5, (0, 0, 255), -1)
    bordered = cv.copyMakeBorder(resized, border, border, border, border, cv.BORDER_CONSTANT, value=[255, 255, 255])
    cv.putText(bordered, f'Cam {index+1}', (border, border*2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return bordered

def stitch_frames(processed_frames, rows, cols, border, size):
    combined = np.zeros((rows * (size[1] + 2 * border), cols * (size[0] + 2 * border), 3), dtype=np.uint8)
    for idx, frame in enumerate(processed_frames):
        row, col = divmod(idx, cols)
        start_y, start_x = row * (size[1] + 2 * border), col * (size[0] + 2 * border)
        combined[start_y:start_y + frame.shape[0], start_x:start_x + frame.shape[1]] = frame
    return combined

def find_dot(img):
    img = cv.GaussianBlur(img,(5,5),0)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grey = cv.threshold(grey, 255*0.4, 255, cv.THRESH_BINARY)[1]
    contours, _ = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, (0,255,0), 1)

    center_x, center_y = None, None
    if contours:
        c = max(contours, key=cv.contourArea)
        moments = cv.moments(c)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv.putText(img, f'({center_x}, {center_y})', (center_x, center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (100,255,100), 1)
            cv.circle(img, (center_x, center_y), 1, (100,255,100), -1)

    return img, [center_x, center_y]


def main():
    camera_params = load_camera_params()
    cameras = Camera(ids=[0, 1, 2, 3], fps=150, resolution=Camera.RES_SMALL, gain=10, exposure=100)
    print("Controls: 'q' - Quit, Up/Down - Adjust Exposure, Left/Right - Adjust Gain")

    while True:
        key = cv.waitKey(1)
        imgs, _ = cameras.read()

        processed_imgs_and_dots = [(find_dot(init_frame(img, i, camera_params)), [None, None]) if img is None else find_dot(init_frame(img, i, camera_params)) for i, img in enumerate(imgs)]
        processed_imgs, dots = zip(*processed_imgs_and_dots)
        combined_frame = combine_frames(processed_imgs, dots)

        cv.imshow("Camera Grid", combined_frame)

        if key == 82 or key == 2490368:  # up arrow
            cameras.exposure = [min(255, exp + 5) for exp in cameras.exposure]
            print(f"Exposure increased: {cameras.exposure}")
        elif key == 84 or key == 2621440:  # down arrow
            cameras.exposure = [max(0, exp - 5) for exp in cameras.exposure]
            print(f"Exposure decreased: {cameras.exposure}")
        elif key == 83 or key == 2555904:  # right arrow
            cameras.gain = [min(63, g + 1) for g in cameras.gain]
            print(f"Gain increased: {cameras.gain}")
        elif key == 81 or key == 2424832:  # left arrow 
            cameras.gain = [max(0, g - 1) for g in cameras.gain]
            print(f"Gain decreased: {cameras.gain}")
        elif key == ord("q"):
            break

    cameras.end()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()