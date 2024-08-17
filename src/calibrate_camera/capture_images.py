import os
import cv2 as cv
import numpy as np

import sys
sys.path.append('/home/george/Projects/drone-swarm-motion-capture')
from src.pseyepy import Camera


def adjust_parameter(param, increment, min_val=None, max_val=None):
    new_val = param + increment
    if min_val is not None:
        new_val = max(min_val, new_val)
    if max_val is not None:
        new_val = min(max_val, new_val)
    return new_val

camera = Camera(fps=150, resolution=Camera.RES_SMALL, gain=10, exposure=100)

print("Controls: Space - Capture, 'q' - Quit, Up/Down - Adjust Exposure, Left/Right - Adjust Gain")

i = 0
path = 'src/calibrate_camera/images/cam4'
while True:
    key = cv.waitKey(1) & 0xFF

    frame, _ = camera.read()
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    frame = np.rot90(frame)
    frame = np.fliplr(frame)
    cv.imshow("PS3 Eye", frame)
    if key == ord(' '):
        filepath = os.path.join(path, f"{i}.jpg")
        cv.imwrite(filepath, frame)
        print(f'Image saved to: {filepath}')
        i += 1
    elif key == 82 or key == 2490368:  # up arrow
        # increase exposure
        camera.exposure[0] = adjust_parameter(camera.exposure[0], 5, max_val=255)
        print("Camera exposure: " + str(camera.exposure[0]))
    elif key == 84 or key == 2621440:  # down arrow
        # decrease exposure
        camera.exposure[0] = adjust_parameter(camera.exposure[0], -5, min_val=0)
        print("Camera exposure: " + str(camera.exposure[0]))
    elif key == 81 or key == 2424832:  # left arrow
        # decrease gain
        camera.gain[0] = adjust_parameter(camera.gain[0], -1, min_val=0)
        print("Camera gain: " + str(camera.gain[0]))
    elif key == 83 or key == 2555904:  # right arrow
        # not working
        camera.gain[0] = adjust_parameter(camera.gain[0], 1, max_val=63)
        print("Camera gain: " + str(camera.gain[0]))
    elif key == ord('q'):
        break

camera.end()