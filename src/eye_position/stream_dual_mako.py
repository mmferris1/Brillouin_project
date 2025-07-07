import threading
import cv2
import numpy as np
import time
from vimba import Vimba
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

def preprocess(frame):
    img = frame if frame.ndim == 2 else frame[..., 0]
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_camera_ids():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        return [cam.get_id() for cam in cams]

def start_camera_stream(camera_id, window_name):
    cam = AlliedVisionCamera(camera_id)
    cam.set_exposure(5000)
    cam.set_gain(10)

    def frame_callback(frame):
        img = frame if frame.ndim == 2 else frame[..., 0]
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(window_name, norm)
        cv2.waitKey(1)

    cam.start_stream(frame_callback)
    return cam

def main():
    camera_ids = get_camera_ids()
    if len(camera_ids) < 2:
        print("[ERROR] Less than 2 cameras found.")
        return

    print(f"[INFO] Found cameras: {camera_ids}")

    # Launch both cameras in parallel threads
    cam1 = AlliedVisionCamera(0)
    cam1.set_exposure(5000)
    cam1.set_gain(10)
    cam1.start_stream(lambda frame: cv2.imshow("Camera 1", preprocess(frame)))

    cam2 = AlliedVisionCamera(1)
    cam2.set_exposure(5000)
    cam2.set_gain(10)
    cam2.start_stream(lambda frame: cv2.imshow("Camera 2", preprocess(frame)))

    try:
        print("[INFO] Streaming")
        while True:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        cam1.stop_stream()
        cam2.stop_stream()
        cam1.close()
        cam2.close()
        cv2.destroyAllWindows()
        print("[INFO] Cameras closed.")

if __name__ == "__main__":
    main()
