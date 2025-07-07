import threading
import cv2
import numpy as np
import time
from vimba import Vimba
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

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
    cam1 = start_camera_stream(camera_ids[0], "Camera 1")
    cam2 = start_camera_stream(camera_ids[1], "Camera 2")

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
