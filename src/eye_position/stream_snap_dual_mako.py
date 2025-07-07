import cv2
import numpy as np
import time
import os
from datetime import datetime
from vimba import Vimba
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

# chanege to path name
SAVE_DIR = r"C:\Users\Mandelstam\Documents\Connor\data\2025-6-5"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_camera_ids():
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        return [cam.get_id() for cam in cams]

def start_camera_stream(camera, window_name):
    def frame_callback(frame):
        img = frame if frame.ndim == 2 else frame[..., 0]
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(window_name, norm)
        cv2.waitKey(1)

    camera.start_stream(frame_callback)

def main():
    camera_ids = get_camera_ids()
    if len(camera_ids) < 2:
        print("[ERROR] Less than 2 cameras found.")
        return

    print(f"[INFO] Found cameras: {camera_ids}")

    cam_left = AlliedVisionCamera(camera_ids[0])
    cam_right = AlliedVisionCamera(camera_ids[1])

    cam_left.set_exposure(5000)
    cam_right.set_exposure(5000)
    cam_left.set_gain(10)
    cam_right.set_gain(10)

    cam_left.set_trigger_mode("Off")
    cam_right.set_trigger_mode("Off")

    start_camera_stream(cam_left, "Left Camera")
    start_camera_stream(cam_right, "Right Camera")

    try:
        print("[INFO] Streaming... press SPACE to capture, 'q' to quit.")
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # Spacebar
                print("[INFO] Capturing images...")
                img_left = cam_left.snap()
                img_right = cam_right.snap()

                if img_left.ndim == 3 and img_left.shape[-1] == 1:
                    img_left = img_left[..., 0]
                if img_right.ndim == 3 and img_right.shape[-1] == 1:
                    img_right = img_right[..., 0]

                timestamp = datetime.now().strftime("%H%M%S")
                left_path = os.path.join(SAVE_DIR, f"left_{timestamp}.bmp")
                right_path = os.path.join(SAVE_DIR, f"right_{timestamp}.bmp")

                cv2.imwrite(left_path, img_left)
                cv2.imwrite(right_path, img_right)

                print(f"[INFO] Saved:\n  {left_path}\n  {right_path}")

    finally:
        cam_left.stop_stream()
        cam_right.stop_stream()
        cam_left.close()
        cam_right.close()
        cv2.destroyAllWindows()
        print("[INFO] Cameras closed.")

if __name__ == "__main__":
    main()
