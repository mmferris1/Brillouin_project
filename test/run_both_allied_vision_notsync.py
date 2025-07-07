import cv2
import numpy as np
import threading
import time
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

# Shared containers for frames
latest_left_frame = None
latest_right_frame = None
frame_lock = threading.Lock()

def left_frame_handler(frame):
    global latest_left_frame
    with frame_lock:
        latest_left_frame = frame.copy()

def right_frame_handler(frame):
    global latest_right_frame
    with frame_lock:
        latest_right_frame = frame.copy()

def main():
    print("[DEBUG] Initializing left and right Allied Vision cameras...")
    left_cam = AlliedVisionCamera(index=0)
    right_cam = AlliedVisionCamera(index=1)

    # Set exposure/gain as desired
    left_cam.set_exposure(5000)
    left_cam.set_gain(10)
    right_cam.set_exposure(5000)
    right_cam.set_gain(10)

    # Start continuous streaming
    left_cam.start_stream(left_frame_handler)
    right_cam.start_stream(right_frame_handler)
    print("[DEBUG] Streaming started for both cameras. Press 'q' or ESC to quit.")

    while True:
        with frame_lock:
            if latest_left_frame is not None and latest_right_frame is not None:
                left_disp = cv2.normalize(latest_left_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                right_disp = cv2.normalize(latest_right_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                if left_disp.ndim == 3 and left_disp.shape[-1] == 1:
                    left_disp = left_disp[..., 0]
                if right_disp.ndim == 3 and right_disp.shape[-1] == 1:
                    right_disp = right_disp[..., 0]

                stacked = np.hstack((left_disp, right_disp))
                cv2.imshow("Live: Left | Right", stacked)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    print("[DEBUG] Shutting down...")
    left_cam.close()
    right_cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
