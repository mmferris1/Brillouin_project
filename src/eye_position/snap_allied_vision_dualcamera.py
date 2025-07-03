import cv2
import numpy as np
import time
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

def initialize_camera(serial):
    cam = AlliedVisionCamera(serial=serial)
    cam.set_exposure(5000)
    cam.set_gain(10)
    cam.set_trigger_mode(mode="On", source="Line1")  # Set to hardware trigger
    return cam

def main():
    # Replace with your actual camera serials
    serial_left = "12345678"
    serial_right = "87654321"

    print("[DEBUG] Initializing cameras...")
    cam_left = initialize_camera(serial_left)
    cam_right = initialize_camera(serial_right)

    try:
        print("[DEBUG] Waiting for trigger pulse...")

        # Cameras will wait until external trigger is received
        image_left = cam_left.snap()
        image_right = cam_right.snap()

        print("[DEBUG] Captured both frames.")

        for i, (img, name) in enumerate(zip([image_left, image_right], ["Left", "Right"])):
            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow(f"{name} Camera", norm)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        print("[DEBUG] Closing cameras...")
        cam_left.close()
        cam_right.close()

if __name__ == "__main__":
    main()
