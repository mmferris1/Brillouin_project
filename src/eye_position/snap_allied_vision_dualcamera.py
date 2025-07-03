import cv2
import numpy as np
import time
from vimba import Vimba
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera


def create_allied_camera_from_index(index):
    with Vimba.get_instance() as vimba:
        cameras = vimba.get_all_cameras()
        if len(cameras) < index + 1:
            raise RuntimeError(f"Camera index {index} not found. Only {len(cameras)} camera(s) connected.")

        # Get the ID of the specific camera and initialize one AlliedVisionCamera
        cam_id = cameras[index].get_id()

    # Create new AlliedVisionCamera that always picks the first camera,
    # but we re-order the list so each instance wraps a different one.
    AlliedVisionCamera._CAMERA_ID_OVERRIDE = cam_id
    return AlliedVisionCamera()


# Patch AlliedVisionCamera to support camera selection (without changing original class)
def override_camera_selection():
    orig_init = AlliedVisionCamera.__init__

    def patched_init(self):
        vimba = Vimba.get_instance()
        vimba.__enter__()
        cameras = vimba.get_all_cameras()

        if not cameras:
            raise RuntimeError("No Allied Vision cameras found.")

        if hasattr(AlliedVisionCamera, "_CAMERA_ID_OVERRIDE"):
            chosen = next((c for c in cameras if c.get_id() == AlliedVisionCamera._CAMERA_ID_OVERRIDE), None)
        else:
            chosen = cameras[0]

        if chosen is None:
            raise RuntimeError("Requested camera ID not found.")

        self.vimba = vimba
        self.camera = chosen
        self.camera.__enter__()
        self.streaming = False
        print(f"[AVCamera] ...Connected to camera: {self.camera.get_id()}")

    AlliedVisionCamera.__init__ = patched_init


# Apply override
override_camera_selection()


def main():
    print("[DEBUG] Initializing both cameras (index-based)...")
    cam_left = create_allied_camera_from_index(0)
    cam_right = create_allied_camera_from_index(1)

    cam_left.set_exposure(5000)
    cam_right.set_exposure(5000)
    cam_left.set_gain(10)
    cam_right.set_gain(10)

    # Trigger mode: external trigger on Line1
    cam_left.set_trigger_mode("On", "Line1")
    cam_right.set_trigger_mode("On", "Line1")

    try:
        print("[DEBUG] Waiting for hardware trigger pulse...")

        image_left = cam_left.snap()
        image_right = cam_right.snap()

        print("[DEBUG] Captured both frames.")

        for img, name in zip([image_left, image_right], ["Left", "Right"]):
            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow(f"{name} Camera", norm)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    finally:
        cam_left.close()
        cam_right.close()
        print("[DEBUG] Cameras closed.")


if __name__ == "__main__":
    main()
