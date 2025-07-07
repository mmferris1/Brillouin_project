import cv2
import numpy as np
from src.eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

def get_trigger_output(cam):
    """trigger and be triggered"""
    try:
        line_selector = cam.camera.get_feature_by_name("LineSelector")
        line_selector.set("Line3")  # Use Line3 for output
        cam.camera.get_feature_by_name("LineMode").set("Output")
        cam.camera.get_feature_by_name("LineSource").set("ExposureActive")
        print("[triggerer] Line3 configured to output trigger during exposure.")
    except Exception as e:
        print(f"[triggerer] Failed to configure trigger output: {e}")

def main():
    print("[DEBUG] Initializing both cameras (index-based)...")
    cam_left = AlliedVisionCamera(index=0)   # triggerer
    cam_right = AlliedVisionCamera(index=1)  # triggered

    # exposure and stuff set up
    cam_left.set_exposure(5000)
    cam_right.set_exposure(5000)
    cam_left.set_gain(10)
    cam_right.set_gain(10)

    # right waits to be triggered
    cam_right.set_trigger_mode("On", "Line1")

    # trigger capture, on Line3?
    cam_left.set_trigger_mode("Off")  # Capture normally
    cam_left.set_acquisition_mode("SingleFrame")  # Or "Continuous" for live stream
    get_trigger_output(cam_left)

    try:
        print("[DEBUG] Triggering master (left) camera via software...")
        cam_left.camera.get_feature_by_name("TriggerSoftware").run()

        print("[DEBUG] Reading images...")
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
