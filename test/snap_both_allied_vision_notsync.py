import os
import cv2
import numpy as np
from eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

SAVE_DIR = r"C:\Users\Mandelstam\Documents\Connor\data\2025-6-5\take8"
NUM_FRAMES = 10


DISPLAY_SCALE = 0.4  # Resize factor for display

def resize_image(img, scale=0.5):
    return cv2.resize(img, (0, 0), fx=scale, fy=scale)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("[DEBUG] Initializing Allied Vision cameras...")

    left_cam = AlliedVisionCamera(index=0)
    right_cam = AlliedVisionCamera(index=1)

    try:
        for cam, label in [(left_cam, "left"), (right_cam, "right")]:
            cam.set_exposure(5000)
            cam.set_gain(10)
            print(f"[DEBUG] {label} camera ready.")

        for i in range(1, NUM_FRAMES + 1):
            print(f"\n[DEBUG] Waiting for key press to capture frame {i}...")
            print("Press any key in the image window to capture the next frame.")

            # Dummy window for input
            cv2.imshow("Press any key to snap", np.zeros((100, 400), dtype=np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            left_img = left_cam.snap()
            right_img = right_cam.snap()

            if left_img.ndim == 3 and left_img.shape[-1] == 1:
                left_img = left_img[..., 0]
            if right_img.ndim == 3 and right_img.shape[-1] == 1:
                right_img = right_img[..., 0]

            # Normalize for display
            left_disp = cv2.normalize(left_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            right_disp = cv2.normalize(right_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Resize for screen
            left_disp_small = resize_image(left_disp, DISPLAY_SCALE)
            right_disp_small = resize_image(right_disp, DISPLAY_SCALE)

            # Add text labels
            cv2.putText(left_disp_small, "Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(right_disp_small, "Right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            # Show both side-by-side
            combined = np.hstack((left_disp_small, right_disp_small))
            cv2.imshow(f"Frame {i} Preview - Left | Right", combined)
            print("[DEBUG] Press any key to save this frame.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save full-resolution originals
            left_path = os.path.join(SAVE_DIR, f"left_{i}.png")
            right_path = os.path.join(SAVE_DIR, f"right_{i}.png")
            cv2.imwrite(left_path, left_img)
            cv2.imwrite(right_path, right_img)
            print(f"[DEBUG] Saved: {left_path}, {right_path}")

    finally:
        print("[DEBUG] Closing cameras...")
        left_cam.close()
        right_cam.close()

if __name__ == "__main__":
    main()
