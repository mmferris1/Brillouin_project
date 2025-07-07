import os
import cv2
import numpy as np
from eye_tracking.devices.allied_vision_camera import AlliedVisionCamera

SAVE_DIR = r"C:\Users\Mandelstam\Documents\Connor\data\2025-6-5\take7"
NUM_FRAMES = 5

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

            # Create a dummy window for key input
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

            # Show both side by side
            combined = np.hstack((left_disp, right_disp))
            cv2.imshow(f"Frame {i} Preview - Left | Right", combined)
            print("[DEBUG] Press any key to save this frame.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Save images
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
