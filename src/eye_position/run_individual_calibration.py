from stereocalibrator import StereoCalibrator  # Replace 'calib' with the actual filename if different

def main():
    # Instantiate the calibrator with known pattern and square size
    calibrator = StereoCalibrator(
        left_dir="left_images",  # Not used here, but required by constructor
        right_dir="right_images",
        pattern_size=(6, 8),
        square_size=0.0025
    )

    # Calibrate the left camera
    print("\n[LEFT CAMERA] Starting intrinsic calibration...")
    K1, D1, size1 = calibrator.calibrate_single_camera(
        image_dir="C:/Users/Mandelstam/Documents/Connor/data/2025-6-5/take8/left",
        #r"C:\Users\Mandelstam\Documents\Connor\data\2025-6-5\left_camera"
        save_path="left_intrinsics.txt"
    )

    # Calibrate the right camera
    print("\n[RIGHT CAMERA] Starting intrinsic calibration...")
    K2, D2, size2 = calibrator.calibrate_single_camera(
        image_dir="C:/Users/Mandelstam/Documents/Connor/data/2025-6-5/take8/right",
        #r"C:\Users\Mandelstam\Documents\Connor\data\2025-6-5\right_camera"
        save_path="right_intrinsics.txt"
    )

    # Optional: Save to binary format as well
    if K1 is not None and K2 is not None:
        import numpy as np
        np.savez("individual_intrinsics.npz",
                 K1=K1, D1=D1, size1=size1,
                 K2=K2, D2=D2, size2=size2)
        print("\n[INFO] Intrinsics saved to 'individual_intrinsics.npz'.")

if __name__ == "__main__":
    main()
