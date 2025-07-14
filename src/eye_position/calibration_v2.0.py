import numpy as np
from stereocalibrator import StereoCalibrator
import os

def main():
    # Initialize calibrator
    calibrator = StereoCalibrator(
        left_dir="/Users/margaretferris/Desktop/take8/left",
        right_dir="/Users/margaretferris/Desktop/take8/right",
        pattern_size=(8, 6),
        square_size=0.0025
    )

    # Run stereo calibration
    retval = calibrator.calibrate()

    if retval:
        # Compute rectification after calibration
        Q = calibrator.compute_rectification()

        # Save all values
        np.savez("stereo_calibration2.npz",
                 K1=calibrator.cameraMatrix1, D1=calibrator.distCoeffs1,
                 K2=calibrator.cameraMatrix2, D2=calibrator.distCoeffs2,
                 R=calibrator.R, T=calibrator.T,
                 E=calibrator.E, F=calibrator.F,
                 R1=calibrator.R1, R2=calibrator.R2,
                 P1=calibrator.P1, P2=calibrator.P2,
                 Q=Q)

        print("[INFO] Calibration results saved to stereo_calibration2.npz")


    else:
        print("[ERROR] Calibration failed.")

if __name__ == "__main__":
    main()
