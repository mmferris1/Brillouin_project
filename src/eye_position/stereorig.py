import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_calibration(path="stereo_calibration.npz"):
    data = np.load(path)
    K1, D1 = data["K1"], data["D1"]
    K2, D2 = data["K2"], data["D2"]
    R, T = data["R"], data["T"]
    return K1, D1, K2, D2, R, T


def plot_stereo_rig(K1, K2, R, T):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin (left camera)
    cam_left = np.array([0, 0, 0])
    ax.scatter(*cam_left, color='blue', label='Left Camera (origin)')

    # Right camera position in left camera coordinate system
    cam_right = T.flatten()
    ax.scatter(*cam_right, color='red', label='Right Camera')

    # Draw baseline
    ax.plot([cam_left[0], cam_right[0]],
            [cam_left[1], cam_right[1]],
            [cam_left[2], cam_right[2]], 'k--', label='Baseline')

    # Draw axes for left camera
    ax.quiver(0, 0, 0, 1, 0, 0, length=0.02, color='r', label='X axis')
    ax.quiver(0, 0, 0, 0, 1, 0, length=0.02, color='g', label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, 1, length=0.02, color='b', label='Z axis')

    ax.text(0, 0, 0.002, "Left Cam", color='blue')
    ax.text(*cam_right + 0.002, "Right Cam", color='red')

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title("Stereo Camera Geometry")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])

    # Set equal aspect and limits
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.show()


def main():
    K1, D1, K2, D2, R, T = load_calibration("stereo_calibration.npz")
    plot_stereo_rig(K1, K2, R, T)


if __name__ == "__main__":
    main()
