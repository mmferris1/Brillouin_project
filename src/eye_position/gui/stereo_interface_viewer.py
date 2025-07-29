import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QDoubleValidator, QPixmap

from src.eye_position.gui.stereo_interface_signaller import StereoInterfaceSignaller
from src.eye_position.gui.stereo_interface_manager import StereoInterfaceManager


class StereoInterfaceViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stereo Vision Pupil gui")

        self.manager = StereoInterfaceManager()
        self.signaller = StereoInterfaceSignaller(self.manager)

        self.init_ui()
        self.connect_signals()

        self.timer = QTimer()
        self.timer.timeout.connect(self.signaller.update_images)
        self.timer.start(200)  # ~5 FPS

    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        # === LEFT COLUMN: IMAGES ===
        image_col = QVBoxLayout()
        self.left_img = QLabel("Left Image")
        self.right_img = QLabel("Right Image")

        for label in [self.left_img, self.right_img]:
            label.setFixedSize(500, 500)
            label.setStyleSheet("background-color: black")

        image_col.addWidget(self.left_img)
        image_col.addWidget(self.right_img)

        self.coord_label = QLabel("3D Pupil Position: (X, Y, Z)")
        image_col.addWidget(self.coord_label)

        layout.addLayout(image_col)

        # === RIGHT COLUMN: CONTROLS ===
        control_col = QVBoxLayout()
        control_col.addWidget(self.create_motor_group())
        control_col.addWidget(self.create_laser_group())
        control_col.addWidget(self.create_camera_group())
        control_col.addStretch()

        layout.addLayout(control_col)

    def create_motor_group(self):
        group = QGroupBox("Motor Control")
        layout = QFormLayout()

        self.axis_inputs = {}
        for axis in ["x", "y", "z"]:
            input_field = QLineEdit("0.0")
            input_field.setValidator(QDoubleValidator(-10000, 10000, 3))
            self.axis_inputs[axis] = input_field

            move_btn = QPushButton(f"Move {axis.upper()}")
            move_btn.clicked.connect(lambda _, a=axis: self.signaller.move_motor_relative(a, float(self.axis_inputs[a].text())))

            abs_btn = QPushButton(f"Goto {axis.upper()}")
            abs_btn.clicked.connect(lambda _, a=axis: self.signaller.move_motor_absolute(a, float(self.axis_inputs[a].text())))

            row = QHBoxLayout()
            row.addWidget(move_btn)
            row.addWidget(abs_btn)
            layout.addRow(f"{axis.upper()} (µm):", input_field)
            layout.addRow(row)

        group.setLayout(layout)
        return group

    def create_laser_group(self):
        group = QGroupBox("Laser Distance")
        layout = QHBoxLayout()

        self.laser_input = QLineEdit("15000")
        self.laser_input.setValidator(QDoubleValidator(0, 30000, 1))

        set_btn = QPushButton("Set Laser (µm)")
        set_btn.clicked.connect(lambda: self.signaller.set_laser_distance(float(self.laser_input.text())))

        layout.addWidget(self.laser_input)
        layout.addWidget(set_btn)
        group.setLayout(layout)
        return group

    def create_camera_group(self):
        group = QGroupBox("Camera Settings")
        layout = QFormLayout()

        self.exposure_input = QLineEdit("10000")
        self.exposure_input.setValidator(QDoubleValidator(1, 100000, 0))
        exposure_btn = QPushButton("Set Exposure")
        exposure_btn.clicked.connect(lambda: self.signaller.set_exposure(float(self.exposure_input.text())))
        layout.addRow("Exposure (µs):", self.exposure_input)
        layout.addRow(exposure_btn)

        self.gain_input = QLineEdit("10.0")
        self.gain_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        gain_btn = QPushButton("Set Gain")
        gain_btn.clicked.connect(lambda: self.signaller.set_gain(float(self.gain_input.text())))
        layout.addRow("Gain:", self.gain_input)
        layout.addRow(gain_btn)

        roi_layout = QHBoxLayout()
        self.roi_x = QLineEdit("0")
        self.roi_y = QLineEdit("0")
        self.roi_w = QLineEdit("640")
        self.roi_h = QLineEdit("480")
        for field in [self.roi_x, self.roi_y, self.roi_w, self.roi_h]:
            field.setValidator(QDoubleValidator(0, 2048, 0))
            roi_layout.addWidget(field)

        roi_btn = QPushButton("Set ROI")
        roi_btn.clicked.connect(lambda: self.signaller.set_roi(
            int(self.roi_x.text()), int(self.roi_y.text()),
            int(self.roi_w.text()), int(self.roi_h.text())
        ))
        layout.addRow("ROI (X Y W H):", roi_layout)
        layout.addRow(roi_btn)

        group.setLayout(layout)
        return group

    def connect_signals(self):
        self.signaller.left_image_ready.connect(self.update_left_image)
        self.signaller.right_image_ready.connect(self.update_right_image)
        self.signaller.pupil3d_ready.connect(self.update_3d_position)
        self.signaller.motor_position_updated.connect(self.log_motor_position)

    def update_left_image(self, pixmap: QPixmap):
        self.left_img.setPixmap(pixmap.scaled(self.left_img.size(), Qt.KeepAspectRatio))

    def update_right_image(self, pixmap: QPixmap):
        self.right_img.setPixmap(pixmap.scaled(self.right_img.size(), Qt.KeepAspectRatio))

    def update_3d_position(self, point: tuple):
        if not any(np.isnan(point)):
            x, y, z = point
            self.coord_label.setText(f"3D Pupil Position: ({x:.2f}, {y:.2f}, {z:.2f}) mm")
        else:
            self.coord_label.setText("3D Pupil Position: (NaN, NaN, NaN)")

    def log_motor_position(self, axis: str, pos: float):
        print(f"{axis.upper()} motor updated: {pos:.3f} µm")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = StereoInterfaceViewer()
    viewer.show()
    sys.exit(app.exec_())
