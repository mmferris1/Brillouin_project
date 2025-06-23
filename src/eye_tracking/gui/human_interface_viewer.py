import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QLineEdit, QFileDialog
)
from PyQt5.QtGui import QPixmap, QDoubleValidator, QImage
from PyQt5.QtCore import Qt, QTimer

from src.eye_tracking.human_interface.human_interface_manager import HumanInterfaceManager
from src.eye_tracking.gui.human_interface_signaler import HumanInterfaceSignaller

class HumanInterfaceViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Human Interface GUI")
        manager = HumanInterfaceManager()
        self.signaller = HumanInterfaceSignaller(manager)

        self.init_ui()
        self.connect_signals()

        # Start periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.signaller.update_image)
        self.timer.start(100)  # ~10 FPS

        self.signaller.fetch_coords()


    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left side: Image + Coordinates
        left_column = QVBoxLayout()
        self.image_label = QLabel("Image here")
        self.image_label.setFixedSize(640, 640)
        self.image_label.setStyleSheet("background-color: black;")
        left_column.addWidget(self.image_label)
        left_column.addWidget(self.create_coords_group())

        # Right side: Controls
        right_column = QVBoxLayout()
        right_column.addWidget(self.create_motor_group())
        right_column.addWidget(self.create_laser_distance_group())
        right_column.addWidget(self.create_camera_settings_group())
        right_column.addWidget(self.create_camera_controls())
        right_column.addStretch()

        # Combine Left and Right
        main_layout.addLayout(left_column)
        main_layout.addLayout(right_column)

    def create_camera_settings_group(self):
        group = QGroupBox("Camera Settings")
        layout = QFormLayout()

        # Exposure
        self.exposure_input = QLineEdit("10000")
        self.exposure_input.setValidator(QDoubleValidator(1.0, 1000000.0, 0))
        exposure_btn = QPushButton("Set Exposure (µs)")
        exposure_btn.clicked.connect(lambda: self.signaller.manager.set_exposure(float(self.exposure_input.text())))
        layout.addRow("Exposure (µs):", self.exposure_input)
        layout.addRow(exposure_btn)

        # Gain
        self.gain_input = QLineEdit("10.0")
        self.gain_input.setValidator(QDoubleValidator(0.0, 100.0, 2))
        gain_btn = QPushButton("Set Gain")
        gain_btn.clicked.connect(lambda: self.signaller.manager.set_gain(float(self.gain_input.text())))
        layout.addRow("Gain:", self.gain_input)
        layout.addRow(gain_btn)

        # Frame Rate
        self.fps_input = QLineEdit("10.0")
        self.fps_input.setValidator(QDoubleValidator(0.1, 1000.0, 1))
        fps_btn = QPushButton("Set Frame Rate (FPS)")
        fps_btn.clicked.connect(lambda: self.signaller.manager.set_frame_rate(float(self.fps_input.text())))
        layout.addRow("Frame Rate (fps):", self.fps_input)
        layout.addRow(fps_btn)

        # ROI
        self.roi_x_input = QLineEdit("0")
        self.roi_y_input = QLineEdit("0")
        self.roi_w_input = QLineEdit("640")
        self.roi_h_input = QLineEdit("480")
        self.roi_x_input.setValidator(QDoubleValidator(0, 639, 0))
        self.roi_y_input.setValidator(QDoubleValidator(0, 479, 0))
        self.roi_w_input.setValidator(QDoubleValidator(1, 640, 0))
        self.roi_h_input.setValidator(QDoubleValidator(1, 480, 0))

        roi_btn = QPushButton("Set ROI")
        roi_btn.clicked.connect(self.set_roi_from_inputs)

        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("X:"))
        roi_layout.addWidget(self.roi_x_input)
        roi_layout.addWidget(QLabel("Y:"))
        roi_layout.addWidget(self.roi_y_input)
        roi_layout.addWidget(QLabel("W:"))
        roi_layout.addWidget(self.roi_w_input)
        roi_layout.addWidget(QLabel("H:"))
        roi_layout.addWidget(self.roi_h_input)

        layout.addRow("ROI:", roi_layout)
        layout.addRow(roi_btn)

        group.setLayout(layout)
        return group

    def set_roi_from_inputs(self):
        try:
            x = int(self.roi_x_input.text())
            y = int(self.roi_y_input.text())
            w = int(self.roi_w_input.text())
            h = int(self.roi_h_input.text())
            self.signaller.manager.set_roi(x, y, w, h)
        except ValueError:
            print("Invalid ROI input")

    def create_laser_distance_group(self):
        self.laser_input = QLineEdit("15000")
        self.laser_input.setValidator(QDoubleValidator(0.0, 30000.0, 1))

        self.laser_set_btn = QPushButton("Set Laser Distance (µm)")
        self.laser_set_btn.clicked.connect(self.set_laser_distance)

        layout = QHBoxLayout()
        layout.addWidget(self.laser_input)
        layout.addWidget(self.laser_set_btn)

        group = QGroupBox("Laser Motor Position")
        group.setLayout(layout)
        return group

    def create_camera_controls(self):
        group = QGroupBox("Camera Controls")
        layout = QHBoxLayout()

        self.snap_btn = QPushButton("Snap Image")
        self.start_stream_btn = QPushButton("Start Stream")
        self.stop_stream_btn = QPushButton("Stop Stream")

        self.snap_btn.clicked.connect(self.signaller.snap_camera_image)
        self.start_stream_btn.clicked.connect(self.signaller.start_camera_stream)
        self.stop_stream_btn.clicked.connect(self.signaller.stop_camera_stream)

        layout.addWidget(self.snap_btn)
        layout.addWidget(self.start_stream_btn)
        layout.addWidget(self.stop_stream_btn)

        group.setLayout(layout)
        return group

    def set_laser_distance(self):
        try:
            distance = float(self.laser_input.text())
            self.signaller.manager.set_laser_distance(distance)
        except ValueError:
            print("Invalid laser distance input.")

    def connect_signals(self):
        self.signaller.image_ready.connect(self.update_image_display)
        self.signaller.pupil_laser_coords_ready.connect(self.update_coords)
        self.signaller.motor_position_updated.connect(self.update_motor_position)

    def create_coords_group(self):
        self.pupil_label = QLabel("Pupil: (x, y)")
        self.laser_label = QLabel("Laser: (x, y)")
        layout = QVBoxLayout()
        layout.addWidget(self.pupil_label)
        layout.addWidget(self.laser_label)
        group = QGroupBox("Coordinates")
        group.setLayout(layout)
        return group

    def create_motor_group(self):
        layout = QFormLayout()

        # X Axis Controls
        self.x_input = QLineEdit("0.0")
        self.x_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 3))
        x_buttons = QHBoxLayout()
        self.x_move_btn = QPushButton("Move X")
        self.x_abs_btn = QPushButton("Go to X")
        self.x_move_btn.clicked.connect(lambda: self.move_motor_relative("x", float(self.x_input.text())))
        self.x_abs_btn.clicked.connect(lambda: self.move_motor_absolute("x", float(self.x_input.text())))
        x_buttons.addWidget(self.x_move_btn)
        x_buttons.addWidget(self.x_abs_btn)
        layout.addRow("X (µm):", self.x_input)
        layout.addRow(x_buttons)

        # Y Axis Controls
        self.y_input = QLineEdit("0.0")
        self.y_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 3))
        y_buttons = QHBoxLayout()
        self.y_move_btn = QPushButton("Move Y")
        self.y_abs_btn = QPushButton("Go to Y")
        self.y_move_btn.clicked.connect(lambda: self.move_motor_relative("y", float(self.y_input.text())))
        self.y_abs_btn.clicked.connect(lambda: self.move_motor_absolute("y", float(self.y_input.text())))
        y_buttons.addWidget(self.y_move_btn)
        y_buttons.addWidget(self.y_abs_btn)
        layout.addRow("Y (µm):", self.y_input)
        layout.addRow(y_buttons)

        # Z Axis Controls
        self.z_input = QLineEdit("0.0")
        self.z_input.setValidator(QDoubleValidator(-1000.0, 1000.0, 3))
        z_buttons = QHBoxLayout()
        self.z_move_btn = QPushButton("Move Z")
        self.z_abs_btn = QPushButton("Go to Z")
        self.z_move_btn.clicked.connect(lambda: self.move_motor_relative("z", float(self.z_input.text())))
        self.z_abs_btn.clicked.connect(lambda: self.move_motor_absolute("z", float(self.z_input.text())))
        z_buttons.addWidget(self.z_move_btn)
        z_buttons.addWidget(self.z_abs_btn)
        layout.addRow("Z (µm):", self.z_input)
        layout.addRow(z_buttons)

        group = QGroupBox("Motor Control")
        group.setLayout(layout)
        return group

    def update_image_display(self, pixmap: QPixmap):
        try:
            if pixmap and not pixmap.isNull():
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
            else:
                print("[Warning] Received null or invalid pixmap.")
        except Exception as e:
            print(f"[ERROR] Failed to update image display: {e}")

    def update_coords(self, pupil: tuple, laser: tuple):
        self.pupil_label.setText(f"Pupil: {pupil}")
        self.laser_label.setText(f"Laser: {laser}")

    def update_motor_position(self, axis: str, pos: float):
        print(f"{axis.upper()} motor at {pos:.3f} µm")

    def move_motor_relative(self, axis: str, delta: float):
        self.signaller.move_motor_relative(axis, delta)

    def move_motor_absolute(self, axis: str, pos: float):
        self.signaller.move_motor_absolute(axis, pos)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = HumanInterfaceViewer()
    gui.show()
    sys.exit(app.exec_())
