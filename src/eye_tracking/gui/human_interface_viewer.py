import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QLineEdit, QFileDialog
)
from PyQt5.QtGui import QPixmap, QDoubleValidator, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer

from src.eye_tracking.human_interface.human_interface_manager import HumanInterfaceManager


class HumanInterfaceSignaller(QObject):
    image_ready = pyqtSignal(QPixmap)
    pupil_laser_coords_ready = pyqtSignal(tuple, tuple)
    motor_position_updated = pyqtSignal(str, float)

    def __init__(self, manager: HumanInterfaceManager):
        super().__init__()
        self.manager = manager

    def update_image(self):
        pixmap = self.manager.get_overlay_image()
        if pixmap:
            self.image_ready.emit(pixmap)

    def fetch_coords(self):
        pupil = self.manager.get_pupil_coords()
        laser = self.manager.get_laser_coords()
        self.pupil_laser_coords_ready.emit(pupil, laser)

    def move_motor_relative(self, axis: str, delta: float):
        self.manager.move_motor_relative(axis, delta)
        position = self.manager.get_motor_position(axis)
        self.motor_position_updated.emit(axis, position)

    def move_motor_absolute(self, axis: str, position: float):
        self.manager.move_motor_absolute(axis, position)
        actual_position = self.manager.get_motor_position(axis)
        self.motor_position_updated.emit(axis, actual_position)


class HumanInterfaceViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Human Interface GUI")
        self.manager = HumanInterfaceManager()
        self.signaller = HumanInterfaceSignaller(self.manager)

        self.init_ui()
        self.connect_signals()

        # Start periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.signaller.update_image)
        self.timer.start(100)  # ~10 FPS

        self.signaller.fetch_coords()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Display Section
        display_row = QHBoxLayout()
        self.image_label = QLabel("Image here")
        self.image_label.setFixedSize(320, 240)
        self.image_label.setStyleSheet("background-color: black;")
        display_row.addWidget(self.image_label)
        display_row.addWidget(self.create_coords_group())
        layout.addLayout(display_row)

        # Control Section
        control_row = QHBoxLayout()
        control_row.addWidget(self.create_motor_group())
        layout.addLayout(control_row)

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
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

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
