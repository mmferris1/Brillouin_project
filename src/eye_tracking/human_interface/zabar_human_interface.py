from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from src.eye_tracking.human_interface.base_zaber_human_interface import BaseZaberHumanInterface

class ZaberHumanInterface(BaseZaberHumanInterface):
    def __init__(self, port="COM5", axis_index=1):
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(port)
        devices = self.connection.detect_devices()
        if not devices:
            raise RuntimeError("No Zaber devices found.")
        self.x_axis = devices[0].get_axis(axis_index)
        self.y_axis = devices[1].get_axis(axis_index)
        self.z_axis = devices[2].get_axis(axis_index)

        self.axis_map = {
            'x': self.x_axis,
            'y': self.y_axis,
            'z': self.z_axis,
        }
        #self.home() - not sure i need

    def updown_abs(self, position_um: float):
        self.y_axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.y_axis.wait_until_idle()

    def leftright_abs(self, position_um: float):
        self.x_axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.x_axis.wait_until_idle()

    def forwardbackwards_abs(self, position_um: float):
        self.z_axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.z_axis.wait_until_idle()

    def updown_rel(self, delta_um):
        self.y_axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.y_axis.wait_until_idle()

    def leftright_rel(self, delta_um):
        self.x_axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.x_axis.wait_until_idle()

    def forwardbackward_rel(self, delta_um):
        self.z_axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.z_axis.wait_until_idle()

    def change_units(self,  which_axis: str, unit):
        """Args: which_axis: 'x', 'y', 'z' """
        self.axis_map[which_axis].set_unit(unit)
        #this won't work as it - don't think we really need

    def set_speed(self, speed_mm_per_s: float):
        self.speed_mm_per_s = speed_mm_per_s
        print(f"[ZaberDummy] speed set to {speed_mm_per_s:.2f} mm/s")

    def set_acceleration(self, accel_native_units: int):
        self.accel_native_units = accel_native_units
        print(f"[ZaberDummy] acceleration set to {accel_native_units}")

    def get_position(self):
        pos_x = self.x_axis.get_position(Units.LENGTH_MICROMETRES)
        pos_y = self.y_axis.get_position(Units.LENGTH_MICROMETRES)
        pos_z = self.z_axis.get_position(Units.LENGTH_MICROMETRES)

        pos = {
            'x': pos_x,
            'y': pos_y,
            'z': pos_z
        }
        print(f"[Zaber] Current position (µm): X={pos_x:.2f}, Y={pos_y:.2f}, Z={pos_z:.2f} µm")
        return pos