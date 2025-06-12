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

    def up_abs(self, position_um: float):
        self.y_axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.y_axis.wait_until_idle()

    def right_abs(self, position_um: float):
        self.x_axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.x_axis.wait_until_idle()

    def forward_abs(self, position_um: float):
        self.z_axis.move_absolute(position_um, Units.LENGTH_MICROMETRES)
        self.z_axis.wait_until_idle()

    def up_rel(self, delta_um):
        self.y_axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.y_axis.wait_until_idle()

    def right_rel(self, delta_um):
        self.x_axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.x_axis.wait_until_idle()

    def forward_rel(self, delta_um):
        self.z_axis.move_relative(delta_um, Units.LENGTH_MICROMETRES)
        self.z_axis.wait_until_idle()

    def change_units(self,  which_axis: str, unit):
        """
        Args:
            which_axis: 'x', 'y', 'z'
        """
        self.axis_map[which_axis].set_unit(unit)
