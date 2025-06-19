class ZaberHumanInterfaceDummy:
    def __init__(self, port="COM5", axis_index=1):
        self.port = port
        self.axis_index = axis_index
        self._positions = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        self.speed_mm_per_s = 10.0
        self.accel_native_units = 600
        self.homed = False
        print(f"[ZaberDummy] Initialized on port {port}, axis {axis_index}")

        #self.home()


    def updown_abs(self, position_um: float):
        self._positions['y'] = position_um
        print(f"[ZaberDummy] Moving up/down absolute → {position_um:.2f} µm")

    def leftright_abs(self, position_um: float):
        self._positions['x'] = position_um
        print(f"[ZaberDummy] Moving left/right absolute → {position_um:.2f} µm")

    def forwardbackwards_abs(self, position_um: float):
        self._positions['z'] = position_um
        print(f"[ZaberDummy] Moving forward/backwars absolute → {position_um:.2f} µm")

    def updown_rel(self, delta_um):
        self._positions['y'] += delta_um
        print(f"[ZaberDummy] Moving up/down relative → {delta_um:+.2f} µm")

    def leftright_rel(self, delta_um):
        self._positions['x'] += delta_um
        print(f"[ZaberDummy] Moving left/right relative → {delta_um:+.2f} µm")

    def forwardbackwards_rel(self, delta_um):
        self._positions['z'] += delta_um
        print(f"[ZaberDummy] Moving forward/backward relative → {delta_um:+.2f} µm")

    def get_position(self):
        pos = {
            'x': self._positions['x'],
            'y': self._positions['y'],
            'z': self._positions['z']
        }
        print(f"[ZaberDummy] Current position (µm): X={pos['x']:.2f}, Y={pos['y']:.2f}, Z={pos['z']:.2f}")
        return pos
