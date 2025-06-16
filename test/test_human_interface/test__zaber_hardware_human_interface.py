from zaber_motion.units import Units
from src.eye_tracking.human_interface.zabar_human_interface import ZaberHumanInterface
import time


def test_zaber_interface():
    try:
        print("Initializing Zaber interface...")
        interface = ZaberHumanInterface(port="COM6") #change to real port name


        interface.set_speed(1)
        # Move using absolute positions
        print("Moving to absolute positions (in µm)...")
        interface.leftright_abs(0)  # 10 mm right
        interface.updown_abs(10000)  # 5 mm up
        interface.forwardbackwards_abs(2000)  # 2 mm forward

        time.sleep(1)

        # Move using relative position
        print("Moving relatively (in µm)...")
        #interface.leftright_rel(10000)  # +1 mm
        #interface.updown_rel(5000)  # -0.5 mm
        #interface.forwardbackward_rel(-1500)  # +1.5 mm

        interface.get_position()

        interface.get_position_class()

        print("Test complete")

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    test_zaber_interface()
