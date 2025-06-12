from zaber_motion.units import Unit
from src.eye_tracking.human_interface.zaber_human_interface import ZaberHumanInterface
import time


def test_zaber_interface():
    try:
        print("Initializing Zaber interface...")
        interface = ZaberHumanInterface(port="COM5") #change to real port name

        print("Changing all axes to micrometre units...")
        interface.change_units('x', Unit.MICROMETRES) #this is probably unnessesary
        interface.change_units('y', Unit.MICROMETRES)
        interface.change_units('z', Unit.MICROMETRES)

        # Move using absolute positions
        print("Moving to absolute positions (in µm)...")
        interface.right_abs(10000)  # 10 mm right
        interface.up_abs(5000)  # 5 mm up
        interface.forward_abs(2000)  # 2 mm forward

        time.sleep(1)

        # Move using relative position
        print("Moving relatively (in µm)...")
        interface.right_rel(1000)  # +1 mm
        #interface.up_rel(-500)  # -0.5 mm
        #interface.forward_rel(1500)  # +1.5 mm

        print("Test complete")

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    test_zaber_interface()
