from abc import ABC, abstractmethod
import numpy as np

class BaseZaberHumanInterface(ABC):

    @abstractmethod

    def get_name(self) -> str:
        """ Get the motor name or type """
        pass

    def up_rel(self):
        """ Move (?) motor up """
        pass

    def down_rel(self):
        """ Move (?) motor down """
        pass

    def left_rel(self):
        """ Move (?) motor to the left """
        pass

    def right_rel(self):
        """ Move (?) motor to the right """
        pass

    def forward_rel(self):
        """ Move (?) motor forward """
        pass

    def back_rel(self):
        """ Move (?) motor backwards """
        pass
    def up_abs(self):
        """ Move (?) motor up """
        pass

    def down_abs(self):
        """ Move (?) motor down """
        pass

    def left_abs(self):
        """ Move (?) motor to the left """
        pass

    def right_abs(self):
        """ Move (?) motor to the right """
        pass

    def forward_abs(self):
        """ Move (?) motor forward """
        pass

    def back_abs(self):
        """ Move (?) motor backwards """
        pass

    def change_units(self):
        """ change units for motor movement"""
        pass

    def get_position(self):
        """ get x,y,z coordinates of a position"""
        pass