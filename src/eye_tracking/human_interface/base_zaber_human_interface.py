from abc import ABC, abstractmethod
import numpy as np

class BaseZaberHumanInterface(ABC):

    @abstractmethod

    def get_name(self) -> str:
        """ Get the motor name or type """
        pass

    def updown_rel(self):
        """ Move (?) motor up and down """
        pass

    def leftright_rel(self):
        """ Move (?) motor to the left and right"""
        pass

    def forwardbackward_rel(self):
        """ Move (?) motor forward and backward """
        pass

    def updown_abs(self):
        """ Move (?) motor up """
        pass

    def leftright_abs(self):
        """ Move (?) motor to the left """
        pass

    def forwardbackwards_abs(self):
        """ Move (?) motor forward and backwards"""
        pass

    def change_units(self):
        """ change units for motor movement"""
        pass

    def get_position(self):
        """ get x,y,z coordinates of a position"""
        pass

    def set_speed(self):
        pass

    def set_acceleration(self):
        pass