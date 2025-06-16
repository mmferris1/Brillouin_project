from abc import ABC, abstractmethod
import numpy as np

class BaseZaberHumanInterface(ABC):

    @abstractmethod
    def get_name(self) -> str:
        """ Get the motor name or type """
        pass

    @abstractmethod
    def updown_rel(self, delta_um):
        """ Move (?) motor up and down """
        pass

    @abstractmethod
    def leftright_rel(self, delta_um):
        """ Move (?) motor to the left and right"""
        pass

    @abstractmethod
    def forwardbackward_rel(self, delta_um):
        """ Move (?) motor forward and backward """
        pass

    @abstractmethod
    def updown_abs(self, position_um: float):
        """ Move (?) motor up """
        pass

    @abstractmethod
    def leftright_abs(self, position_um: float):
        """ Move (?) motor to the left """
        pass

    @abstractmethod
    def forwardbackwards_abs(self, position_um: float):
        """ Move (?) motor forward and backwards"""
        pass

    #def change_units(self):
        #""" change units for motor movement"""
        #pass

    @abstractmethod
    def get_position(self):
        """ get x,y,z coordinates of a position"""
        pass

    @abstractmethod
    def set_speed(self):
        pass

    @abstractmethod
    def set_acceleration(self):
        pass