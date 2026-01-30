"""
Lane Keeping Assist System (LKAS)
Module 02 for RC Car Autonomous Driving
"""
from .lkas import LaneKeepingAssist
from .tracking.lane_tracker import LaneTracker
from .detection.departure_detector import DepartureDetector, DepartureThresholds
from .control.pid_controller import PIDController, PIDParams
from .alert.warning_system import WarningSystem

__all__ = [
    'LaneKeepingAssist',
    'LaneTracker',
    'DepartureDetector',
    'DepartureThresholds',
    'PIDController',
    'PIDParams',
    'WarningSystem'
]

__version__ = '1.0.0'
__author__ = "SEA:ME Autonomous Driving Team"
