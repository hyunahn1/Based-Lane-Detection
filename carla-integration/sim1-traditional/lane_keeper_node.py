"""
Lane Keeper Node (Module 02 Integration)
"""
import sys
from pathlib import Path

# Add Module 02 to path
module02_path = Path(__file__).parent.parent.parent / '02-lane-keeping-assist'
sys.path.insert(0, str(module02_path))

from typing import Dict
import numpy as np


class LaneKeeperNode:
    """
    Module 02 wrapper for CARLA integration
    """
    def __init__(
        self,
        kp: float = 1.5,
        ki: float = 0.1,
        kd: float = 0.8,
        track_width: float = 1.5
    ):
        # PID Controller
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        
        # Departure detection
        self.track_width = track_width
        self.safe_distance = track_width / 2 * 0.7  # 70% of half width
        
        print("✅ Lane Keeper initialized")
        print(f"   PID: Kp={kp}, Ki={ki}, Kd={kd}")
    
    def compute_control(
        self,
        lateral_offset: float,
        heading_error: float,
        velocity: float,
        dt: float = 0.033
    ) -> Dict:
        """
        Compute steering control
        
        Args:
            lateral_offset: meters (+ = right, - = left)
            heading_error: radians
            velocity: m/s
            dt: time step (seconds)
        
        Returns:
            {
                'steering': float (degrees),
                'throttle': float (0-1),
                'risk_level': int (0-5),
                'warning': str,
                'should_intervene': bool
            }
        """
        # Error = lateral offset + heading component
        error = lateral_offset + heading_error * 0.5
        
        # PID calculation
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)  # Anti-windup
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        # PID output
        steering = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        steering = np.clip(steering, -45, 45)  # degrees
        
        # Risk assessment
        risk_level = self._assess_risk(lateral_offset, heading_error, velocity)
        
        # Throttle control
        if risk_level >= 4:
            throttle = 0.3  # Slow down
        elif risk_level >= 2:
            throttle = 0.5
        else:
            throttle = 0.7  # Normal speed
        
        # Warning message
        warnings = {
            0: "SAFE",
            1: "MONITOR",
            2: "CAUTION",
            3: "WARNING",
            4: "CRITICAL",
            5: "EMERGENCY"
        }
        
        return {
            'steering': float(steering),
            'throttle': float(throttle),
            'risk_level': risk_level,
            'warning': warnings.get(risk_level, "UNKNOWN"),
            'should_intervene': risk_level >= 3
        }
    
    def _assess_risk(
        self,
        lateral_offset: float,
        heading_error: float,
        velocity: float
    ) -> int:
        """
        Assess departure risk (0-5)
        """
        # Distance to lane edge
        distance_to_edge = self.track_width / 2 - abs(lateral_offset)
        
        # Time to crossing (TTC)
        lateral_velocity = velocity * np.sin(heading_error)
        
        if abs(lateral_velocity) > 0.01:
            ttc = distance_to_edge / abs(lateral_velocity)
        else:
            ttc = float('inf')
        
        # Risk levels
        if distance_to_edge < 0:
            return 5  # OFF TRACK
        elif ttc < 0.5:
            return 4  # CRITICAL
        elif ttc < 1.0:
            return 3  # WARNING
        elif ttc < 2.0:
            return 2  # CAUTION
        elif abs(lateral_offset) > self.safe_distance:
            return 1  # MONITOR
        else:
            return 0  # SAFE
    
    def reset(self):
        """Reset PID state"""
        self.integral = 0.0
        self.prev_error = 0.0
