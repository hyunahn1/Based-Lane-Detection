"""
CARLA Interface
Connection, vehicle, sensor management
"""
import carla
import numpy as np
from typing import Optional, Dict
import time


class CarlaInterface:
    """
    CARLA 시뮬레이터 인터페이스
    """
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 2000,
        timeout: float = 10.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None
        
        self.latest_image: Optional[np.ndarray] = None
    
    def connect(self):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA at {self.host}:{self.port}...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        print("✅ Connected to CARLA")
    
    def spawn_vehicle(self, spawn_point: Optional[carla.Transform] = None):
        """Spawn RC car"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Use small vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"✅ Vehicle spawned at {spawn_point.location}")
        
        # Wait for vehicle to settle
        time.sleep(1.0)
        
        return self.vehicle
    
    def spawn_camera(self):
        """Spawn RGB camera"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Camera settings
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        
        # Mount on vehicle (front, center, elevated)
        camera_transform = carla.Transform(
            carla.Location(x=0.5, z=0.3)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # Listen to camera
        self.camera.listen(self._on_camera_update)
        print("✅ Camera spawned and listening")
        
        return self.camera
    
    def _on_camera_update(self, image):
        """Camera callback"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # Remove alpha
        array = array[:, :, ::-1]  # BGR to RGB
        
        self.latest_image = array.copy()
    
    def get_latest_image(self) -> Optional[np.ndarray]:
        """Get latest camera image"""
        return self.latest_image
    
    def get_vehicle_state(self) -> Dict:
        """Get vehicle state"""
        if self.vehicle is None:
            return {}
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return {
            'location': transform.location,
            'rotation': transform.rotation,
            'velocity': speed,
            'heading': transform.rotation.yaw
        }
    
    def get_lane_info(self) -> Dict:
        """
        Get lane information for RL
        
        Returns:
            lateral_offset: 차선 중심으로부터의 횡방향 거리 (m)
            heading_error: 차선 방향과의 각도 차이 (rad)
        """
        if self.vehicle is None:
            return {'lateral_offset': 0.0, 'heading_error': 0.0}
        
        # Get current waypoint
        vehicle_location = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        
        if current_waypoint is None:
            return {'lateral_offset': 0.0, 'heading_error': 0.0}
        
        # Calculate lateral offset
        waypoint_location = current_waypoint.transform.location
        vehicle_transform = self.vehicle.get_transform()
        
        # Vector from waypoint to vehicle
        dx = vehicle_location.x - waypoint_location.x
        dy = vehicle_location.y - waypoint_location.y
        
        # Waypoint forward vector
        waypoint_yaw = np.deg2rad(current_waypoint.transform.rotation.yaw)
        waypoint_forward = np.array([np.cos(waypoint_yaw), np.sin(waypoint_yaw)])
        
        # Calculate lateral offset (perpendicular distance)
        vehicle_vector = np.array([dx, dy])
        lateral_offset = np.cross(waypoint_forward, vehicle_vector)
        
        # Calculate heading error
        vehicle_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
        heading_error = np.arctan2(
            np.sin(vehicle_yaw - waypoint_yaw),
            np.cos(vehicle_yaw - waypoint_yaw)
        )
        
        return {
            'lateral_offset': float(lateral_offset),
            'heading_error': float(heading_error)
        }
    
    def get_obstacle_distance(self) -> float:
        """
        Get distance to nearest obstacle
        
        Returns:
            distance: 가장 가까운 차량까지의 거리 (m), 없으면 10.0
        """
        if self.vehicle is None:
            return 10.0
        
        vehicle_location = self.vehicle.get_location()
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        min_distance = 10.0  # Default max distance
        
        for other_vehicle in vehicle_list:
            if other_vehicle.id == self.vehicle.id:
                continue
            
            other_location = other_vehicle.get_location()
            distance = vehicle_location.distance(other_location)
            
            if distance < min_distance:
                min_distance = distance
        
        return float(min_distance)
    
    def apply_control(self, steering: float, throttle: float):
        """
        Apply vehicle control
        
        Args:
            steering: degrees [-45, 45]
            throttle: [0, 1]
        """
        if self.vehicle is None:
            return
        
        control = carla.VehicleControl()
        control.steer = np.clip(steering / 45.0, -1.0, 1.0)  # Normalize to [-1, 1]
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = 0.0
        control.hand_brake = False
        
        self.vehicle.apply_control(control)
    
    def cleanup(self):
        """Cleanup resources"""
        actors_to_destroy = []
        
        if self.camera:
            actors_to_destroy.append(self.camera)
        if self.vehicle:
            actors_to_destroy.append(self.vehicle)
        
        if self.client:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
        
        print("✅ Cleanup complete")
