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
