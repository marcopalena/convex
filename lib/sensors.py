import numpy as np
import shortuuid
from enum import Enum
from scipy.spatial.transform import Rotation as R

from .utils import get_world_to_local_matrix, compute_intrinsic_matrix

class SensorModality(Enum):
    CAMERA = 1
    LIDAR = 2
    RADAR = 3
    GNSS = 4

class LidarParameters:
    """
    A class to describe the parameters of a calibrated LiDAR sensor.
    """

    def __init__(self, max_range_m, total_fov_deg, speed_tr_per_min, resolution_deg, vertical_ray_num, horizontal_ray_num, elev_min_angle_deg, elev_max_angle_deg):
        
        # Initialize LiDAR parameters
        self.max_range_m = max_range_m
        self.total_fov_deg = total_fov_deg
        self.speed_tr_per_min = speed_tr_per_min
        self.resolution_deg = resolution_deg
        self.vertical_ray_num = vertical_ray_num
        self.horizontal_ray_num = horizontal_ray_num
        self.elev_min_angle_deg = elev_min_angle_deg
        self.elev_max_angle_deg = elev_max_angle_deg

class CameraIntrinsic:
    """
    A class to describe the intrinsic parameters of a calibrated camera.
    """

    def __init__(self, resolution_width_px, resolution_height_px, focal_length_mm, sensor_width_mm, sensor_height_mm, horizontal_fov_deg, vertical_fov_deg):
        
        # Initialize camera intrinsics
        self.resolution_width_px = resolution_width_px
        self.resolution_height_px = resolution_height_px
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.horizontal_fov_deg = horizontal_fov_deg
        self.vertical_fov_deg = vertical_fov_deg

        # Create calibration matrix
        self.calibration_matrix = compute_intrinsic_matrix(resolution_width_px, resolution_height_px, horizontal_fov_deg, vertical_fov_deg)

    def __str__(self):
        return f"CameraIntrinsic(calibration_matrix={self.calibration_matrix})"

    def __repr__(self):
        return f"CameraIntrinsic(calibration_matrix={self.calibration_matrix})"

    def info(self):
        print(f"W: {self.resolution_width_px}, H: {self.resolution_height_px}, fl: {self.focal_length_mm}, sw: {self.sensor_width_mm}, sh: {self.sensor_height_mm}, hfov: {self.horizontal_fov_deg}, vfov: {self.vertical_fov_deg}" )

class SensorType:
    """
    A class to describe a specific sensor type within the simulation.
    """

    def __init__(self, channel, modality, modality_str):
        """
        Initialize a new sensor type. 
        :param channel: Sensor channel name.
        :param modality: Sensor modality.
        """

        # Initialize sensor type
        self.token = shortuuid.uuid()
        self.channel = channel
        self.modality = modality
        self.modality_str = modality_str

    def __str__(self):
        return f"SensorType(token='{self.token}', channel='{self.channel}', modality={self.modality})"

    def __repr__(self):
        return f"SensorType(token='{self.token}', channel='{self.channel}', modality={self.modality})"


class CalibratedSensor:
    """
    A class to describe a particular sensor as calibrated on a particular agent.
    """

    def __init__(self, sensor_type, agent, name, translation, rotation):

        # Initialize calibrated sensor
        self.token = shortuuid.uuid()
        self.sensor_type = sensor_type
        self.agent = agent
        self.name = name
        self.translation = translation
        self.rotation = rotation

        # Get world to ego transformation matrix
        self.transform_matrix = get_world_to_local_matrix(translation, rotation)

    def rotation_quaternion(self):
        return R.from_euler('ZYX', self.rotation, degrees=False).as_quat()

    def rotation_matrix(self):
        return R.from_euler('ZYX', self.rotation, degrees=False).as_matrix()

    def __str__(self):
        return f"CalibratedSensor(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"

    def __repr__(self):
        return f"CalibratedSensor(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"

class CalibratedCamera(CalibratedSensor):
    """
    A class to describe a particular camera as calibrated on a particular agent.
    """

    def __init__(self, sensor_type, agent, name, translation, rotation, camera_intrinsic):

        # Initialize calibrated camera
        super().__init__(sensor_type, agent, name, translation, rotation)
        self.camera_intrinsic = camera_intrinsic

    def world_camera_pose(self, agent_pose):
        """
        Return the camera pose in world coordinates.

        Parameters:
            agent_pose (EgoPose): the pose of the agent that is equipped with the camera.

        Returns:
            (EgoPose) the camera pose in world coordinates.
        """

        # Transform camera center in homogeneous coordinates
        camera_center_homogeneous_local = np.array([*self.translation, 1])

        # Perform the transformation
        camera_center_homogeneous_world =  np.dot(agent_pose.transform_matrix, camera_center_homogeneous_local)

        # Compose world camera rotation
        camera_rotation_world = R.from_matrix(np.dot(agent_pose.rotation_matrix(), self.rotation_matrix())).as_euler("ZYX")

        # Return world camera pose
        return camera_center_homogeneous_world[:3], camera_rotation_world

    def world_to_camera(self, world_point, agent_pose):
        """
        Transforms a point from world coordinates to camera coordinates.

        Parameters:
            world_point (np.ndarray): A numpy array of shape (3,) or (4,) representing the point in world coordinates.
            agent_pose (EgoPose): the pose of the agent that is equipped with the camera.

        Returns:
            (np.ndarray) A numpy array of shape (3,) representing the point in camera coordinates.
        """

        # Ensure the world_point is in homogeneous coordinates
        if world_point.shape == (3,):
            world_point = np.append(world_point, 1)

        # Transform the point
        world_to_agent = np.linalg.inv(agent_pose.transform_matrix) @ world_point
        camera_point = np.linalg.inv(self.transform_matrix) @ world_to_agent

        # Convert back to 3D (non-homogeneous coordinates)
        return camera_point[:3] / camera_point[3]

    def __str__(self):
        return f"CalibratedCamera(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"

    def __repr__(self):
        return f"CalibratedCamera(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"
    

class CalibratedLiDAR(CalibratedSensor):
    """
    A class to describe a particular LiDAR sensor as calibrated on a particular agent.
    """

    def __init__(self, sensor_type, agent, name, translation, rotation, lidar_parameters):

        # Initialize calibrated LiDAR sensor
        super().__init__(sensor_type, agent, name, translation, rotation)
        self.lidar_parameters = lidar_parameters

    def __str__(self):
        return f"CalibratedLiDAR(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"

    def __repr__(self):
        return f"CalibratedLiDAR(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"
    
class CalibratedGNSS(CalibratedSensor):
    """
    A class to describe a particular GNSS sensor as calibrated on a particular agent.
    """

    def __init__(self, sensor_type, agent, name, translation, rotation):

        # Initialize calibrated GNSS sensor
        super().__init__(sensor_type, agent, name, translation, rotation)

    def __str__(self):
        return f"CalibratedGNSS(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"

    def __repr__(self):
        return f"CalibratedGNSS(token='{self.token}', agent_id='{self.agent.id}', translation={self.translation}, rotation={self.rotation})"
    