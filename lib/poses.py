import shortuuid
import numpy as np
import datetime
from .utils import get_world_to_local_matrix
from scipy.spatial.transform import Rotation as R

EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
DEFAULT_TRANSLATIONAL_SPEED_THRESHOLD = 0.01
DEFAULT_ROTATIONAL_SPEED_THRESHOLD = 0.001

class EgoPose:

    def __init__(self, timestamp, translation, rotation):
        """
        Initialize an ego vehicle pose.

        Parameters:

            timestamp (float): the timestamp in seconds
            translation (np.ndarray): the vehicle translation in m as a numpy array containing (X, Y, Z)
            rotation (np.ndarray): the vehicle rotation in rads as a numpy array containing (Heading, Pitch, Roll)
        """

        # Initialize ego pose
        self.token = shortuuid.uuid()
        self.timestamp = EPOCH + datetime.timedelta(seconds=timestamp)
        self.translation = translation
        self.rotation = rotation

        # Get world to ego transformation matrix
        self.transform_matrix = get_world_to_local_matrix(translation, rotation)
    
    def rotation_quaternion(self):
        return R.from_euler('ZYX', self.rotation, degrees=False).as_quat()

    def rotation_matrix(self):
        return R.from_euler('ZYX', self.rotation, degrees=False).as_matrix()
    
    def world_to_ego(self, world_point):
        """
        Transforms a point from world coordinates to ego coordinates.

        Parameters:
        - world_point: A numpy array of shape (3,) or (4,) representing the point in world coordinates.

        Returns:
        - A numpy array of shape (3,) representing the point in camera coordinates.
        """

        # Ensure the world_point is in homogeneous coordinates
        if world_point.shape == (3,):
            world_point = np.append(world_point, 1)

        # Transform the point
        ego_point = np.linalg.inv(self.transform_matrix) @ world_point

        # Convert back to 3D (non-homogeneous coordinates)
        return ego_point[:3] / ego_point[3]

    def __str__(self):
        return f"EgoPose(token='{self.token}', timestamp='{self.timestamp.timestamp()}', translation={self.translation}, rotation={self.rotation})"

    def __repr__(self):
        return f"EgoPose(token='{self.token}', timestamp='{self.timestamp.timestamp()}', translation={self.translation}, rotation={self.rotation})"
    
class EgoSpeed:

    def __init__(self, timestamp, translational_speed, rotational_speed):
        """
        Initialize an ego vehicle speed.

        Parameters:

            timestamp (float): the timestamp in seconds
            translational_speed (np.ndarray): the vehicle translational speed in m/s as a numpy array containing (X, Y, Z)
            rotational_speed (np.ndarray): the vehicle rotational speed in rad/s as a numpy array containing (Heading, Pitch, Roll)
        """

        # Initialize ego pose
        self.token = shortuuid.uuid()
        self.timestamp = EPOCH + datetime.timedelta(seconds=timestamp)
        self.translational_speed = translational_speed
        self.rotational_speed = rotational_speed

    def is_moving(self):
        """
        Checks whether the speed indicates the reference object is moving or stationary based on its linear and rotational speeds.
        Returns:
            bool: True if the vehicle is moving, False if stationary.
        """

        # Compute speed components magnitude
        linear_speed_magnitude = np.linalg.norm(self.translational_speed)
        rotational_speed_magnitude = np.linalg.norm(self.rotational_speed)

        # Check if both speeds are below their respective thresholds
        is_stationary = (linear_speed_magnitude < DEFAULT_TRANSLATIONAL_SPEED_THRESHOLD) and (rotational_speed_magnitude < DEFAULT_ROTATIONAL_SPEED_THRESHOLD)
        return not is_stationary

    def __str__(self):
        return f"EgoSpeed(token='{self.token}', timestamp='{self.timestamp.timestamp()}', translational_speed={self.translational_speed}, rotational_speed={self.rotational_speed})"

    def __repr__(self):
        return f"EgoSpeed(token='{self.token}', timestamp='{self.timestamp.timestamp()}', translational_speed={self.translational_speed}, rotational_speed={self.rotational_speed})"