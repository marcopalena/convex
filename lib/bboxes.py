import numpy as np
from scipy.spatial.transform import Rotation as R

from enum import Enum

class ReferenceFrame(Enum):
    WORLD_FRAME = 0,
    EGO_VEHICLE_FRAME = 1,
    SENSOR_FRAME = 2

class BBox:
    """
    A class representing a 3D bounding box. All location data is given with respect to the global coordinate system.
    """

    def __init__(self, corners, reference_frame):

        # Initialize bounding box info
        self.corners = corners
        if reference_frame == 0:
            self.reference_frame = ReferenceFrame.WORLD_FRAME
        elif reference_frame == 1:
            self.reference_frame = ReferenceFrame.EGO_VEHICLE_FRAME
        elif reference_frame == 2:
            self.reference_frame = ReferenceFrame.SENSOR_FRAME

        # Calculate bounding box center, dimensions and orientation
        center, dimensions, orientation = self.bounding_box_properties(corners)
        self.center = center
        self.dimensions = dimensions
        self.orientation = orientation


    def bounding_box_properties(self, corners):
        """
        Calculate the center, dimensions, and orientation of a bounding box given its 8 corner points in 3D space.

        Parameters:
            corners (list or numpy.ndarray): A (8, 3) array representing the 3D coordinates of the bounding box corners,
                assumed to be ordered such that edges are aligned with principal axes (RBL, FBL, FBR, RBR, RTL, FTL, FTR, RTR).

        Returns:
            center (numpy.ndarray): A (3,) array representing the center of the bounding box.
            dimensions (numpy.ndarray): A (3,) array representing the width, height, and depth of the bounding box.
            quaternion (numpy.ndarray): A (4,) array representing the orientation of the bounding box as a quaternion (x, y, z, w).
        """

        # Convert corners to a NumPy array for easier manipulation
        corners = np.array(corners)
        if corners.shape != (8, 3):
            raise ValueError("Input must be an (8, 3) array representing the corner points of the bounding box.")

        # Compute the center of the bounding box
        center = np.mean(corners, axis=0)

        # Compute the vectors representing the edges of the box
        edges = np.array([
            corners[1] - corners[0],  # Edge from rear bottom left corner to front bottom left corner ()
            corners[3] - corners[0],  # Edge from rear bottom left corner to rear bottom right corner
            corners[4] - corners[0],  # Edge from rear bottom left corner to rear top left corner
        ])

        # The dimensions are the magnitudes of the edge vectors
        dimensions = np.linalg.norm(edges, axis=1)

        # Normalize the edge vectors to find the principal axes
        principal_axes = np.array([edge / np.linalg.norm(edge) for edge in edges])

        # The orientation is represented by the rotation matrix formed by the principal axes
        rotation_matrix = np.column_stack(principal_axes)

        # Convert the rotation matrix to a quaternion
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        # Return bounding-box center, dimensions and orientation
        return center, dimensions, quaternion

    def contains_point(self, point):
        """
        Checks if a point lies within a the 3D bounding box.

        Args:
            point (tuple or list or np.ndarray): The 3D coordinates of the point (x, y, z).

        Returns:
            bool: True if the point lies inside the bounding box, False otherwise.
        """

        # Convert inputs to NumPy arrays
        point = np.array(point)
        
        # Compute the rotation matrix from the quaternion
        rotation_matrix = R.from_quat(self.orientation).as_matrix()
        
        # Transform the point into the local coordinate system of the box
        local_point = np.dot(rotation_matrix.T, point - self.center)
        
        # Check if the transformed point is within the axis-aligned box
        return all(-self.dimensions <= local_point) and all(local_point <= self.dimensions)

    @classmethod
    def average(cls, bboxes):
        """
        Computes an average bounding box from a list of bounding boxes with the same .

        Parameters:
            bboxes ([bboxes.BBox]): A list of bboxes to average.

        Returns:
            bbox (bboxes.BBox): The average bbox computed.
        """

        # Sanity checks
        assert all([bbox.reference_frame == bboxes[0].reference_frame for bbox in bboxes])

        # Average corner points
        corners = np.stack([bbox.corners for bbox in bboxes]).mean(axis=0)
        
        # Return the average bbox
        return BBox(corners, bboxes[0].reference_frame)

    ###################################################
    # PROPERTIES
    ###################################################

    def is_world_coordinates(self):
        return self.reference_frame == ReferenceFrame.WORLD_FRAME
    
    def is_ego_coordinates(self):
        return self.reference_frame == ReferenceFrame.EGO_VEHICLE_FRAME
    
    def is_sensor_coordinates(self):
        return self.reference_frame == ReferenceFrame.SENSOR_FRAME

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"BBox(center='{self.center}', dimensions={self.dimensions}, orientation={self.orientation})"

    def __repr__(self):
        return f"BBox(center='{self.center}', dimensions={self.dimensions}, orientation={self.orientation})"
