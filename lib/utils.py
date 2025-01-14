import cv2
import math
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


def compute_intrinsic_matrix(image_width, image_height, hfov, vfov):
    """
    Computes the intrinsic matrix of a camera given its image resolution and HFOV.

    Args:
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        hfov (float): Horizontal field of view in degrees.
        vfov (float): Vertical field of view in degrees.

    Returns:
        np.ndarray: A 3x3 intrinsic matrix.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive integers.")
    if hfov <= 0 or hfov >= 180:
        raise ValueError("HFOV must be in the range (0, 180) degrees.")
    if vfov <= 0 or vfov >= 180:
        raise ValueError("VFOV must be in the range (0, 180) degrees.")

    # Convert HFOV and VFOV from degrees to radians
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)

    # Compute focal lengths
    fx = image_width / (2 * math.tan(hfov_rad / 2))
    fy = image_height / (2 * math.tan(vfov_rad / 2))
    # fy = fx * (image_height / image_width)  # Assume square pixels

    # Compute principal point (image center)
    cx = image_width / 2
    cy = image_height / 2

    # Construct the intrinsic matrix
    intrinsic_matrix = np.array([
        [fx,  0,  cx],
        [ 0, fy,  cy],
        [ 0,  0,   1]
    ])

    return intrinsic_matrix

def translation_to_transform(translation):
    """
    Converts a 3D translation into a homogeneous transformation matrix.

    Args:
        translation (tuple or list or np.ndarray): A 3D position (x, y, z).

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    if len(translation) != 3:
        raise ValueError("Position must be a 3-element tuple, list, or array.")

    # Initialize the homogeneous transformation matrix as an identity matrix
    transform = np.eye(4)

    # Set the translation part
    transform[:3, 3] = translation

    return transform

def rotation_to_transform(rotation):
    """
    Converts a 3D rotation into a homogeneous transformation matrix.

    Args:
        rotation_angles: A tuple or list (heading, pitch, roll) in degrees.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    if len(rotation) != 3:
        raise ValueError("Position must be a 3-element tuple, list, or array.")

    # Initialize the homogeneous transformation matrix as an identity matrix
    transform = np.eye(4)

    rotation_matrix = R.from_euler('ZYX', rotation, degrees=False).as_matrix()

    print(rotation)

    # Set the translation part
    transform[:3, :3] = rotation_matrix

    return transform

def get_world_to_local_matrix(translation, rotation_angles):
    """
    Computes the transformation matrix from world coordinates to local coordinates.

    Parameters:
    - translation: A tuple or list (X, Y, Z) representing the translation of the camera.
    - rotation_angles: A tuple or list (heading, pitch, roll) in degrees.

    Returns:
    - A 4x4 numpy array representing the transformation matrix.
    """

    # Unpack translation
    x, y, z = translation

    # Create a rotation matrix from heading, pitch, roll
    # Heading (yaw), pitch, roll correspond to ZYX intrinsic rotations
    rotation_matrix = R.from_euler('ZYX', rotation_angles, degrees=False).as_matrix()

    # Create the transformation matrix
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set the rotation part
    transformation_matrix[:3, 3] = np.array([x, y, z])  # Set the translation part

    return transformation_matrix


def draw_2d_box(image, top_left, bottom_right, color=(0, 255, 0), thickness=1):
    """
    Draws a 2D box on an image given its pixel coordinates.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        top_left (tuple): The (x, y) coordinates of the top-left corner of the box.
        bottom_right (tuple): The (x, y) coordinates of the bottom-right corner of the box.
        color (tuple): The color of the box in BGR format (default is green).
        thickness (int): The thickness of the box outline in pixels (default is 2).

    Returns:
        np.ndarray: The image with the drawn 2D box.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")
    if not (isinstance(top_left, tuple) and len(top_left) == 2):
        raise ValueError("top_left must be a tuple of two coordinates (x, y).")
    if not (isinstance(bottom_right, tuple) and len(bottom_right) == 2):
        raise ValueError("bottom_right must be a tuple of two coordinates (x, y).")

    # Draw the rectangle on the image
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    cv2.line(image, top_left, top_right, color, thickness)
    cv2.line(image, top_right, bottom_right, color, thickness)
    cv2.line(image, bottom_right, bottom_left, color, thickness)
    cv2.line(image, bottom_left, top_left, color, thickness)
    
    return image

def draw_3d_box(image, corners, color=(0, 255, 0), thickness=1):
    """
    Draws a 3D box on an image given its pixel coordinates.

    Args:
        image (np.ndarray): The image on which to draw the box (H x W x 3).
        corners (list or np.ndarray): A list or array of shape (8, 2), containing the pixel coordinates of the box's corners.
        color (tuple): The color of the box lines in (B, G, R) format (default is green).
        thickness (int): The thickness of the box lines (default is 2).

    Returns:
        np.ndarray: The image with the 3D box drawn on it.
    """
    if len(corners) != 8 or any(len(corner) != 2 for corner in corners):
        raise ValueError("Corners must be a list or array of 8 (x, y) coordinate pairs.")

    # Convert corners to a NumPy array
    corners = np.array(corners, dtype=int)

    # Define the edges of the 3D box
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Draw the edges on the image
    for start, end in edges:
        cv2.line(image, tuple(corners[start]), tuple(corners[end]), color, thickness)

    return image

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)

    viewpad[:view.shape[0], :view.shape[1]] = view
    np.set_printoptions(suppress=True)

    alignpad = np.eye(4)
    align = R.from_euler('XYZ', [np.pi/2, 0, -np.pi/2], degrees=False).as_matrix()
    alignpad[:3, :3] = align
    # print(alignpad)


    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    # print(points)
    # print(points.shape)
    points = np.dot(alignpad, points)
    points = np.dot(viewpad, points)
    # print(points)
    points = points[:3, :]


    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


@dataclass
class VehicleDimensions:
    """Class to represent vehicle dimensions."""
    length_mm: float
    width_mm: float
    height_mm: float
    weight_kg: float
    wheel_base_mm: float
    front_track_mm: float
    rear_track_mm: float
    rear_overhang_mm: float