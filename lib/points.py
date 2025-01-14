import open3d as o3d
import open3d.core as o3c

class LidarPointAnnotation:
    """
    A class to represent an annotation attached to a LiDAR point.
    """

    def __init__(self, lidar_point, target_id, target_type):

        # Initialize LiDAR point annotation
        self.lidar_point = lidar_point
        self.target_id = target_id
        self.target_type = target_type

    def __str__(self):
        return f"LidarPointAnnotation(target_id={self.target_id}, target_type={self.target_type})"

    def __repr__(self):
        return f"LidarPointAnnotation(target_id={self.target_id}, target_type={self.target_type})"

class LidarPoint:
    """
    A class to represent a single 3D point detected by a LiDAR scan.
    """

    def __init__(self, coordinates, normal, intensity, distance):

        # Initialize LiDAR point
        self.coordinates = coordinates
        self.normal = normal
        self.intensity = intensity
        self.distance = distance

    def __str__(self):
        return f"LidarPoint(coordinates='{self.coordinates}')"

    def __repr__(self):
        return f"LidarPoint(coordinates='{self.coordinates}')"

class LidarPointCloud:
    """
    A class to represent a cloud of points detected by a LiDAR scan.
    """

    def __init__(self, positions, normals, intensities, target_ids, target_types):

        # Initialize LiDAR point cloud
        self.positions = positions
        self.normals = normals
        self.intensities = intensities
        self.target_ids = target_ids
        self.target_types = target_types

    def to_open3d(self):

        # Compose point cloud spec
        point_cloud_spec = {}
        point_cloud_spec["positions"] = o3c.Tensor(self.positions, o3c.float32)
        point_cloud_spec["normals"] = o3c.Tensor(self.normals, o3c.float32)
        point_cloud_spec["intensities"] = o3c.Tensor(self.intensities, o3c.float32)

        # Create and return Open3D point cloud
        return o3d.t.geometry.PointCloud(point_cloud_spec).to_legacy()