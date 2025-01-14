import shortuuid
from collections import defaultdict

class Sample:

    def __init__(self, scene, frame_number, timestamp, is_key_frame):

        # Initialize sample
        self.token = shortuuid.uuid()
        self.scene = scene
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.is_key_frame = is_key_frame
        self.image_sample_data = {}
        self.point_cloud_sample_data = {}
        self.gnss_coordinates_sample_data = {}
        self.annotations = []

    @property
    def id(self):
        return f"{self.scene.id}.{self.frame_number}"

class SampleData:
    def __init__(self, timestamp, path, format, sample, ego_pose, sensor):

        # Initialize sample data
        self.token = shortuuid.uuid()
        self.timestamp = timestamp
        self.path = path
        self.format = format
        self.sample = sample
        self.ego_pose = ego_pose
        self.sensor = sensor

class ImageSampleData(SampleData):
    def __init__(self, timestamp, path, format, image, sample, ego_pose, sensor):
        super().__init__(timestamp, path, format, sample, ego_pose, sensor)
        self.height, self.width = image.shape[:2]
        # self.image = image

    def __str__(self):
        return f"ImageSampleData(token='{self.token}', timestamp='{self.timestamp}', path={self.path}, height={self.height}, width={self.width})"

    def __repr__(self):
        return f"ImageSampleData(token='{self.token}', timestamp='{self.timestamp}', path={self.path}, height={self.height}, width={self.width})"
    

class PointCloudSampleData(SampleData):
    def __init__(self, timestamp, path, format, point_cloud, sample, ego_pose, sensor):
        super().__init__(timestamp, path, format, sample, ego_pose, sensor)
        self.point_cloud = point_cloud

    def __str__(self):
        return f"PointCloudSampleData(token='{self.token}', timestamp='{self.timestamp}', path={self.path}, num_points={len(self.point_cloud)})"

    def __repr__(self):
        return f"PointCloudSampleData(token='{self.token}', timestamp='{self.timestamp}', path={self.path}, num_points={len(self.point_cloud)})"
    
class GNSSCoordinatesSampleData(SampleData):
    def __init__(self, timestamp, path, format, gnss_coordinates, sample, ego_pose, sensor):
        super().__init__(timestamp, path, format, sample, ego_pose, sensor)
        self.gnss_coordinates = gnss_coordinates

    def __str__(self):
        return f"GNSSCoordinatesSampleData(token='{self.token}', timestamp='{self.timestamp}', path={self.path}, gnss_coordinates={self.gnss_coordinates})"

    def __repr__(self):
        return f"GNSSCoordinatesSampleData(token='{self.token}', timestamp='{self.timestamp}', path={self.path}, gnss_coordinates={self.gnss_coordinates})"