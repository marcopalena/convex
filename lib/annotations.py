import shortuuid

class Instance:
    """
    A class to represent target instances, e.g. particular vehicle or pedestrian in a scene. 
    """

    def __init__(self, target, scene):

        # Initialize instance
        self.token = shortuuid.uuid()
        self.scene = scene
        self.target = target

        # Keep a list of sample annotations referring to this instance
        self.annotations = []

    ###################################################
    # PROPERTIES
    ###################################################

    @property
    def num_annotations(self):
        return len(self.annotations)

    @property
    def id(self):
        return f"{self.target.id}@{self.scene.id}"
    
    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"Instance(id={self.id}, class='{self.target.supertype}.{self.target.type}', nbr_annotation={self.num_annotations})"

    def __repr__(self):
        return f"Instance(id={self.id}, class='{self.target.supertype}.{self.target.type}', nbr_annotation={self.num_annotations})"
    
class SampleAnnotation:
    """
    A class to represent a sample annotation. Annotates a given sample with a bounding box defining the position of an 
    object instance seen in a sample.
    """

    def __init__(self, sample, bbox, instance, attributes, visibility_level):

        # Initialize sample annotation
        self.token = shortuuid.uuid()
        self.instance = instance
        self.sample = sample
        self.bbox = bbox
        self.attributes = attributes
        self.visibility_level = visibility_level

        # Count number of LiDAR points in the bounding box
        self.num_lidar_points = self.count_lidar_points_in_box(sample, bbox)


    def count_lidar_points_in_box(self, sample, bbox):
        """
        Counts the number of LiDAR points lying within a given bounding box for a sample.
        
        Parameters:
            sample (samples.Sample): A sample object.
            bbox (bboxes.BBox): A bounding box object.

        Returns:
            (int): number of LiDAR points in the bounding box
        """

        # Initialize counter
        count = 0

        # Iterate over each point cloud sample data
        for point_cloud_sample_data in sample.point_cloud_sample_data.values():

            # Get LiDAR point cloud
            lidar_point_cloud =  point_cloud_sample_data.point_cloud

            # Count point within the given bbox
            for point in lidar_point_cloud.positions:
                if bbox.contains_point(point):
                    count += 1

        # Return number of LiDAR points in box
        return count

    ###################################################
    # PROPERTIES
    ###################################################

    @property
    def frame(self):
        return self.sample.frame_number
    
    @property
    def target_supertype(self):
        return self.instance.target.supertype

    @property
    def target_type(self):
        return self.instance.target.type

    ###################################################
    # MAGIC METHODS
    ###################################################
    
    def __str__(self):
        return f"SampleAnnotation(sample={self.sample.id}, target={self.instance.target.id}, class='{self.instance.target.category}', num_lidar_points={self.num_lidar_points})"

    def __repr__(self):
        return f"SampleAnnotation(sample={self.sample.id}, target={self.instance.target.id}, class='{self.instance.target.category}', num_lidar_points={self.num_lidar_points})"
    