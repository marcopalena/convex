import os
import re
import shortuuid
import shutil
import logging
import cv2
import math
import datetime
import numpy as np
from enum import Enum
from collections import defaultdict
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c

from ..logs import SceneLog
from ..samples import Sample, ImageSampleData, PointCloudSampleData, GNSSCoordinatesSampleData
from ..annotations import Instance, SampleAnnotation
from ..bboxes import BBox
from ..points import LidarPointCloud
from ..targets import Vehicle, Pedestrian

###################################################
# CONSTANTS
###################################################

TIME_OF_DAY_2_LIGHT_COND = {
    "6AM" : "twilight",
    "10PM": "nighttime",
    "12PM": "daytime"
}

DEFAULT_SAMPLED_DATA_FOLDER = "sampled"     # Default name of the sampled data folder
DEFAULT_SAMPLED_FRAMES_FOLDER = "frames"    # Default name of the sampled image frames folder
DEFAULT_SAMPLED_POINTS_FOLDER = "points"    # Default name of the sampled point clouds folder
DEFAULT_SAMPLED_GNSS_FOLDER = "gnss"        # Default name of the sampled gnss coordinates folder

DEFAULT_IMAGE_FRAME_FORMAT = "PNG"
DEFAULT_POINTS_CLOUD_FORMAT = "PCD"
DEFAULT_GNSS_COORDINATES_FORMAT = "NPY"        

###################################################
# REGEXPS
###################################################

VEHICLE_AGENT_FEED_REGEXP = ".*_Veh([0-9]+)_"
ROADSIDE_AGENT_FEED_REGEXP = ".*_Ped([0-9]+)_"

###################################################
# UTILITY CLASSES
###################################################

class SceneState(Enum):
    LOADED = 1
    SAMPLED = 2
    ANNOTATED = 3

###################################################
# SCENE
###################################################

class Scene:
    """
    A class representing a scene.
    """

    def __init__(self, scene_log_path, base_scene, weather_cond_label, time_of_day_label, timestamp):

        # Initialize scene
        self.token = shortuuid.uuid()
        self.base_scene = base_scene
        self.scene_log_path = scene_log_path
        self.weather_cond_label = weather_cond_label
        self.time_of_day_label = time_of_day_label
        self.timestamp = timestamp
        self.samples = []
        self.ego_poses = {}

        # Set up sampled data folder paths
        self.sampled_data_folder = os.path.join(
            self.base_scene.dataset.output_folder_path, 
            DEFAULT_SAMPLED_DATA_FOLDER, 
            self.base_scene.scenario, 
            self.base_scene.id,
            self.weather_cond_label, 
            self.time_of_day_label
        )
        self.frames_folder = os.path.join(self.sampled_data_folder, DEFAULT_SAMPLED_FRAMES_FOLDER)
        self.points_folder = os.path.join(self.sampled_data_folder, DEFAULT_SAMPLED_POINTS_FOLDER)
        self.gnss_folder = os.path.join(self.sampled_data_folder, DEFAULT_SAMPLED_GNSS_FOLDER)

        # Create scene log
        self.log = SceneLog(scene_log_path, self.timestamp, self)

        # Compute valid sampling window
        self.start_frame, self.end_frame = self.log.compute_valid_sampling_window()

        # Set scene state as loaded
        self.state = SceneState.LOADED

    ###################################################
    # PROPERTIES
    ###################################################

    @property
    def light_cond_label(self):
        return TIME_OF_DAY_2_LIGHT_COND[self.time_of_day_label]
    
    @property
    def id(self):
        return f"{self.base_scene.id}.{self.weather_cond_label}.{self.time_of_day_label}"
    
    @property
    def num_samples(self):
        return len(self.samples)
    
    @property
    def num_annotations(self):
        count = 0
        for sample in self.samples:
            count += len(sample.annotations)
        return count
    
    ###################################################
    # GETTERS
    ###################################################

    def is_nighttime(self):
        return self.light_cond_label == "nighttime"

    def is_daytime(self):
        return self.light_cond_label == "daytime"
    
    def is_twight(self):
        return self.light_cond_label == "twilight"
    

    ###################################################
    # SAMPLING METHODS
    ###################################################

    def sample(self, sampling_frequency_hz, start_frame, end_frame, clear = False, overwrite = True):
        """
        Sample scene data from logs. The sampling process creates a list of samples
        for the scene. Each sample may be associated with multipled sample data, including
        image frames, LiDAR point clouds, GNSS coordinates, etc.

        First the method defines a suitable sampling window for the scene by looking at
        the maximum frame interval for which all the exported data poiunts are aligned and 
        available.

        Parameters:
            sampling_frequency_hz (int): Frequency (Hz) at which data should be sampled.
            start_frame (int): Number of the frame from which to start sampling
            end_frame (int): Number of the frame up to which to stop sampling
            clear (bool): A boolean indicating whether already sampled data should be cleared before resampling.
            overwrite (bool): A boolean indicating whether or not already sampled data should be overwritten.
        """

        # Clear sampled data if required
        if clear:
            self.clear_sampled_data()

        # Determine frame spacing
        frame_spacing = math.floor(self.base_scene.scaner_sampling_freq_hz/sampling_frequency_hz)

        # Initialize sampling process
        self.initialize_sampling(start_frame, end_frame, frame_spacing)

        # Initialize scenes's samples
        self.samples = []

        # Initialize counters
        count = 0
        num_image_frames = 0
        num_point_clouds = 0
        num_gnss_coordinates = 0

        # Sample scene
        frame = start_frame 
        for frame in tqdm(range(start_frame, end_frame, frame_spacing)):

            # Create sample
            time_offset = datetime.timedelta(seconds=frame / self.base_scene.scaner_sampling_freq_hz)
            timestamp = self.base_scene.timestamp + time_offset
            sample = Sample(self, frame, timestamp, True)

            # Extract image frame sample data for all agents
            num_image_frames += self.extract_image_frames(sample, overwrite)

            # Extract LiDAR points sample data for all agents
            num_point_clouds += self.extract_point_clouds(sample, overwrite)

            # Extract GNSS sample data for all agents
            num_gnss_coordinates += self.extract_gnss_coordinates(sample, overwrite)

            # Add sample to scene's samples
            self.samples.append(sample)
            count += 1

        # Log sampling results
        logging.info(f"Sampled {count} frames ({num_image_frames} imgs, {num_point_clouds} pcds, {num_gnss_coordinates} coords)")
            
        # Finalize sampling process
        self.finalize_sampling()

        # Set scene state as sampled
        self.state = SceneState.SAMPLED


    def clear_sampled_data(self):
        """
        Clears already sampled data.
        """

        # Remove sampled data directory tree
        if os.path.exists(self.sampled_data_folder):
            shutil.rmtree(self.sampled_data_folder)

    def initialize_sampling(self, start_frame, end_frame, frame_spacing):
        """
        Initializes the scene sampling process.

        Parameters:
            start_frame (int): Number of the frame from which to start sampling
            end_frame (int): Number of the frame up to which to stop sampling
            frame_spacing (int): Number of frames at which each new sample is taken.

        Returns:
            (dict): initialized sampling log.
        """

        # Create folder structure for sampled data
        self.create_sampled_data_folders()

        # Start video captures
        for video_log in self.log.video_feeds.values():
            video_log.init_capture()

    def create_sampled_data_folders(self):
        """
        Creates the folder structure to collect sampled data.
        """

        # Create base folders for sampled data
        if not os.path.exists(self.sampled_data_folder):
            os.makedirs(self.sampled_data_folder)
        if not os.path.exists(self.frames_folder):
            os.makedirs(self.frames_folder)
        if not os.path.exists(self.points_folder):
            os.makedirs(self.points_folder)
        if not os.path.exists(self.gnss_folder):
            os.makedirs(self.gnss_folder)

        # Create agents subfolders for sampled frames and point clouds
        for agent in self.base_scene.agents.values():
            agent_frame_folder = os.path.join(self.frames_folder, f"{agent.id:02d}")
            agent_points_folder = os.path.join(self.points_folder, f"{agent.id:02d}")
            agent_gnss_folder = os.path.join(self.gnss_folder, f"{agent.id:02d}")
            if not os.path.exists(agent_frame_folder):
                os.makedirs(agent_frame_folder)
            if not os.path.exists(agent_points_folder):
                os.makedirs(agent_points_folder)
            if not os.path.exists(agent_gnss_folder):
                os.makedirs(agent_gnss_folder)

    def finalize_sampling(self):
        """
        Finalizes the scene's sampling process.
        """

        # End video captures
        for video_log in self.log.video_feeds.values():
            video_log.end_capture()

    def extract_image_frames(self, sample, overwrite = True):
        """
        Populates the given sample with image frames data extracted from the video feed of each agent.
        """

        # Get sample frame number and timestamp
        frame = sample.frame_number
        timestamp = sample.timestamp

        # Extract image frames for each agent
        count = 0
        for agent in self.base_scene.agents.values():

            # Get agent's camera sensor
            if not agent.has_camera_sensor():
                continue
            sensor = agent.calibrated_camera

            # Get current agent's ego pose
            agent_ego_pose = self.log.vehicle_log.get_ego_pose(agent.id, frame)
            self.ego_poses[agent_ego_pose.token] = agent_ego_pose

            # Extract image frame from the agent's video log
            frame_path, frame_image = self.extract_image_frame(frame, timestamp, agent, overwrite)

            # Create image sample data for the extracted frame
            sample_data = ImageSampleData(timestamp, frame_path, DEFAULT_IMAGE_FRAME_FORMAT, frame_image, sample, agent_ego_pose, sensor)
            sample.image_sample_data[agent.id] = sample_data
            count += 1

        # Return number of extracted frame images for the sample
        return count


    def extract_image_frame(self, frame, timestamp, agent, overwrite = True):
        """
        Extracts a single image frame from the video log of an agent.
        """

        # Compose agent image frame folder and video log path
        agent_frame_folder = os.path.join(self.frames_folder, f"{agent.id:02d}")
        agent_video_log = self.log.video_feeds[agent.id]

        # Sanity checks
        assert os.path.exists(agent_video_log.path)  
        assert os.path.exists(agent_frame_folder)  

        # Get the agent's video log capture
        capture = agent_video_log.vidcap

        # Set the frame to capture as the current frame
        capture.set(1, frame)

        # Read current capture frame
        _, image = capture.read()
        assert image is not None

        # Save image frame
        save_path = os.path.join(agent_frame_folder, f"{agent.id:02d}_{frame:010d}_{timestamp.timestamp()}.{DEFAULT_IMAGE_FRAME_FORMAT.lower()}")
        if not os.path.exists(save_path) or overwrite:
            cv2.imwrite(save_path, image)

        # Return sampled frame path
        return save_path, image

    def extract_point_clouds(self, sample, overwrite = True):
        """
        Populates the given sample with lidar point cloud data extracted from the LiDAR log of each agent.
        """

        # Get sample frame number and timestamp
        frame = sample.frame_number
        timestamp = sample.timestamp

        # Extract point clouds for each agent
        count = 0
        for agent in self.base_scene.agents.values():

            # Get agent's LiDAR sensor
            if not agent.has_lidar_sensor():
                continue
            sensor = agent.calibrated_lidar

            # Get current agent's ego pose
            agent_ego_pose = self.log.vehicle_log.get_ego_pose(agent.id, frame)
            self.ego_poses[agent_ego_pose.token] = agent_ego_pose
            
            # Extract point cloud info from the agent's LiDAR log
            annotated_lidar_points = self.log.lidar_log.get_agent_points_at_frame(agent, frame)
            positions = [lidar_point.coordinates for lidar_point, _ in annotated_lidar_points]
            normals = [lidar_point.normal for lidar_point, _ in annotated_lidar_points]
            intensities = [lidar_point.intensity for lidar_point, _ in annotated_lidar_points]
            target_ids = [lidar_point_annotation.target_id for _, lidar_point_annotation in annotated_lidar_points]
            target_types = [lidar_point_annotation.target_type for _, lidar_point_annotation in annotated_lidar_points]

            # Create point cloud
            lidar_point_cloud = LidarPointCloud(positions, normals, intensities, target_ids, target_types)

            # Create point cloud blob file
            open3d_point_cloud = lidar_point_cloud.to_open3d()
            agent_points_folder = os.path.join(self.points_folder, f"{agent.id:02d}")
            point_cloud_path = os.path.join(agent_points_folder, f"{agent.id:02d}_{frame:010d}_{timestamp.timestamp()}.{DEFAULT_POINTS_CLOUD_FORMAT.lower()}")
            if not os.path.exists(point_cloud_path) or overwrite:
                o3d.io.write_point_cloud(point_cloud_path, open3d_point_cloud, format='auto', write_ascii=False, compressed=False, print_progress=False)

            # Create point cloud sample data for the LiDAR scan
            sample_data = PointCloudSampleData(timestamp, point_cloud_path, DEFAULT_POINTS_CLOUD_FORMAT, lidar_point_cloud, sample, agent_ego_pose, sensor)
            sample.point_cloud_sample_data[agent.id] = sample_data
            count += 1

        # Return number of point clouds generated for sample
        return count

    def extract_gnss_coordinates(self, sample, overwrite = True):
        """
        Populates the given sample with GNSS coordinates data extracted from the GNSS log of each agent.
        """

        # Get sample frame number and timestamp
        frame = sample.frame_number
        timestamp = sample.timestamp

        # Extract GNSS coordinates for each agent
        count = 0
        for agent in self.base_scene.agents.values():

            # Get agent's GNSS sensor
            if not agent.has_gnss_sensor():
                continue
            sensor = agent.calibrated_gnss

            # Get current agent's ego pose
            agent_ego_pose = self.log.vehicle_log.get_ego_pose(agent.id, frame)
            self.ego_poses[agent_ego_pose.token] = agent_ego_pose

            # Extract GNSS coordinates from the agent's GNSS log
            gnss_coordinates = self.log.gnss_log.get_agent_gnss_at_frame(agent, frame)

            # Create GNSS coordinates blob file
            agent_gnss_folder = os.path.join(self.gnss_folder, f"{agent.id:02d}")
            gnss_coordinates_path = os.path.join(agent_gnss_folder, f"{agent.id:02d}_{frame:010d}_{timestamp.timestamp()}.{DEFAULT_GNSS_COORDINATES_FORMAT.lower()}")
            if not os.path.exists(gnss_coordinates_path) or overwrite:
                np.save(gnss_coordinates_path, gnss_coordinates)

            # Create GNSS coordinates sample data
            sample_data = GNSSCoordinatesSampleData(timestamp, gnss_coordinates_path, DEFAULT_POINTS_CLOUD_FORMAT, gnss_coordinates, sample, agent_ego_pose, sensor)
            sample.gnss_coordinates_sample_data[agent.id] = sample_data
            count += 1

        # Return number of point clouds generated for sample
        return count

    ###################################################
    # ANNOTATION METHODS
    ###################################################

    def annotate(self, filter_empty_bboxes=False):
        """
        Annotates a scene with bounding-box information.
        """

        # Annotate moveable targets
        self.annotate_moveable_targets(filter_empty_bboxes)

        # Annotate infrastructural targets
        self.annotate_infrastructural_targets(filter_empty_bboxes)

        # Set scene state as annotate
        self.state = SceneState.ANNOTATED

    def annotate_moveable_targets(self, filter_empty_bboxes=False):
        """
        Annotates a scene with bounding-box information for moveable targets.
        """

        # Initialize moveable targets instances dictionary
        instances = dict()

        # Annotate loop
        count = 0
        count_attr = 0
        for sample in tqdm(self.samples):
            
            # Collect bounding boxes detected by each agent for the current sample
            sample_bboxes = self.collect_sample_moveable_bboxes(sample)

            # Create sample annotations from bounding boxes
            for target_id, bbox in sample_bboxes.items():

                # Fetch bounding-box target
                target = self.base_scene.targets[target_id]

                # Get or create instance for the target
                if target_id not in instances:
                    instance = Instance(target, self)
                    instances[target_id] = instance
                else:
                    instance = instances[target_id]

                # Compute annotation attributes
                attributes = self.compute_attributes(target, sample)

                # Compute visibility level
                visibility_level = self.compute_visibility_level(target, sample)

                # Create sample annotation
                annotation = SampleAnnotation(sample, bbox, instance, attributes, visibility_level)

                # Filter bbox annotations with no LiDAR points if need be
                if filter_empty_bboxes and annotation.num_lidar_points == 0:
                    continue

                # Add annotation to the sample
                sample.annotations.append(annotation)

                # Add sample annotation reference to the instance
                instance.annotations.append(annotation)

                # Update attributes counter
                count_attr += len(attributes)

            # Update annotations counter
            count += len(sample.annotations)

        # Save scene moveable targets instances
        self.instances = instances

        # Log annotating results
        logging.info(f"Added {count} bbox annotations with {count_attr} attributes")

    def annotate_infrastructural_targets(self, instances, filter_empty_bboxes=False):
        """
        Annotates a scene with bounding-box information for infrastructural targets.
        """

        # Initialize infrastructural targets instances dictionary
        instances = dict()

        # Annotate loop
        count = 0
        count_attr = 0
        for sample in tqdm(self.samples):
            
            # Collect bounding boxes detected by each agent for the current sample
            sample_bboxes = self.collect_sample_infrastructual_bboxes(sample)

            # Create sample annotations from bounding boxes
            invalid_boxes = 0
            for target_id, bbox in sample_bboxes.items():

                # Fetch bounding-box infrastructural target
                target = self.base_scene.infrastructures[target_id]

                # Get or create instance for the target
                if target_id not in instances:
                    instance = Instance(target, self)
                    instances[target_id] = instance
                else:
                    instance = instances[target_id]

                # Compute annotation attributes
                attributes = []

                # Compute visibility level
                visibility_level = self.compute_visibility_level(target, sample)

                # Create sample annotation
                try:
                    annotation = SampleAnnotation(sample, bbox, instance, attributes, visibility_level)
                except ValueError:
                    invalid_boxes += 1
                    continue

                # Filter bbox annotations with no LiDAR points if need be
                if filter_empty_bboxes and annotation.num_lidar_points == 0:
                    continue

                # Add annotation to the sample
                sample.annotations.append(annotation)

                # Add sample annotation reference to the instance
                instance.annotations.append(annotation)

                # Update attributes counter
                count_attr += len(attributes)

            # Update annotations counter
            count += len(sample.annotations)

        # Save scene infrastuctural targets instances
        self.infra_instances = instances

        # Log annotating results
        logging.info(f"Added {count} bbox infrastructural annotations with {count_attr} attributes ({invalid_boxes} invalid boxes)")

    def collect_sample_moveable_bboxes(self, sample):
        """
        Collects bounding box annotations for moveable targets detected from all agents 
        for a given sample. Since multiple agent may detect a bounding
        box for the same target, these bounding boxes are first collected
        in a dictionary of lists and then averaged.

        Parameters:
            sample (samples.Sample): A sample object.

        Returns:
             sample_bboxes ({int:bboxes.BBox}): A dictionary of BBoxes with target ids for keys.
        """

        # Get sample frame number
        frame = sample.frame_number

        # Initialize sample bounding boxes dictionary
        sample_bboxes = defaultdict(list)

        # Collect moveable targets sample bounding boxes from each agent
        for agent in self.base_scene.agents.values():
            target_bboxes = self.log.bbox_log.get_agent_bboxes_at_frame(agent, frame)
            for target_id, bbox in target_bboxes:
                sample_bboxes[target_id].append(bbox)

        # Return average bboxes for each target detected at sample
        return { target_id : BBox.average(bboxes) for target_id, bboxes in sample_bboxes.items()}

    def collect_sample_infrastructual_bboxes(self, sample):
        """
        Collects bounding box annotations for infrastructural targets detected from all agents 
        for a given sample. Since multiple agent may detect a bounding
        box for the same target, these bounding boxes are first collected
        in a dictionary of lists and then averaged.

        Parameters:
            sample (samples.Sample): A sample object.

        Returns:
             sample_bboxes ({int:bboxes.BBox}): A dictionary of BBoxes with target ids for keys.
        """

        # Get sample frame number
        frame = sample.frame_number

        # Initialize sample bounding boxes dictionary
        sample_bboxes = defaultdict(list)

        # Collect infrastructural targets sample bounding boxes from each agent
        for agent in self.base_scene.agents.values():
            target_bboxes = self.log.bbox_infra_log.get_agent_bboxes_at_frame(agent, frame)
            for target_id, bbox in target_bboxes:
                sample_bboxes[target_id].append(bbox)

        # Return average bboxes for each target detected at sample
        return { target_id : BBox.average(bboxes) for target_id, bboxes in sample_bboxes.items()}


    def compute_attributes(self, target, sample):

        # Initialize attribute list
        attributes = []

        # Get sample frame number
        frame = sample.frame_number

        # Compute attributes for vehicle targets
        if isinstance(target, Vehicle):
            
            # Get vehicle ego speed
            ego_speed = self.log.vehicle_log.get_ego_speed(target.id, frame)

            # Check if vehicle is moving or stationary
            if ego_speed.is_moving():
                moving_attribute = self.base_scene.dataset.attributes["vehicle.moving"]
                attributes.append(moving_attribute)
            else:
                stopped_attribute = self.base_scene.dataset.attributes["vehicle.stopped"]
                attributes.append(stopped_attribute)

        # Compute attributes for pedestrian targets
        if isinstance(target, Pedestrian):

            # Get pedestrian ego speed
            ego_speed = self.log.pedestrian_log.get_ego_speed(target.id, frame)

            # Check if pedestrian is moving or stationary
            if ego_speed.is_moving():
                moving_attribute = self.base_scene.dataset.attributes["pedestrian.moving"]
                attributes.append(moving_attribute)
            else:
                stopped_attribute = self.base_scene.dataset.attributes["pedestrian.stopped"]
                attributes.append(stopped_attribute)

        # Return list of computed attributes
        return attributes

    def compute_visibility_level(self, target, sample):
        return list(self.base_scene.dataset.visibility_levels.values())[0]

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"Scene(token='{self.token}', base_scene='{self.base_scene.id}', weather='{self.weather_cond_label}, light='{self.light_cond_label}')"

    def __repr__(self):
        return f"Scene(token='{self.token}', base_scene='{self.base_scene.id}', weather='{self.weather_cond_label}, light='{self.light_cond_label}')"