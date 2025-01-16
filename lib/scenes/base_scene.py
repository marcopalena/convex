import json
import sys
import os
import re
import shortuuid
import numpy as np
import logging
import datetime
from collections import defaultdict
from enum import Enum

from ..targets import Vehicle, Pedestrian, Infrastructure
from ..utils import VehicleDimensions
from ..agent import VehicleAgent, RoadsideAgent
from ..sensors import SensorModality, SensorType, CalibratedCamera, CalibratedLiDAR, CalibratedGNSS, CameraIntrinsic, LidarParameters
from .scene import Scene, SceneState
from ..logs import SceneLog

from ..poses import EgoPose
from ..utils import translation_to_transform, rotation_to_transform

###################################################
# CONSTANTS
###################################################

WHEATER_COND_LABELS = ["sunny", "rainy", "snowy", "cloudy"]
TIME_OF_DAY_LABELS = ["6AM", "10PM", "12PM"]

BASE_SCENE_INFO_STRING = '''\
Base Scene ({})
    Token: {}
    Path: {}
    Scenario: {}
    Description: {}
    Vehicles: {}
    Pedestrians: {}
    Agents: {}
      CAVs: {}
      RSUs: {}
    Sensors Types: {}
    Calibrated Sensors: {}'''

DEFAULT_IGNORE_SCENE_DICT = {
    "sunny" : {
        "6AM" : False,
        "10PM" : False,
        "12PM" : False
    },
    "cloudy" : {
        "6AM" : False,
        "10PM" : False,
        "12PM" : False
    },
    "snowy" : {
        "6AM" : False,
        "10PM" : False,
        "12PM" : False
    },
    "rainy" : {
        "6AM" : False,
        "10PM" : False,
        "12PM" : False
    }
}

###################################################
# LOGGER
###################################################
logging.getLogger().setLevel(logging.INFO)

###################################################
# UTILITY CLASSES
###################################################

class BaseSceneState(Enum):
    LOADED = 1
    SAMPLED = 2
    ANNOTATED = 3

class BaseSceneIncompleteError(Exception):
    pass

###################################################
# BASE SCENE
###################################################

class BaseScene:
    """
    A class to represent a base scene setup in SCANeR Studio.
    """

    def __init__(self, descriptor_path, base_scene_descr, dataset):
        """
        Create an empty base scene from a folder path. Expects to find a "scene.json" base scene
        descriptor file in the folder.

        Parameters:
            descriptor_path (str): Path of the base scene descriptor file.
        """

        # Initialize base scene
        self.token = shortuuid.uuid()
        self.descriptor_path = descriptor_path
        self.dataset = dataset
        self.path = os.path.dirname(descriptor_path)
        self.ignore_scenes = dict(DEFAULT_IGNORE_SCENE_DICT)

        # Initialize base scene
        self.id = base_scene_descr['id']
        self.scenario = base_scene_descr['scenario']
        self.description = base_scene_descr['description']
        self.duration_ms = base_scene_descr['duration_ms']
        self.timestamp = datetime.datetime.fromtimestamp(base_scene_descr['timestamp'])
        self.scaner_sampling_freq_hz = base_scene_descr['scaner_sampling_freq_hz']

        # Fetch map
        map_name = base_scene_descr['map']
        self.map = self.dataset.maps[map_name]

        # Sanity check on folder structure
        if not self.check_folder_structure():
            logging.warning(f"Incomplete base scene: {self.path}")
            raise BaseSceneIncompleteError()
        
        # Load base scene
        self.load_base_scene(base_scene_descr)

        # Load scenes
        self.load_scenes(base_scene_descr['scenes'])

        # Set base scene state
        self.state = BaseSceneState.LOADED

    ###################################################
    # SANITY CHECKS METHODS
    ###################################################

    def check_folder_structure(self):
        """
        Checks if all first-level and second-level subfolders exist for a base scene.

        Returns:
            (bool): True if all subfolders at both levels exist, False otherwise.
        """
        for weather_cond_label in WHEATER_COND_LABELS:
            weather_cond_path = os.path.join(self.path, weather_cond_label)
            if not os.path.isdir(weather_cond_path):
                return False
            for time_of_day_label in TIME_OF_DAY_LABELS:
                time_of_day_path = os.path.join(weather_cond_path, time_of_day_label)
                if not os.path.isdir(time_of_day_path):
                    return False
        return True
    
    ###################################################
    # BASE SCENE LOADING METHODS
    ###################################################

    def load_base_scene(self, base_scene_descr):
        """
        Loads base scene info from the descriptor file.

        Parameters:
            base_scene_descr (dict): Dictionary describing the base scene read from the descriptor file.
        """

        # # Parse ignore scenes section
        # if 'ignore_scenes' in base_scene_descr:
        #     self.parse_ignore_scenes(base_scene_descr['ignore_scenes'])

        # Parse targets
        self.parse_targets(base_scene_descr['targets'])

        # Parse agents
        self.parse_agents(base_scene_descr['agents'])

        # Parse sensors
        self.parse_sensors(base_scene_descr['sensors'])

    # def parse_ignore_scenes(self, ignore_scenes_descr):    
    #     """
    #     Parses ignore scenes section from the descriptor file.

    #     Parameters:
    #         ignore_scenes_descr (list): List reporting the scenes to ignore from the descriptor file.
    #     """    
    #     for scene_to_ignore in ignore_scenes_descr:
    #         tokens = scene_to_ignore.split(".")
    #         if len(tokens) == 1 and tokens[0] in WHEATER_COND_LABELS:
    #             self.ignore_scenes[tokens[0]] = {time_of_day_label : True for time_of_day_label in TIME_OF_DAY_LABELS}
    #         elif len(tokens) == 2 and tokens[0] in WHEATER_COND_LABELS and tokens[1] in TIME_OF_DAY_LABELS:
    #             self.ignore_scenes[tokens[0]][tokens[1]] = True

    def parse_targets(self, targets_descr):
        """
        Parses target objects info from the descriptor file.

        Parameters:
            targets_descr (dict): Dictionary describing the targets populating the base scene.
        """

        # Initialize targets dictionary
        self.targets = {}

        # Parse vehicle targets
        self.parse_vehicle_targets(targets_descr['vehicles'])

        # Parse pedestrian targets
        self.parse_pedestrian_targets(targets_descr['pedestrians'])

        # Parse road infrastructure targets
        self.parse_infrastructure_targets(targets_descr['infrastructure'])

    def parse_vehicle_targets(self, vehicles_descr):
        """
        Parses vehicles targets info from the descriptor file.

        Parameters:
            vehicles_descr (dict): Dictionary describing the vehicle targets populating the base scene.
        """

        # Initialize vehicle targets dictionary
        self.vehicles = {}

        # Parse and collect target vehicles info into a dict
        for vehicle_descr in vehicles_descr:

            # Collect vehicle target info
            vehicle_id = vehicle_descr["id"]
            vehicle_name = vehicle_descr["name"]
            vehicle_scaner_type = vehicle_descr["scaner_type"]
            vehicle_dimensions = vehicle_descr["dimensions"]

            # Fetch category
            vehicle_category_name = vehicle_descr["category"]
            vehicle_category = self.dataset.categories[vehicle_category_name]

            # Create vehicle
            vehicle = Vehicle(vehicle_id, vehicle_name, vehicle_scaner_type, vehicle_category, vehicle_dimensions)

            # Add vehicle to vehicles and targets dicts
            self.vehicles[vehicle_id] = vehicle
            self.targets[vehicle_id] = vehicle

    # def parse_vehicle_dimensions(self, vehicle_dimensions_descr):
    #     """
    #     Parses vehicle dimensions info from the descriptor file.

    #     Parameters:
    #         vehicle_dimensions_descr (dict): Dictionary describing the dimensions of a vehicle.

    #     Returns:
    #         (VehicleDimensions): object representing the parse vehicle dimensions.
    #     """

    #     # Collect vehicle dimensions info
    #     length_mm = vehicle_dimensions_descr["length_mm"]
    #     width_mm = vehicle_dimensions_descr["width_mm"] 
    #     height_mm = vehicle_dimensions_descr["height_mm"]
    #     weight_kg = vehicle_dimensions_descr["weight_kg"]
    #     wheel_base_mm = vehicle_dimensions_descr["wheel_base_mm"]
    #     # front_track_mm = vehicle_dimensions_descr["front_track_mm"]
    #     # rear_track_mm = vehicle_dimensions_descr["rear_track_mm"]
    #     # rear_overhang_mm = vehicle_dimensions_descr["rear_overhang_mm"]
        
    #     # TODO: make it just a dict

    #     # Return vehicle dimension object
    #     return VehicleDimensions(length_mm, width_mm, height_mm, weight_kg, wheel_base_mm, 0, 0, 0)

    def parse_pedestrian_targets(self, pedestrians_descr):
        """
        Parses pedestrian targets info from the descriptor file.

        Parameters:
            pedestrians_descr (dict): Dictionary describing the pedestrian targets populating the base scene.
        """

        # Initialize pedestrian targets dictionary
        self.pedestrians = {}

        # Parse and collect pedestrians info into a dict
        for pedestrian_descr in pedestrians_descr:

            # Collect pedestrian target info
            pedestrian_id = pedestrian_descr["id"]
            pedestrian_name = pedestrian_descr["name"]
            pedestrian_scaner_type = pedestrian_descr["scaner_type"]
            pedestrian_visibility =  True if pedestrian_descr["visibility"] is True else False

            # Fetch category
            pedestrian_category_name = pedestrian_descr["category"]
            pedestrian_category = self.dataset.categories[pedestrian_category_name]

            # Create pedestrian
            pedestrian = Pedestrian(pedestrian_id, pedestrian_name, pedestrian_scaner_type, pedestrian_category, pedestrian_visibility)

            # Add pedestrian to pedestrians and targets dicts
            self.pedestrians[pedestrian_id] = pedestrian
            self.targets[pedestrian_id] = pedestrian
    
    def parse_infrastructure_targets(self, infrastructures_descr):
        """
        Parses infrastructure targets info from the descriptor file.

        Parameters:
            infrastructures_descr (dict): Dictionary describing the infrastructure targets populating the base scene.
        """

        # Initialize infrastructure targets dictionary
        self.infrastructures = {}

        # Parse and collect infrastructures info into a dict
        for infrastructure_descr in infrastructures_descr:

            # Collect infrastructure target info
            infrastructure_id = infrastructure_descr["id"]
            infrastructure_name = infrastructure_descr["name"]
            infrastructure_scaner_type = infrastructure_descr["scaner_type"]

            # Fetch category
            infrastructure_category_name = infrastructure_descr["category"]
            infrastructure_category = self.dataset.categories[infrastructure_category_name]

            # Create infrastructure
            infrastructure = Infrastructure(infrastructure_id, infrastructure_name, infrastructure_scaner_type, infrastructure_category)

            # Add infrastructure to infrastructures dict
            self.infrastructures[infrastructure_id] = infrastructure

    def parse_agents(self, agents_descr):
        """
        Parses agents info from the descriptor file.

        Parameters:
            agents_descr (dict): Dictionary describing the agents in the base scene.
        """

        # Initialize agents dictionary
        self.agents = {}

        # Parse and collect vehicle agents info
        for vehicle_agent_id in agents_descr["vehicle"]:
            vehicle = self.vehicles[vehicle_agent_id]
            self.agents[vehicle_agent_id] = VehicleAgent(vehicle_agent_id, vehicle)

        # Parse and collect roadside agents info
        for roadside_agent_id in agents_descr["roadside"]:
            pedestrian = self.pedestrians[roadside_agent_id]
            pedestrian.visible = False
            self.agents[roadside_agent_id] = RoadsideAgent(roadside_agent_id, pedestrian)

    def parse_sensors(self, sensors_descr):
        """
        Parses sensors info from the descriptor file.

        Parameters:
            sensors_descr (dict): Dictionary describing the sensors in the base scene.
        """

        # Initialize calibrated sensors dictionary
        self.sensors = {}

        # Parse and collect sensor types and calibrated sensors
        for sensor_descr in sensors_descr:

            # Parse sensor type
            sensor_type = self.parse_sensor_type(sensor_descr)

            # Get sensor's agent
            agent_id = sensor_descr["agent_id"]
            agent = self.agents[agent_id]

            # Compose sensor's name
            name = f"{sensor_type.channel}_{agent_id:02d}"

            # Parse calibrated sensor info
            translation = sensor_descr["translation"]
            rotation = sensor_descr["rotation"]
            if sensor_type.modality == SensorModality.CAMERA:
                camera_intrinsic = self.parse_camera_intrinsic(sensor_descr)
                calibrated_sensor = CalibratedCamera(sensor_type, agent, name, translation, rotation, camera_intrinsic)

                # sys.exit(0)

                agent.calibrated_camera = calibrated_sensor
            elif sensor_type.modality == SensorModality.LIDAR:
                lidar_parameters = self.parse_lidar_parameters(sensor_descr)
                calibrated_sensor =  CalibratedLiDAR(sensor_type, agent, name, translation, rotation, lidar_parameters)
                agent.calibrated_lidar = calibrated_sensor
            elif sensor_type.modality == SensorModality.GNSS:
                calibrated_sensor =  CalibratedGNSS(sensor_type, agent, name, translation, rotation)
                agent.calibrated_gnss = calibrated_sensor

            # Add calibrated sensor to scene dict
            self.sensors[name] = calibrated_sensor

    def parse_sensor_type(self, sensor_descr):
        """
        Parses the sensor type info from the descriptor file.

        Parameters:
            sensor_descr (dict): Dictionary describing the sensor.

        Returns:
            (SensorType): object representing the sensor type.
        """

        # Get channel and modality
        channel = sensor_descr["channel"]
        modality_str = sensor_descr["modality"]
        if sensor_descr["modality"] == "camera":
            modality = SensorModality.CAMERA
        elif sensor_descr["modality"] == "lidar":
            modality = SensorModality.LIDAR
        elif sensor_descr["modality"] == "gnss":
            modality = SensorModality.GNSS

        # Create new or get existing sensor type
        if channel not in self.dataset.sensor_types:
            sensor_type =  SensorType(channel, modality, modality_str)
            self.dataset.sensor_types[channel] = sensor_type
        else:
            sensor_type = self.dataset.sensor_types[channel]

        # Return sensor type
        return sensor_type

    def parse_camera_intrinsic(self, sensor_descr):
        """
        Parses the camera intrinsic info from the descriptor file.

        Parameters:
            sensor_descr (dict): Dictionary describing the sensor.

        Returns:
            (CameraIntrinsic): object representing the camera intrinsic parameters.
        """

        # Get intrinsic parameters
        resolution_width_px = sensor_descr["resolution_width_px"]
        resolution_height_px = sensor_descr["resolution_height_px"]
        focal_length_mm = sensor_descr["focal_length_mm"]
        sensor_width_mm = sensor_descr["sensor_width_mm"]
        sensor_height_mm = sensor_descr["sensor_height_mm"]
        horizontal_fov_deg = sensor_descr["horizontal_fov_deg"]
        vertical_fov_deg = sensor_descr["vertical_fov_deg"]

        # Create camera intrinsic
        camera_intrinsic = CameraIntrinsic(
            resolution_width_px, resolution_height_px,
            focal_length_mm, 
            sensor_width_mm, sensor_height_mm, 
            horizontal_fov_deg, vertical_fov_deg
        )

        # Return camera intrinsic
        return camera_intrinsic

    def parse_lidar_parameters(self, sensor_descr):
        """
        Parses the LiDAR sensor parameters info from the descriptor file.

        Parameters:
            sensor_descr (dict): Dictionary describing the sensor.

        Returns:
            (LidarParameters): object representing the LiDAR sensor parameters.
        """

        # Get LiDAR parameters
        max_range_m = sensor_descr["maxrange_m"]
        total_fov_deg = sensor_descr["total_fov_deg"]
        speed_tr_per_min = sensor_descr["speed_trpermin"]
        resolution_deg = sensor_descr["resolution_deg"]
        vertical_ray_num = sensor_descr["verticalraynum"]
        horizontal_ray_num = sensor_descr["horizontalraynum"]
        elev_min_angle_deg = sensor_descr["elev_minangle_deg"]
        elev_max_angle_deg = sensor_descr["elev_maxangle_deg"]

        # Create LiDAR parameters
        lidar_parameters = LidarParameters(
            max_range_m, 
            total_fov_deg, 
            speed_tr_per_min, 
            resolution_deg, 
            vertical_ray_num, horizontal_ray_num, 
            elev_min_angle_deg, elev_max_angle_deg
        )

        # Return LiDAR parameters
        return lidar_parameters

    ###################################################
    # SCENES LOADING
    ###################################################

    def load_scenes(self, scenes_descr):
        """
        Loads scenes derived from the current base scene for different variations of
        weather condition and time of day.
        """

        # Initialize scenes dictionary
        self.scenes = {}
        for weather_cond_label in WHEATER_COND_LABELS:
            self.scenes[weather_cond_label] = defaultdict(dict)

        # Load scenes
        for scene_descr in scenes_descr.values():

            # Get scene info
            weather_cond_label = scene_descr["weather"]
            time_of_day_label = scene_descr["time_of_day"]
            log_path = scene_descr["log"]
            timestamp = datetime.datetime.fromtimestamp(scene_descr["timestamp"])

            # Log scene loading
            logging.info(f"Loading scene: {self.id}/{weather_cond_label}/{time_of_day_label}")

            # Load scene
            scene_log_path = os.path.join(self.path, log_path)
            scene = Scene(scene_log_path, self, weather_cond_label, time_of_day_label, timestamp)
            self.scenes[weather_cond_label][time_of_day_label] = scene

            # Add scene log to map
            self.map.logs.append(scene.log)

            # TODO: relax
            break

    def fetch_timestamp(self, weather_cond_label, time_of_day_label):
        scene_video_feeds_path = os.path.join(self.path, weather_cond_label, time_of_day_label, "raw", "feeds")
        video_feed_file = list(os.walk(scene_video_feeds_path))[0][2][0]
        res = re.match(".*_([0-9]+)\.avi$", video_feed_file)
        if res is not None:
            return res.group(1)
        return None
    
    def fetch_duration(self, weather_cond_label, time_of_day_label):

        from ..logs import VideoLog

        scene_video_feeds_path = os.path.join(self.path, weather_cond_label, time_of_day_label, "raw", "feeds")
        for root, _, files in os.walk(scene_video_feeds_path):
            for file in files:
                video_log = VideoLog(os.path.join(root, file), None)
                print(video_log.num_frames)

    ###################################################
    # PROPERTIES
    ###################################################
    
    @property
    def num_vehicles(self):
        return len(self.vehicles)
    
    @property
    def num_pedestrians(self):
        return len(self.pedestrians)

    @property
    def num_agents(self):
        return len(self.agents)
    
    @property
    def num_vehicle_agents(self):
        return len([agent for agent in self.agents.values() if isinstance(agent, VehicleAgent)])

    @property
    def num_roadside_agents(self):
        return len([agent for agent in self.agents.values() if isinstance(agent, RoadsideAgent)])
    
    @property
    def num_sensor_types(self):
        return len(self.dataset.sensor_types)
    
    @property
    def num_sensors(self):
        return len(self.sensors)
    
    @property
    def vehicle_agents(self):
        return [agent for agent in self.agents.values() if isinstance(agent, VehicleAgent)]
    
    @property
    def roadside_agents(self):
        return [agent for agent in self.agents.values() if isinstance(agent, RoadsideAgent)]
    
    ###################################################
    # SAMPLING METHODS
    ###################################################

    def compute_sampling_window(self):
        """
        Computes a sampling window common to all scenes.

        Returns:
            start_frame (int): Number of the frame from which to start sampling
            end_frame (int): Number of the frame up to which to stop sampling
        """

        # Compute sampling window
        start_frames = []
        end_frames = []
        for weather_cond_label in self.scenes:
            for time_of_day_label in self.scenes[weather_cond_label]:
                scene = self.scenes[weather_cond_label][time_of_day_label]
                start_frames.append(scene.start_frame)
                end_frames.append(scene.end_frame)
        start_frame = max(start_frames)
        end_frame = min(end_frames)

        # Return sampling window boundaries
        return start_frame, end_frame

    def sample(self, sampling_freq_hz, clear = False, overwrite=True, disable_cache = False):
        """
        Sample each scene derived from the base scene.

        Parameters:
            sampling_freq_hz (float): Sampling frequency in Hz.
            clear (bool): If True clears sampled data before sampling.
            overwrite (bool): If True overwrites previously sampled data.
        """

        # Compute sampling window
        start_frame, end_frame = self.compute_sampling_window()

        # Iterate over all scenes
        for weather_cond_label in self.scenes:
            for time_of_day_label in self.scenes[weather_cond_label]:

                # Get scene
                scene = self.scenes[weather_cond_label][time_of_day_label]

                # Do not resample scenes loaded from cache
                if not disable_cache and not scene.state == SceneState.LOADED:
                    logging.info(f"Skipping sampling of already sampled scene: {self.id}/{weather_cond_label}/{time_of_day_label} (start={start_frame}, end={end_frame}, num_samples={scene.num_samples})")
                    continue

                # Log sampling scene
                logging.info(f"Sampling scene: {self.id}/{weather_cond_label}/{time_of_day_label} (start={start_frame}, end={end_frame})")

                # Sample scene
                scene.sample(sampling_freq_hz, start_frame, end_frame, clear, overwrite)

        # Set base scene state
        self.state = BaseSceneState.SAMPLED

    ###################################################
    # ANNOTATING METHODS
    ###################################################

    def annotate(self, disable_cache=False, filter_empty_bboxes=False):

       # Iterate over all scenes
        for weather_cond_label in self.scenes:
            for time_of_day_label in self.scenes[weather_cond_label]:

                # Get scene
                scene = self.scenes[weather_cond_label][time_of_day_label]

                # Do not annotate again scenes loaded from cache
                # if not disable_cache and scene.state == SceneState.ANNOTATED:
                #     logging.info(f"Skipping annotating of already annotated scene: {self.id}/{weather_cond_label}/{time_of_day_label} (num_annotations={scene.num_annotations})")
                #     continue

                # Log annotating scene
                logging.info(f"Annotating scene: {self.id}/{weather_cond_label}/{time_of_day_label}")

                # Annotate scene
                scene.annotate(filter_empty_bboxes)

        # Set base scene state
        self.state = BaseSceneState.ANNOTATED
            
    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"BaseScene(token='{self.token}', id='{self.id}', description='{self.description}', state={self.state})"

    def __repr__(self):
        return f"BaseScene(token='{self.token}', id='{self.id}', description='{self.description}', state={self.state})"
    
    ###################################################
    # UTILITY METHODS
    ###################################################

    def info(self):
        """
        Returns a string summarizing the information about a base scene.
        """
        return BASE_SCENE_INFO_STRING.format(
            self.id, 
            self.token, 
            self.path,
            self.scenario, 
            self.description, 
            self.num_vehicles, 
            self.num_pedestrians, 
            self.num_agents, 
            self.num_vehicle_agents, 
            self.num_roadside_agents,
            self.num_sensor_types,
            self.num_sensors
        )

    def dump(self):
            """
            Dump scene in nuScenes format.
            """

            # Collect JSON data
            data = {}
            data['token'] = self.token
            data['id'] = self.id
            data['description'] = self.description
            data['token'] = self.token

            # Dump as JSON