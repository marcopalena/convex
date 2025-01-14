import os
import cv2
import logging
import shutil
import pickle
import datetime
import shortuuid
import numpy as np
import json
from enum import Enum
from shapely.geometry import MultiPoint, LineString, Point, box

from .scenes import BaseScene, BaseSceneIncompleteError
from .maps import Map
from .categories import Category, Attribute
from .visibility import VisibilityLevel
from .sensors import CalibratedCamera
from .utils import draw_3d_box, view_points, get_world_to_local_matrix, draw_2d_box
from .poses import EgoPose

###################################################
# LOGGER
###################################################
logging.getLogger().setLevel(logging.INFO)

###################################################
# CONSTANTS
###################################################
DEFAULT_CACHE_FILE_NAME = "dataset.pickle"
DEFAULT_DATASET_FILE_NAME = "dataset.json"
DEFAULT_BASE_SCENES_FOLDER_NAME = "scenes"
DEFAULT_CACHE_FOLDER_NAME = "cache"

###################################################
# UTILITY CLASSES
###################################################

class DatasetState(Enum):
    LOADED = 1
    SAMPLED = 2
    ANNOTATED = 3

###################################################
# CONVEX DATASET
###################################################

class ConVexDataset:
    """
    A class to represent the ConVex dataset.
    """

    def __init__(self, sampling_freq_hz : int, dataset_root : str, output_folder : str, descriptor_name : str, clear_cache=False, disable_cache=False, auto_write_cache=True):

        # Initialize dataset
        self.token = shortuuid.uuid()
        self.sampling_freq_hz = sampling_freq_hz
        self.dataset_root = dataset_root
        self.descriptor_name = descriptor_name

        # Initialize base scenes, output and cache folder paths
        self.base_scenes_folder_path = os.path.join(dataset_root, DEFAULT_BASE_SCENES_FOLDER_NAME)
        self.cache_folder_path = os.path.join(dataset_root, DEFAULT_CACHE_FOLDER_NAME)
        self.output_folder_path = os.path.join(dataset_root, output_folder)

        # Create output folder
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)

        # Clear cache if needed
        if clear_cache:
            self.clear_cache()

        # Parse dataset descriptor
        dataset_descriptor_path = os.path.join(self.dataset_root, DEFAULT_DATASET_FILE_NAME)
        with open(dataset_descriptor_path, 'r') as file:
            dataset_descr = json.load(file)

        # Load maps
        self.load_maps(dataset_descr['maps'])

        # Load categories
        self.load_categories(dataset_descr['categories'])

        # Load visibility levels
        self.load_visibility_levels(dataset_descr['visibility'])

        # Load attributes
        self.load_attributes(dataset_descr['attributes'])

        # Load base scenes
        self.load_base_scenes(dataset_descr['base_scenes'], disable_cache, auto_write_cache)

    ###################################################
    # DATASET GENERATION
    ###################################################

    def load_maps(self, maps_descr):

        # Initialize maps dict
        self.maps = dict()

        # Create maps
        for map_descr in maps_descr:
            name = map_descr['name']
            description = map_descr['description']
            mask = map_descr['mask']
            map = Map(name, description, mask)
            self.maps[name] = map

    def load_categories(self, categories_descr):

        # Initialize categories dict
        self.categories = dict()

        # Create categories
        for category_descr in categories_descr:
            name = category_descr['name']
            description = category_descr['description']
            type = category_descr['type']
            category = Category(name, description, type)
            self.categories[name] = category

    def load_visibility_levels(self, visibility_levels_descr):
        
        # Initialize visibility levels dict
        self.visibility_levels = dict()

        # Create visibility levels
        for visibility_level_descr in visibility_levels_descr:
            name = visibility_level_descr['name']
            description = visibility_level_descr['description']
            min_val = visibility_level_descr['min']
            max_val = visibility_level_descr['max']
            visibility_level = VisibilityLevel(name, description, min_val, max_val)
            self.visibility_levels[name] = visibility_level

    def load_attributes(self, attributes_descr):
        
        # Initialize attributes dict
        self.attributes = dict()

        # Create attributes
        for attribute_descr in attributes_descr:
            name = attribute_descr['name']
            description = attribute_descr['description']
            attribute = Attribute(name, description)
            self.attributes[name] = attribute

    def load_base_scenes(self, base_scenes_descr, disable_cache=False, auto_write_cache=True):

        # Load base scenes
        self.base_scenes = []
        self.sensor_types = {}
        for base_scene_descr in base_scenes_descr:
            base_scene_path = os.path.join(self.dataset_root, base_scene_descr["path"])
            descriptor_path = os.path.join(base_scene_path, self.descriptor_name)

            # If base scene descriptor file is present in the base scene directory, load the base scene
            if os.path.exists(descriptor_path):
                try:

                    # Read base scene descriptor
                    with open(descriptor_path, 'r') as file:
                        base_scene_descr = json.load(file)

                    # Get base scene id
                    base_scene_id = base_scene_descr["id"]

                    # Load base scene from cache if available, create new base scene otherwise
                    if not disable_cache and self.is_base_scene_cached(base_scene_id):
                        logging.info(f"Loading base scene from cache: {base_scene_path}")
                        base_scene = self.load_base_scene_from_cache(base_scene_id)
                    else:
                        logging.info(f"Loading base scene from descriptor: {base_scene_path}")
                        base_scene = BaseScene(descriptor_path, base_scene_descr, self)

                    # Add base scene to list
                    self.base_scenes.append(base_scene)

                    # Log base scene info
                    logging.info(base_scene.info())

                except BaseSceneIncompleteError:
                    logging.error(f"Incomplete base scene: {base_scene_path}")
                    pass
        
        # Set dataset state
        self.state = DatasetState.LOADED

        # Save base scenes to cache
        if auto_write_cache:
            self.cache_base_scenes()

    def sample(self, clear=False, overwrite=True, auto_write_cache=True):
        
        # Sample base scenes
        for base_scene in self.base_scenes:
            base_scene.sample(self.sampling_freq_hz, clear, overwrite)

        # Set dataset state
        self.state = DatasetState.SAMPLED

        # Save base scenes to cache
        if auto_write_cache:
            self.cache_base_scenes()

    def annotate(self, auto_write_cache=True, disable_cache=False, filter_empty_bboxes=False):

        # Annotate base scenes
        for base_scene in self.base_scenes:
            base_scene.annotate(disable_cache, filter_empty_bboxes)

        # Set dataset state
        self.state = DatasetState.SAMPLED

        # Save base scenes to cache
        if auto_write_cache:
            self.cache_base_scenes()


    def post_process_coords(self, corner_coords, imsize):
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)

            if isinstance(img_intersection, LineString) or isinstance(img_intersection, Point):
                return None

            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords]
            )

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y

    def render_2d_boxes(self):

        # Render 2D boxes for all base scenes
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    for sample in scene.samples:

                        # Get bounding boxes for the current sample 
                        bboxes = [(annotation.instance.target.id, annotation.bbox) for annotation in sample.annotations if annotation.num_lidar_points > 0]

                        for agent in base_scene.agents.values():

                            # Get current agent frame image
                            sample_data = sample.image_sample_data[agent.id]
                            frame_image = cv2.imread(sample_data.path, cv2.IMREAD_COLOR)

                            # Get agent ego pose
                            agent_pose = scene.log.vehicle_log.get_ego_pose(agent.id, sample.frame_number)

                            # Get agent's calibrated camera
                            camera = agent.calibrated_camera

                            # Draw bboxes
                            bboxes_image = frame_image
                            for target_id, bbox in bboxes:

                                # Skip bbox for the sensing agent]
                                if target_id == agent.id:
                                    continue

                                # corners = np.array([[10, 0, 0]])
                                # camera.translation = [3, 0, 0]
                                # camera.rotation = [0, 0, 0]
                                # camera.transform_matrix = get_world_to_local_matrix(camera.translation, camera.rotation)
                                # agent_pose = EgoPose(0.0, [30,  0,   0], [-np.pi, 0,  0])

                                # Transform bbox corners into camera coordinates
                                corners = bbox.corners
                                camera_corners = np.vstack([camera.world_to_camera(corner, agent_pose) for corner in corners]).T
                                # print(corners)
                                # print([camera.world_to_camera(corner, agent_pose) for corner in corners])
                                # print(agent_pose.translation)
                                # print(agent_pose.rotation)
                                # print(camera.translation)
                                # print(camera.rotation)

                                # Filter out corners that are not in front of the calibrated sensor
                                in_front = np.argwhere(camera_corners[0, :] > 0).flatten()
                                corners_3d = camera_corners[:, in_front]


                                # Perform 3D to 2D projection
                                corners_2d = view_points(corners_3d, camera.camera_intrinsic.calibration_matrix, True).T[:, :2]

                                # Compute intersection of the convex hull of the reprojected bbox corner and the image canvas
                                final_points = self.post_process_coords(corners_2d, (camera.camera_intrinsic.resolution_width_px, camera.camera_intrinsic.resolution_height_px))

                                # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                                if final_points is None:
                                    continue
                                else:
                                    min_x, min_y, max_x, max_y = final_points

                                # Draw 2D bbox
                                bboxes_image = draw_2d_box(frame_image, (int(max_x), int(min_y)), (int(min_x), int(max_y)))

                                # Draw bbox
                                # if corners_2d.shape[0] == 8:
                                #     bboxes_image = draw_3d_box(frame_image, corners_2d)

                                # break

                            # Write image with bboxes
                            output_path = "test.png"
                            frame = sample.frame_number
                            timestamp = sample.timestamp
                            rendered_data_folder = os.path.join(
                                self.output_folder_path, 
                                "rendered", 
                                base_scene.scenario, 
                                weather_label, 
                                scene.time_of_day_label
                            )
                            agent_rendered_folder = os.path.join(rendered_data_folder, f"{agent.id:02d}")
                            if not os.path.exists(agent_rendered_folder):
                                os.makedirs(agent_rendered_folder)
                            output_path = os.path.join(agent_rendered_folder, f"{agent.id:02d}_{frame:010d}_{timestamp.timestamp()}.png")
                            cv2.imwrite(output_path, bboxes_image)
                #             break
                #         break
                #     break
                # break


    ###################################################
    # DATASET CACHING
    ###################################################

    def clear_cache(self):
        logging.info("Clearing cache")
        if os.path.exists(self.cache_folder_path):
            shutil.rmtree(self.cache_folder_path)

    def is_base_scene_cached(self, base_scene_id):
        cache_file_path = os.path.join(self.cache_folder_path, base_scene_id)
        return os.path.exists(cache_file_path)

    def cache_base_scenes(self):

        # Create cache folder
        if not os.path.exists(self.cache_folder_path):
            os.makedirs(self.cache_folder_path)

        # Serialize base scenes
        for base_scene in self.base_scenes:
            cache_file_path = os.path.join(self.cache_folder_path, base_scene.id)
            with open(cache_file_path, "wb") as outfile:
                pickle.dump(base_scene, outfile)  

    def load_base_scene_from_cache(self, base_scene_id):
        cache_file_path = os.path.join(self.cache_folder_path, base_scene_id)
        with open(cache_file_path, "rb") as infile:
            return pickle.load(infile)
        
    ###################################################
    # DUMPING METHODS
    ###################################################

    def dump(self, name):
        """
        Dumps the ConVex dataset in JSON format.
        """

        # Create dataset dump folder
        dataset_dump_path = os.path.join(self.output_folder_path, name)
        if not os.path.exists(dataset_dump_path):
            os.makedirs(dataset_dump_path)

        # Dump sensor types
        self.dump_sensor_types(name)

        # Dump calibrated sensors
        self.dump_sensors(name)

        # Dump categories
        self.dump_categories(name)

        # Dump attributes
        self.dump_attributes(name)

        # Dump visibility levels
        self.dump_visibility_levels(name)

        # Dump maps
        self.dump_maps(name)

        # Dump scenes
        self.dump_scenes(name)

        # Dump logs
        self.dump_logs(name)

        # Dump samples
        self.dump_samples(name)

        # Dump sample datas
        self.dump_sample_datas(name)

        # Dump ego poses
        self.dump_ego_poses(name)

        # Dump instances
        self.dump_instances(name)

        # Dump sample annotations
        self.dump_sample_annotations(name)

    def dump_sensor_types(self, name):
        """
        Dumps sensor types to a JSON file.
        """

        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "sensor.json")

        # Collect sensor type records
        sensor_type_records = []
        for sensor_type in self.sensor_types.values():
            sensor_type_records.append({
                "token" : sensor_type.token,
                "channel": sensor_type.channel,
                "modality": sensor_type.modality_str
            })

        # Dump sensor type records
        with open(output_file_path, 'a') as output_file:
            json.dump(sensor_type_records, output_file, indent=2)

    def dump_sensors(self, name):
        """
        Dumps calibrated sensors to a JSON file.
        """

        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "calibrated_sensor.json")

        # Collect sensor records
        sensor_records = []
        for base_scene in self.base_scenes:
            for sensor in base_scene.sensors.values():

                camera_intrinsic = []
                if isinstance(sensor, CalibratedCamera):
                    camera_intrinsic = sensor.camera_intrinsic.calibration_matrix.tolist()

                sensor_records.append({
                    "token": sensor.token,
                    "sensor_token": sensor.sensor_type.token,
                    "translation": sensor.translation,
                    "rotation" : sensor.rotation_quaternion().tolist(),
                    "camera_intrinsic" : camera_intrinsic
                })

        # Dump sensor records
        with open(output_file_path, 'a') as output_file:
            json.dump(sensor_records, output_file, indent=2)


    def dump_categories(self, name):
        """
        Dumps categories to a JSON file.
        """

        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "category.json")

        # Collect category records
        category_records = []
        for index, category in enumerate(self.categories.values()):
            category_records.append({
                "token" : category.token,
                "name": category.name,
                "description": category.description,
                "index" : index
            })

        # Dump category records
        with open(output_file_path, 'w') as output_file:
            json.dump(category_records, output_file, indent=2)

    def dump_attributes(self, name):
        """
        Dumps attributes to a JSON file.
        """

        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "attribute.json")

        # Collect attribute records
        attribute_records = []
        for attribute in self.attributes.values():
            attribute_records.append({
                "token" : attribute.token,
                "name": attribute.name,
                "description": attribute.description
            })

        # Dump category records
        with open(output_file_path, 'w') as output_file:
            json.dump(attribute_records, output_file, indent=2)

    def dump_visibility_levels(self, name):
        """
        Dumps visibility levels to a JSON file.
        """

        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "visibility.json")

        # Collect visibility level records
        visibility_level_records = []
        for visibility_level in self.visibility_levels.values():
            visibility_level_records.append({
                "token" : visibility_level.token,
                "level": f"v{visibility_level.min}-{visibility_level.max}",
                "description": visibility_level.description
            })

        # Dump category records
        with open(output_file_path, 'w') as output_file:
            json.dump(visibility_level_records, output_file, indent=2)

    def dump_maps(self, name):
        """
        Dumps maps to a JSON file.
        """

        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "map.json")

        # Collect map level records
        map_records = []
        for map in self.maps.values():
            map_records.append({
                "token" : map.token,
                "log_tokens" : [log.token for log in map.logs],
                "category" : "semantic_prior",
                "filename": map.mask
            })

        # Dump map records
        with open(output_file_path, 'w') as output_file:
            json.dump(map_records, output_file, indent=2)   
    
    def dump_scenes(self, name):
        """
        Dumps scenes to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "scene.json")

        # Collect scene records
        scene_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    scene_records.append({
                        "token": scene.token,
                        "name": f"{scene.base_scene.id}.{scene.weather_cond_label}.{scene.time_of_day_label}",
                        "description": scene.base_scene.description,
                        "log_token": scene.log.token,
                        "nbr_samples": len(scene.samples),
                        "first_sample_token": scene.samples[0].token,
                        "last_sample_token": scene.samples[-1].token
                    })

        # Dump scene records
        with open(output_file_path, 'a') as output_file:
            json.dump(scene_records, output_file, indent=2)   

    def dump_logs(self, name):
        """
        Dumps logs to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "log.json")

        # Collect log records
        log_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    log_records.append({
                        "token": scene.log.token,
                        "logfile": scene.log.path,
                        "vehicle": "log",
                        "date_captured" : scene.log.timestamp.strftime('%Y-%m-%d')
                    })

        # Dump log records
        with open(output_file_path, 'a') as output_file:
            json.dump(log_records, output_file, indent=2)   

    def dump_samples(self, name):
        """
        Dumps samples to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "sample.json")

        # Collect sample records
        sample_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    prev_token = ""
                    for i, sample in enumerate(scene.samples):
                        prev_token = scene.samples[i-1].token if i > 0 else ""
                        next_token = scene.samples[i+1].token if i < len(scene.samples) - 1 else ""
                        sample_records.append({
                            "token": sample.token,
                            "timestamp": int(sample.timestamp.timestamp() * 1000),
                            "scene_token": sample.scene.token,
                            "next" : next_token,
                            "prev" : prev_token
                        })


        # Dump sample records
        with open(output_file_path, 'a') as output_file:
            json.dump(sample_records, output_file, indent=2) 

    def dump_sample_datas(self, name):
        """
        Dumps sample datas to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "sample_data.json")

        # Collect sample data records
        sample_data_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    for i, sample in enumerate(scene.samples):

                        # Collect image frame sample data records
                        for agent_id, image_sample_data in sample.image_sample_data.items():

                            # Fetch sample data from the same sensor that precedes this in time
                            prev_token = ""
                            for prev_sample in scene.samples[:i][::-1]:
                                if agent_id in prev_sample.image_sample_data:
                                    prev_token = prev_sample.image_sample_data[agent_id].token
                                    break

                            # Fetch sample data from the same sensor that follows this in time
                            next_token = ""
                            for next_sample in scene.samples[i+1:]:
                                if agent_id in next_sample.image_sample_data:
                                    next_token = next_sample.image_sample_data[agent_id].token
                                    break

                            # Add image frame sample data record
                            sample_data_records.append({
                                "token": image_sample_data.token,
                                "sample_token": sample.token,
                                "ego_pose_token": image_sample_data.ego_pose.token,
                                "calibrated_sensor_token": image_sample_data.sensor.token,
                                "filename": image_sample_data.path,
                                "fileformat": image_sample_data.format.lower(),
                                "width": image_sample_data.width,
                                "height": image_sample_data.height,
                                "timestamp": int(image_sample_data.timestamp.timestamp() * 1000),
                                "is_key_frame": True,
                                "next" : next_token,
                                "prev" : prev_token
                            })

                        # Collect point clouds sample data records
                        for agent_id, point_cloud_sample_data in sample.point_cloud_sample_data.items():

                            # Fetch sample data from the same sensor that precedes this in time
                            prev_token = ""
                            for prev_sample in scene.samples[:i][::-1]:
                                if agent_id in prev_sample.point_cloud_sample_data:
                                    prev_token = prev_sample.point_cloud_sample_data[agent_id].token
                                    break

                            # Fetch sample data from the same sensor that follows this in time
                            next_token = ""
                            for next_sample in scene.samples[i+1:]:
                                if agent_id in next_sample.point_cloud_sample_data:
                                    next_token = next_sample.point_cloud_sample_data[agent_id].token
                                    break

                            # Add point cloud sample data record
                            sample_data_records.append({
                                "token": point_cloud_sample_data.token,
                                "sample_token": sample.token,
                                "ego_pose_token": point_cloud_sample_data.ego_pose.token,
                                "calibrated_sensor_token": point_cloud_sample_data.sensor.token,
                                "filename": point_cloud_sample_data.path,
                                "fileformat": point_cloud_sample_data.format.lower(),
                                "width": 0,
                                "height": 0,
                                "timestamp": int(point_cloud_sample_data.timestamp.timestamp() * 1000),
                                "is_key_frame": True,
                                "next" : next_token,
                                "prev" : prev_token
                            })

                        # Collect GNSS coordinates sample data records
                        for agent_id, gnss_coord_sample_data in sample.gnss_coordinates_sample_data.items():

                            # Fetch sample data from the same sensor that precedes this in time
                            prev_token = ""
                            for prev_sample in scene.samples[:i][::-1]:
                                if agent_id in prev_sample.gnss_coordinates_sample_data:
                                    prev_token = prev_sample.gnss_coordinates_sample_data[agent_id].token
                                    break

                            # Fetch sample data from the same sensor that follows this in time
                            next_token = ""
                            for next_sample in scene.samples[i+1:]:
                                if agent_id in next_sample.gnss_coordinates_sample_data:
                                    next_token = next_sample.gnss_coordinates_sample_data[agent_id].token
                                    break

                            # Add GNSS coordinates sample data record
                            sample_data_records.append({
                                "token": gnss_coord_sample_data.token,
                                "sample_token": sample.token,
                                "ego_pose_token": gnss_coord_sample_data.ego_pose.token,
                                "calibrated_sensor_token": gnss_coord_sample_data.sensor.token,
                                "filename": gnss_coord_sample_data.path,
                                "fileformat": gnss_coord_sample_data.format.lower(),
                                "width": 0,
                                "height": 0,
                                "timestamp": int(gnss_coord_sample_data.timestamp.timestamp() * 1000),
                                "is_key_frame": True,
                                "next" : next_token,
                                "prev" : prev_token
                            })

        # Dump sample data records
        with open(output_file_path, 'a') as output_file:
            json.dump(sample_data_records, output_file, indent=2) 

    def dump_ego_poses(self, name):
        """
        Dumps ego poses to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "ego_pose.json")

        # Collect ego pose records
        ego_pose_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    for ego_pose in scene.ego_poses.values():

                        timestamp = ego_pose.timestamp - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + scene.timestamp
                        ego_pose_records.append({
                            "token": ego_pose.token,
                            "timestamp": int(timestamp.timestamp() * 1000),
                            "translation": ego_pose.translation.tolist(),
                            "rotation" : ego_pose.rotation_quaternion().tolist(),
                        })

        # Dump sample records
        with open(output_file_path, 'a') as output_file:
            json.dump(ego_pose_records, output_file, indent=2) 

    def dump_instances(self, name):
        """
        Dumps instances to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "instance.json")

        # Collect instance records
        instance_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    for instance in scene.instances.values():
                        instance_records.append({
                            "token": instance.token,
                            "category_token": instance.target.category.token,
                            "nbr_annotations": len(instance.annotations),
                            "first_annotation_token" : instance.annotations[0].token,
                            "last_annotation_token" : instance.annotations[-1].token
                        })
                    for instance in scene.infra_instances.values():
                        instance_records.append({
                            "token": instance.token,
                            "category_token": instance.target.category.token,
                            "nbr_annotations": len(instance.annotations),
                            "first_annotation_token" : instance.annotations[0].token,
                            "last_annotation_token" : instance.annotations[-1].token
                        })

        # Dump sample records
        with open(output_file_path, 'a') as output_file:
            json.dump(instance_records, output_file, indent=2) 
            
    def dump_sample_annotations(self, name):
        """
        Dumps sample annotations to a JSON file.
        """
        
        # Compose output file path
        output_file_path = os.path.join(self.output_folder_path, name, "sample_annotation.json")

        # Collect sample annotation records
        sample_annotation_records = []
        for base_scene in self.base_scenes:
            for weather_label in base_scene.scenes:
                for scene in base_scene.scenes[weather_label].values():
                    for i, sample in enumerate(scene.samples):
                        for annotation in sample.annotations:

                            # Get instance token
                            instance_token = annotation.instance.token
                            
                            # Fetch sample annotation from the same object instance that precedes this in time
                            prev_token = ""
                            for prev_sample in scene.samples[:i][::-1]:
                                for prev_annotation in prev_sample.annotations:
                                    if prev_annotation.instance.token == instance_token:
                                        prev_token = prev_annotation.token
                                        break

                            # Fetch sample annotation from the same object instance that follows this in time
                            next_token = ""
                            for next_sample in scene.samples[i+1:]:
                                for next_annotation in next_sample.annotations:
                                    if next_annotation.instance.token == instance_token:
                                        next_token = next_annotation.token
                                        break
                            
                            # Add sample annotation record
                            sample_annotation_records.append({
                                "token": annotation.token,
                                "sample_token": annotation.sample.token,
                                "instance_token": annotation.instance.token,
                                "attribute_tokens" : [attribute.token for attribute in annotation.attributes],
                                "visibility_token" : annotation.visibility_level.token,
                                "translation": annotation.bbox.center.tolist(),
                                "size": annotation.bbox.dimensions.tolist(),
                                "rotation": annotation.bbox.orientation.tolist(),
                                "nbr_lidar_points": annotation.num_lidar_points,
                                "nbr_radar_points": 0,
                                "next": next_token,
                                "prev": prev_token 
                            })

        # Dump sample records
        with open(output_file_path, 'a') as output_file:
            json.dump(sample_annotation_records, output_file, indent=2) 

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"ConVexDataset(token='{self.token}', sampling_freq_hz={self.sampling_freq_hz}, num_base_scenes={len(self.base_scenes)})"

    def __repr__(self):
        return f"ConVexDataset(token='{self.token}', sampling_freq_hz={self.sampling_freq_hz}, num_base_scenes={len(self.base_scenes)})"
