import os
import sys
import argparse
import logging
from lib import ConVexDataset

import numpy as np
from datetime import datetime
from lib.poses import EgoPose
from lib.sensors import CalibratedCamera
from lib.bboxes import BBox


###################################################
# CONSTANTS
###################################################

DEFAULT_OUTPUT_FOLDER_NAME = "output"                   # Default name of the output folder
DEFAULT_BASE_SCENES_DESCRIPTOR_NAME = "scene.json"      # Default name of the base scene descriptor file
DEFAULT_SAMPLING_FREQ_HZ = 5                            # Default sampling frequency (Hz)

###################################################
# LOGGER
###################################################
logging.getLogger().setLevel(logging.INFO)

###################################################
# ARGUMENT PARSE
###################################################
parser = argparse.ArgumentParser(description='Generate the ConVex dataset')
parser.add_argument('dataset_root', help='Folder to be used as root to generate the dataset.' , nargs='?', type=str, default=os.getcwd()) 
parser.add_argument('dataset_name', help='Name of the dataset to generate' , nargs='?', type=str, default="ConVex-1.0-test") 
parser.add_argument('-o', '--output_folder', help='Folder in which the dataset will be dumped.', type=str, default=DEFAULT_OUTPUT_FOLDER_NAME)
parser.add_argument('-s', '--sampling_freq', help='Scene sampling frequency (Hz).', type=int, default=DEFAULT_SAMPLING_FREQ_HZ)
parser.add_argument('-d','--descriptor_name', help='Name of the base scene descriptor files.', type=str, default=DEFAULT_BASE_SCENES_DESCRIPTOR_NAME)

if __name__ == "__main__":

    # Parse arguments
    args = parser.parse_args()

    # Create ConVex dataset
    dataset = ConVexDataset(args.sampling_freq, args.dataset_root, args.output_folder, args.descriptor_name, clear_cache=True)

    # # Sample dataset
    # dataset.sample()

    # # Annotate dataset
    # dataset.annotate()

    # # Dump dataset
    # dataset.dump(args.dataset_name)

    # sys.exit(0)
    # dataset.render_2d_boxes()


    # Load base_scenes
    # base_scenes = load_base_scenes(DEFAULT_BASE_SCENES_FOLDER)

    # # Sample base scenes
    # sample_base_scenes(base_scenes)




    # scene = base_scenes[0].scenes['sunny']['6AM']
    # print(scene.bbox_log.get_bboxes_at_frame(0, 15))
    # print(scene.bbox_log.get_bboxes_at_frame(1, 15))

    # corners = np.array([
    #     [0, 4, 0],
    #     [0, 0, 0],
    #     [-2, 0, 0],
    #     [-2, 4, 0],
    #     [0, 4, 2],
    #     [0, 0, 2],
    #     [-2, 0, 2],
    #     [-2, 4, 2],
    # ])

    # bbox = BBox(corners, 0, 0)
    # print(bbox)



    # camera_intrinsic = base_scenes[0].sensors['VEHICLE_FRONT_CAMERA_00'].camera_intrinsic
    # sensor_type = base_scenes[0].sensor_types['VEHICLE_FRONT_CAMERA']
    # ego_pose = EgoPose(0.0, np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    # camera = CalibratedCamera(sensor_type, base_scenes[0].agents[0], "Fuffa", np.array([1, 0, 0]), np.array([-np.pi/2, 0, 0]), camera_intrinsic)
    # print(ego_pose)
    # print(camera)




    # base_scene = base_scenes[0].scenes["sunny"]["6AM"]
    # base_scenes[0].sample(clear=True)
    # scene.annotate()
    # print(scenes[0].agents)
    # print(scenes[0].num_roadside_agents)

    # Dump dataset





    # # Check moving/stopped attributes
    # moving_cnt = 0
    # stopped_cnt = 0
    # min_ego_speed_tr = np.array([10000, 10000, 10000])
    # min_ego_speed_rot = np.array([10000, 10000, 10000])
    # min_tr_norm = np.linalg.norm(min_ego_speed_tr)
    # min_rot_norm = np.linalg.norm(min_ego_speed_rot)
    # for base_scene in dataset.base_scenes:
    #     for weather_label in base_scene.scenes:
    #         for scene in base_scene.scenes[weather_label].values():
    #             for sample in scene.samples:
    #                 for vehicle in base_scene.vehicles.values():
    #                     ego_speed = scene.log.vehicle_log.get_ego_speed(vehicle.id, sample.frame_number)
                        
    #                     ego_tr_norm = np.linalg.norm(ego_speed.translational_speed)
    #                     ego_rot_norm = np.linalg.norm(ego_speed.rotational_speed)

    #                     if ego_tr_norm < min_tr_norm:
    #                         min_ego_speed_tr = ego_speed.translational_speed
    #                         min_tr_norm = ego_tr_norm

    #                     if ego_rot_norm < min_rot_norm:
    #                         min_ego_speed_rot = ego_speed.rotational_speed
    #                         min_rot_norm = ego_rot_norm

    #                     if ego_speed.is_moving():
    #                         moving_cnt += 1
    #                     else:
    #                         stopped_cnt += 1
    # print(moving_cnt)
    # print(stopped_cnt)
    # print(min_ego_speed_tr)
    # print(min_ego_speed_rot)