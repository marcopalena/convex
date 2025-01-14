import os
import re
import shortuuid

from .gnss_log import GNSSLog
from .bbox_log import BBoxLog
from .vehicle_log import VehicleLog
from .pedestrian_log import PedestrianLog
from .lidar_log import LiDARLog
from .log import VideoLog

###################################################
# CONSTANTS
###################################################

DEFAULT_RAW_DATA_FOLDER = "raw"                         # Default name of the raw data folder
DEFAULT_RAW_VIDEO_FEEDS_FOLDER = "feeds"                # Default name of the raw video feeds folder
DEFAULT_RAW_DATA_LOGS_FOLDER = "data"                   # Default name of the raw data logs folder

DEFAULT_RAW_VEHICLES_LOG_NAME = "vehData"               # Default name of the raw vehicle data log
DEFAULT_RAW_PEDESTRIAN_LOG_NAME = "pedesData"           # Default name of the raw vehicle data log
DEFAULT_RAW_BBOX_MOV_LOG_NAME = "bboxData_MovOnly"      # Default name of the raw bounding box data log
DEFAULT_RAW_BBOX_INFRA_LOG_NAME = "bboxData_infraOnly"  # Default name of the raw bounding box data log
DEFAULT_RAW_LIDAR_LOG_NAME = "lidarData"                # Default name of the raw LiDAR data log
DEFAULT_RAW_GNSS_LOG_NAME = "gpsData"                   # Default name of the raw GNSS data log
DEFAULT_RAW_VIDEO_EXTENSION = "avi"                     # Default extensions of the raw video feed files
DEFAULT_RAW_LOG_EXTENSION = "csv"                       # Default extensions of the raw data log files

###################################################
# REGEXPS
###################################################

VEHICLE_AGENT_VIDEO_FEED_REGEXP = ".*_Veh([0-9]+)_"
ROADSIDE_AGENT_VIDEO_FEED_REGEXP = ".*_Ped([0-9]+)_"

###################################################
# SCENE LOG
###################################################

class SceneLog:
    """
    A class to represent the log of a scene, the log is the file or collection of
    files that are recorder from SCANeR Studio during a simulation.
    """
        
    def __init__(self, path, timestamp, scene):

        # Initialize scene log
        self.token = shortuuid.uuid()
        self.path = path
        self.timestamp = timestamp
        self.scene = scene

        # Set up raw data folder paths
        self.raw_data_folder = os.path.join(self.path, DEFAULT_RAW_DATA_FOLDER)
        self.video_feeds_folder = os.path.join(self.raw_data_folder, DEFAULT_RAW_VIDEO_FEEDS_FOLDER)
        self.data_logs_folder = os.path.join(self.raw_data_folder, DEFAULT_RAW_DATA_LOGS_FOLDER)

        # Load video feeds
        self.load_video_feeds()

        # Load data logs
        self.load_data_logs()

    ###################################################
    # LOADING METHODS
    ###################################################

    def load_video_feeds(self):
        """
        Loads the video feeds extracted for the scene.
        """

        # Scan video feed files of the scene
        video_feeds = {}
        for root, _, files in os.walk(self.video_feeds_folder):
            for file in files:
                if file.endswith(f".{DEFAULT_RAW_VIDEO_EXTENSION}"):
                    
                    # Resolve agent
                    res = re.match(VEHICLE_AGENT_VIDEO_FEED_REGEXP, file)
                    if res is not None:
                        agent_id = int(res.group(1))
                    res = re.match(ROADSIDE_AGENT_VIDEO_FEED_REGEXP, file)
                    if res is not None:
                        agent_id = int(res.group(1))
                    agent = self.scene.base_scene.agents[agent_id]

                    # Create video log
                    video_feeds[agent_id] = VideoLog(os.path.join(root, file), agent)
        
        # Set video feeds
        self.video_feeds = video_feeds

    def load_data_logs(self):
        """
        Loads the text data logs extracted for the scene.
        """

        # Get SCANeR sampling frequency
        scaner_sampling_freq_hz = self.scene.base_scene.scaner_sampling_freq_hz
        
        # Load vehicle data log
        vehicle_log_path = os.path.join(self.data_logs_folder, f"{DEFAULT_RAW_VEHICLES_LOG_NAME}.{DEFAULT_RAW_LOG_EXTENSION}")
        self.vehicle_log = VehicleLog(vehicle_log_path, scaner_sampling_freq_hz)

        # Load pedestrial data log
        pedestrian_log_path = os.path.join(self.data_logs_folder, f"{DEFAULT_RAW_PEDESTRIAN_LOG_NAME}.{DEFAULT_RAW_LOG_EXTENSION}")
        self.pedestrian_log = PedestrianLog(pedestrian_log_path, scaner_sampling_freq_hz)

        # Load movable targets bounding boxes log
        bbox_log_path = os.path.join(self.data_logs_folder, f"{DEFAULT_RAW_BBOX_MOV_LOG_NAME}.{DEFAULT_RAW_LOG_EXTENSION}")
        self.bbox_log = BBoxLog(bbox_log_path, scaner_sampling_freq_hz, True)

        # Load infrastructure targets bounding boxes log
        bbox_infra_log_path = os.path.join(self.data_logs_folder, f"{DEFAULT_RAW_BBOX_INFRA_LOG_NAME}.{DEFAULT_RAW_LOG_EXTENSION}")
        self.bbox_infra_log = BBoxLog(bbox_infra_log_path, scaner_sampling_freq_hz, False)

        # Load GNSS data log
        gnss_log_path = os.path.join(self.data_logs_folder, f"{DEFAULT_RAW_GNSS_LOG_NAME}.{DEFAULT_RAW_LOG_EXTENSION}")
        self.gnss_log = GNSSLog(gnss_log_path, scaner_sampling_freq_hz)

        # Load LiDAR data log
        lidar_log_path = os.path.join(self.data_logs_folder, f"{DEFAULT_RAW_LIDAR_LOG_NAME}.{DEFAULT_RAW_LOG_EXTENSION}")
        self.lidar_log = LiDARLog(lidar_log_path, scaner_sampling_freq_hz)

    ###################################################
    # LOG PROCESSING METHODS
    ###################################################

    def compute_valid_sampling_window(self):
        """
        Compute the start and end frame number of the valid time window to sample the scene.
        """

        # Get valid data logs sampling start frames
        data_logs_start_frames = [self.bbox_log.start_frame, self.vehicle_log.start_frame, self.lidar_log.start_frame, self.gnss_log.start_frame]

        # Get valid data logs sampling end frames
        data_logs_end_frames = [self.bbox_log.end_frame, self.vehicle_log.end_frame, self.lidar_log.end_frame, self.gnss_log.end_frame]

        # Get valid video logs sampling end frames
        video_logs_end_frames = [video_log.num_frames for video_log in self.video_feeds.values()]

        # Determine sampling window for the scene
        start_frame = max(data_logs_start_frames)
        end_frame = min(data_logs_end_frames + video_logs_end_frames)

        # Return sampling window
        return start_frame, end_frame

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"SceneLog(token='{self.token}', path='{self.path}')"

    def __repr__(self):
        return f"SceneLog(token='{self.token}', path='{self.path}')"
