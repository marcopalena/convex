import re
import pandas as pd
import numpy as np
from .log import SCANeRDataLog
from ..points import LidarPoint, LidarPointCloud, LidarPointAnnotation

# Regular expression to extract sensor ID from log headings
SENSOR_VEHICLE_ID_RE = r"\[([0-9\-]+)\]\.LidarPoints-vhlId"

LIDAR_POINTS_FIELDS = [
    'vhlId',            # Vehicle to which the LiDAR sensor is attached to
    # 'frameId',          # ??
    'timeOfUpdate',     # Time of last update of the LiDAR points
    'referenceFrame'   # Frame with respect to which point coordinates are reported
    # 'rotationSpeed',    # Current rotation speed of the LiDAR
    # 'hRange',           # Angle range covered by the LidarPoints message
    # 'sequenceNb',       # Number of LiDAR sequences in the message
    # 'verticalRaysNb',   # Number of vertical rays?
    # 'nearestPointId'    # Id of the nearest point on a target reached by a ray (is it the id of the ray or the target?)
]

POINTS_INFO_FIELDS = [
    'rayId',                # Unique identifier assigned to each ray in the pattern
    # 'hAngleInPattern',      # Horizontal angle of a ray in the pattern
    # 'vAngleInPattern',      # Vertical angle of a ray in the pattern
    # 'hAngleInSensor',       # Horizontal angle of a ray from the lidar view
    # 'vAngleInSensor',       # Vertical angle of a ray from the lidar view
    'hit',                  # Boolean indicating whether the ray has hit a target
    # 'targetId',             # NOT USED
    'targetScanerId',       # Id of the target that has been hit (this id can be forced to -1 if the max detection range for the type of target hitted is exceeded or if hit is false.)
    'targetScanerType',     # Type of target hit
    'posXInRefFrame',       # X position of the impact point in the reference frame
    'posYInRefFrame',       # Y position of the impact point in the reference frame
    'posZInRefFrame',       # Z position of the impact point in the reference frame
    'normalXInRefFrame',    # X position of the normal at the impact point in the reference frame
    'normalYInRefFrame',    # Y position of the normal at the impact point in the reference frame
    'normalZInRefFrame',    # Z position of the normal at the impact point in the reference frame
    'distance',             # Distance between the hit point and the sensor
    'intensity',            # NOT USED
    # 'reflectivity',         # NOT USED
    # 'echo'                  # NOT USED
]

SEQUENCE_INFO_ARRAY_FIELDS = [
    'hAngle'                # Horizontal angle at which the pattern is thrown
]

POINTS_INFO_FIELDS_OF_INTEREST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

SCANER_TARGET_TYPE = {
    0 : 'UNKNOWN',
    1 : 'RIGID',
    2 : 'TRACTOR',
    3 : 'SEMI_TRAILER',
    4 : 'TRAILER',
    5 : 'CAR',
    6 : 'BUS',
    7 : 'MOTORBIKE',
    8 : 'BICYCLE',
    9 : 'PEDESTRIAN',
    10 : 'STATIC_OBJECT',
    11 : 'TRAM',
    12 : 'TRAIN',
    13 : 'ALIVEBEING',
    14 : 'AIRPLANE',
    15 : 'PUSHBACK',
    16 : 'ROADSIGN',
    17 : 'OBSTACLE',
    18 : 'TRAFFICLIGHT',
    19 : 'ROADMARK',
    20 : 'ROAD',
    21 : 'SIDEWALK'
}

class LiDARLog(SCANeRDataLog):
    """
    Class to parse and navigate LiDAR log files extracted from SCANeR Studio.
    """

    def __init__(self, path, sample_freq):

        # Call super class constructor
        super().__init__(path, sample_freq) 

        # Parse log file
        self.parse()

    ###################################################
    # PARSING
    ###################################################

    def parse(self):
        """
        Parses a LiDAR log file.
        """

        # Read raw bounding box data
        df = pd.read_csv(self.path, sep="\t")
        
        # Get sensor ids
        self.get_sensor_ids(df)

        # Get frames
        self.get_frames(df)

        # self.get_sensor_info(df)

        # Get sizes of the bounding box array for each sensor
        # self.get_sequence_info_array_sizes(df)
        self.get_points_info_array_sizes(df)

        # Parse LiDAR points
        self.parse_points(df)

        # Compute valid sampling window
        self.compute_valid_sampling_window(df)

    def get_sensor_ids(self, df):
        """
        Extract LiDAR sensor ids and corresponding vehicle ids.
        """

        # Extract sensor ids
        ids = set()
        for header in df.columns.to_list():
            res = re.match(SENSOR_VEHICLE_ID_RE, header)
            if res is not None:
                ids.add(res.group(1))
        self._sensor_ids = sorted(list(ids))

        # Extract vehicle ids to which the sensors are attached
        df = df.iloc[:, 1:len(self._sensor_ids)+1].dropna().astype('int64')
        df = df.rename(columns=lambda x: re.sub(r'^\[','', re.sub(r'\].*$', '', x))).iloc[0]
        self._agent_ids = [int(df.loc[sensor_id]) for sensor_id in self._sensor_ids]

        # Compose sensor ID - vehicle ID mapping dictionaries
        self._sensId2vehId = dict(map(lambda i,j : (i,j) , self._sensor_ids, self._agent_ids))
        self._sensId2index = dict(map(lambda i,j : (i,j) , self._sensor_ids, range(0, len(self._sensor_ids))))
        self._vehId2sensId = dict(map(lambda i,j : (i,j) , self._agent_ids, self._sensor_ids))

    def get_sequence_info_array_sizes(self, df):
        """
        Extract the size of the sequence info arrays for each sensor.
        """
        sequence_info_array_sizes = []
        for sensor_id in self._sensor_ids:
            indices = []
            for header in df.columns.to_list():
                res = re.match(fr"^\[{sensor_id}\].LidarPoints-SequenceInfosArray-hAngle.([0-9]+)$", header)
                if res is not None:
                    indices.append(int(res.group(1)))
            sequence_info_array_sizes.append(max(indices) + 1)
        self._sequence_info_array_sizes = sequence_info_array_sizes

    def get_points_info_array_sizes(self, df):
        """
        Extract the size of the points info arrays for each sensor.
        """
        points_info_array_sizes = []
        for sensor_id in self._sensor_ids:
            indices = []
            for header in df.columns.to_list():
                res = re.match(fr"^\[{sensor_id}\].LidarPoints-PointsInfosArray-rayId.([0-9]+)$", header)
                if res is not None:
                    indices.append(int(res.group(1)))
            points_info_array_sizes.append(max(indices) + 1)
        self._points_info_array_sizes = points_info_array_sizes


    def get_lidar_points_num_cols(self):
        return len(LIDAR_POINTS_FIELDS)*len(self.sensor_ids)
    
    def get_sequence_info_array_cols_per_field(self):
        return sum(self._sequence_info_array_sizes)

    def get_sequence_info_array_num_cols(self):
        cols_per_field = self.get_sequence_info_array_cols_per_field()
        return (
            1 + # LidarPoints-SequenceInfosArrayCount 
            cols_per_field * len(SEQUENCE_INFO_ARRAY_FIELDS)
        )
    
    def get_points_info_array_cols_per_field(self):
        return sum(self._points_info_array_sizes)

    def get_points_info_array_num_cols(self):
        cols_per_field = self.get_points_info_array_cols_per_field()
        return (
            1 + # LidarPoints-PointsInfosArrayCount
            cols_per_field * len(POINTS_INFO_FIELDS)
        )

    def get_points_info_array_cols_offset(self, index):
        return sum(self._points_info_array_sizes[:index])

    # def get_sensor_info(self, df):
    #     """
    #     """

    #     # Extract sensor info columns
    #     slices = [0]
    #     columns = ['frameId', 'timeOfUpdate', 'referenceFrame', 'rotationSpeed', 'hRange', 'sequenceNb', 'verticalRaysNb', 'nearestPointId']
    #     start = 1 + len(self._sensor_ids)
    #     end = start + len(columns)*len(self._sensor_ids)
    #     slices.append(slice(start, end))
    #     selector = np.r_[*slices]
    #     df = df.iloc[:, selector]

    #     # Stack columns by timeframe
    #     df = df.set_index(['time']).stack(future_stack=True)

    #     # Flatten multi-index and rename columns
    #     df = df.reset_index().rename(columns={'level_1': 'field', 0: 'value'})
        
    #     # Extract sensor ids for each row
    #     df['id'] = df['field'].str.extract(pat = r"^\[([0-9\-]+)\]")
    #     df['field'] = df['field'].str.extract(pat = r"\[[0-9\-]+\]\.[^\-]+\-([^\.]+)")

    #     # Pivot table using timestamp and id as index and the values of parameter as columns
    #     df = df.pivot(index=['time', 'id'], columns='field').dropna()
    #     print(df)
    #     print(df.loc[4.35, ('value', 'nearestPointId')])
        
    #     # Flatten multi-index and rename columns
    #     # df = df.reset_index().rename(columns={'level_1': 'id', 0: 'value'})


    #     # bbox_array_sizes = []
    #     # for sensor_id in self.sensor_ids:
    #     #     indices = []
    #     #     for header in df.columns.to_list():
    #     #         res = re.match(fr"^\[{sensor_id}\].SensorMovableTargetsBoundingBoxes-boundingBoxesArray-id.([0-9]+)$", header)
    #     #         if res is not None:
    #     #             indices.append(int(res.group(1)))
    #     #     bbox_array_sizes.append(max(indices) + 1)
    #     # self._bbox_array_sizes = bbox_array_sizes

    def parse_points(self, df):
        """
        Parse LiDAR points for each sensor.
        """
        self._points_dfs =  dict()
        for sensor_id in self.sensor_ids:
            self.parse_sensor_points(df, sensor_id)


    def parse_sensor_points(self, df, sensor_id):
        """
        Parse LiDAR points for a given sensor.
        """

        # Get index of sensor_id
        index = self._sensId2index[sensor_id]

        # Compute column slices
        slices = [0]    # Time slice
        offset = 1 + self.get_lidar_points_num_cols() #+ self.get_sequence_info_array_num_cols() + 1
        for i in POINTS_INFO_FIELDS_OF_INTEREST:
            start = offset + i*self.get_points_info_array_cols_per_field() + self.get_points_info_array_cols_offset(index)
            end = start + self._points_info_array_sizes[index]
            slices.append(slice(start, end))

        # Select bounding boxes columns for the given sensor
        selector = np.r_[*slices]
        df = df.iloc[:, selector]

        # Stack columns by timeframe
        df = df.set_index(['time']).stack(future_stack=True)

        # Flatten multi-index and rename columns
        df = df.reset_index().rename(columns={'level_1': 'field', 0: 'value'})

        # Add column reporting indices
        df["index"] = df['field'].str.extract(pat = r"\.([0-9]+)$")

        # Rename field column
        df["field"] = df['field'].str.extract(pat = r"\[[0-9\-]+\]\.[^\-]+\-[^\-]+\-([^\.]+)\.")

        # Pivot table using timestamp and id as index and the values of parameter as columns
        df = df.pivot(index=['time', 'index'], columns='field')

        # Filter rays that have not hit a target
        df = df.loc[df[('value', 'hit')] == 1.0]
        
        # Set lidar points dataframe
        self._points_dfs[index] = df.dropna()

    def get_frames(self, df):
        """
        Extracts list of frames from the lidar log file.
        """
        self._frames = np.arange(0.0, df['time'].max() + self.sampling_period, self.sampling_period).round(2)

    ###################################################
    # PROPERTIES
    ###################################################
    @property
    def sensor_ids(self):
        return self._sensor_ids
    
    @property
    def agent_ids(self):
        return self._agent_ids
    
    @property
    def num_agents(self):
        return len(self.agent_ids)
    
    @property
    def num_frames(self):
        return len(self._frames)

    @property
    def frames(self):
        return self._frames
    

    ###################################################
    # LIDAR LOG QUERIES
    ###################################################
    
    def get_agent_points_at_frame(self, agent, frame):
        """
        Returns a list of LidarPoint objects corresponding to the 3D points
        detected by the LiDAR sensor equipped to a given agent at a given 
        frame, along with the ids of their targets.

        Parameters:
            agent (Agent): An given agent object.
            frame (int): An integer specifying a given frame number.

        Returns:
            points ([(int, Point)]): A list of 2-uples each containing an integer specifying the target id 
            and a bounding box object.
        """

        # Check if agent points are present in log
        lidar_points = []
        if not agent.id in self._vehId2sensId:
            return lidar_points

        # Convert sensing agent id to index
        index = self._sensId2index[self._vehId2sensId[agent.id]]

        # Convert frame number to time offset
        time = frame/self.sampling_freq

        # Get agent points dataframe at frame
        df = self._points_dfs[index].loc[time]

        # Create points from dataframe rows
        for index, row in df.iterrows():

            # Parse point information from row
            coordinates, normal, target_id, target_type, intensity, distance = self.parse_lidar_point_row(row)

            # Create LiDAR point
            lidar_point = LidarPoint(coordinates, normal, intensity, distance)

            # Create LiDAR point annotation
            lidar_point_annotation = LidarPointAnnotation(lidar_point, target_id, target_type)

            # Add LiDAR point and annotation tuple to list
            lidar_points.append((lidar_point, lidar_point_annotation))

        # Return list of detected targets with their bboxes
        return lidar_points

    def parse_lidar_point_row(self, row):
        """
        Parses a dataframe row containing LiDAR point information detected by 
        a given agent at a given frame.

        Parameters:
            row (pd.Series): A pd.Series containing information about a LiDAR point.

        Returns:

        """
        
        # Parse target id and type from row
        target_id = int(row.loc[('value', 'targetScanerId')])
        target_type = int(row.loc[('value', 'targetScanerType')])

        # Parse intensity and distance from sensor
        intensity = float(row.loc[('value', 'intensity')])
        distance = float(row.loc[('value', 'distance')])
        
        # Parse point coordinates
        point_coordinates = np.array([row.loc[('value', 'posXInRefFrame')], row.loc[('value', 'posYInRefFrame')], row.loc[('value', 'posZInRefFrame')]])
        normal = np.array([row.loc[('value', 'normalXInRefFrame')], row.loc[('value', 'normalYInRefFrame')], row.loc[('value', 'normalZInRefFrame')]])
        
        # Return points info
        return point_coordinates, normal, target_id, target_type, intensity, distance
    

    def compute_valid_sampling_window(self, df):
        """
        Compute the start and end frame number of the valid time window of observations.
        """

        # Select only timeframe and time of update columns
        slices = [0]
        start = 1 + len(self._sensor_ids)
        end = start + len(self._sensor_ids)
        slices.append(slice(start, end))
        selector = np.r_[*slices]
        df = df.iloc[:, selector]

        # Discard sampling times for which at least a LiDAR sensor has no updates
        df = df.set_index(['time']).dropna()

        # Set valid sampling window boundaries
        self.start_frame = round(df.index[0] / self.sampling_period)
        self.end_frame = round(df.index[-1] / self.sampling_period)