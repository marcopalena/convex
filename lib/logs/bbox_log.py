import re
import pandas as pd
import numpy as np
from .log import SCANeRDataLog
from ..bboxes import BBox, ReferenceFrame

# Regular expression to extract sensor ID from log headings
SENSOR_MOVEABLE_TARGET_VEHICLE_ID_RE = r"\[([0-9\-]+)\]\.SensorMovableTargetsBoundingBoxes-vhlId"
SENSOR_INFRASTRUCTURE_TARGET_VEHICLE_ID_RE = r"\[([0-9\-]+)\]\.SensorInfrastructureTargetsBoundingBoxes-vhlId"

class BBoxLog(SCANeRDataLog):
    """
    Class to parse and navigate bounding box log files extracted from SCANeR Studio.
    """

    def __init__(self, path, sample_freq, moveable):

        # Call super class constructor
        super().__init__(path, sample_freq) 
        self.moveable = moveable

        # Parse log file
        self.parse()

    ###################################################
    # PARSING
    ###################################################

    def parse(self):
        """
        Parses a bounding box log file.
        """

        # Read raw bounding box data
        df = pd.read_csv(self.path, sep="\t")
        if not self.moveable:
            with open("output2.txt", "w") as f:
                f.writelines(df.columns)


        # Get sensor ids
        self.get_sensor_ids(df)

        # Get frames
        self.get_frames(df)

        # Get sizes of the bounding box array for each sensor
        self.get_bbox_array_sizes(df)

        # Parse bounding boxes
        self.parse_bboxes(df)

        # Compute valid sampling window
        self.compute_valid_sampling_window(df)

    def get_sensor_ids(self, df):
        """
        Each camera sensor in SCANeR Studio can output bbox information. The collected
        bbox info in the bbox log are labelled with sensor ids so we need to collect them
        alongside the corresponding vehicle ids.
        """

        # Extract sensor ids
        ids = set()
        for header in df.columns.to_list():
            if self.moveable:
                regexp = SENSOR_MOVEABLE_TARGET_VEHICLE_ID_RE
            else:
                regexp = SENSOR_INFRASTRUCTURE_TARGET_VEHICLE_ID_RE
            res = re.match(regexp, header)
            if res is not None:
                ids.add(res.group(1))
        self._sensor_ids = sorted(list(ids))

        # Extract vehicle ids to which the sensors are attached
        df = df.iloc[:, 1:len(self._sensor_ids)+1].dropna().astype('int64')
        df = df.rename(columns=lambda x: re.sub(r'^\[','', re.sub(r'\].*$', '', x))).iloc[0]
        self._agent_ids = [int(df.loc[sensor_id]) for sensor_id in self._sensor_ids]

        # Compose sensor ID - vehicle ID mapping dictionaries
        self._sensId2vehId = dict(map(lambda i,j : (i,j) , self._sensor_ids, self._agent_ids))
        self._sensId2index = dict(map(lambda i,j : (i,j) , self._sensor_ids, range(0, len(self.sensor_ids))))
        self._vehId2sensId = dict(map(lambda i,j : (i,j) , self._agent_ids, self._sensor_ids))

    def get_bbox_array_sizes(self, df):
        """
        Extract the size of the bbox array for each sensor.
        """

        bbox_array_sizes = []
        for sensor_id in self.sensor_ids:
            indices = []
            for header in df.columns.to_list():
                if self.moveable:
                    res = re.match(fr"^\[{sensor_id}\].SensorMovableTargetsBoundingBoxes-boundingBoxesArray-id.([0-9]+)$", header)
                else:
                    res = re.match(fr"^\[{sensor_id}\].SensorInfrastructureTargetsBoundingBoxes-boundingBoxesArray-id.([0-9]+)$", header)
                if res is not None:
                    indices.append(int(res.group(1)))

            if len(indices) > 0:
                bbox_array_sizes.append(max(indices) + 1)
            else:
                bbox_array_sizes.append(0)
       
        self._bbox_array_sizes = bbox_array_sizes

    def get_tot_bboxes_array_offset(self):
        return sum(self._bbox_array_sizes)

    def get_bboxes_array_offset(self, index):
        return sum(self._bbox_array_sizes[:index])

    def parse_bboxes(self, df):
        self._bbox_dfs =  dict()
        for sensor_id in self.sensor_ids:
            self.parse_sensor_bboxes(df, sensor_id)


    def compute_valid_sampling_window(self, df):
        """
        Compute the start and end frame number of the valid time window of observations.
        """

        # Select sampling time and time of update columns
        start = 1+len(self.sensor_ids)+1+(2+8*3)*self.get_tot_bboxes_array_offset()
        end = start + len(self.sensor_ids)
        slices = [0]
        slices.append(slice(start, end))
        selector = np.r_[*slices]
        df = df.iloc[:, selector]

        # Discard sampling times for which at least a sensor has no bbox observations
        df = df.set_index(['time']).dropna()

        # Set valid sampling window boundaries
        self.start_frame = round(df.index[0] / self.sampling_period)
        self.end_frame = round(df.index[-1] / self.sampling_period)

    # def estimate_frames_to_skip(self, df):

    #     # Get sampling time and time of update columns
    #     start = 1+len(self.sensor_ids)+1+(2+8*3)*self.get_tot_bboxes_array_offset()
    #     end = start + len(self.sensor_ids)
    #     slices = [0]
    #     slices.append(slice(start, end))
    #     selector = np.r_[*slices]
    #     df = df.iloc[:, selector]

    #     # Compute time at which updates normalize
    #     df = df.set_index(['time']).dropna()
    #     std = df.std(axis='columns')
    #     mean = df.mean(axis='columns')
    #     delta = (mean - df.index).abs()
    #     check_update_time_std = std  < 0.1
    #     df['mean'] = mean
    #     df['delta'] = delta
    #     df['std'] = std
    #     df['check_update_time_std'] = check_update_time_std
    #     # assert update_time_normalized.is_monotonic_increasing   # If update times do not normalize abort
        
    #     # Debug
    #     # df.iloc[:, [-4, -3, -2, -1]].to_csv('debug.csv', float_format='%.4f')
    #     # print(df.index[0])
    #     # print(check_update_time_std[::-1].idxmin())
    #     # print(self.sampling_period)
    #     # print(round(check_update_time_std[::-1].idxmin() / self.sampling_period))

    #     # Compute number of frames to skip
    #     # self.frames_to_skip = round(check_update_time_std[::-1].idxmin() / self.sampling_period)+1
    #     self.frames_to_skip = round(df.index[0] / self.sampling_period)
    #     self.start_frame = round(df.index[0] / self.sampling_period)
    #     self.end_frame = round(df.index[0] / self.sampling_period)


    def parse_sensor_bboxes(self, df, sensor_id):
        
        # Get index of sensor_id
        index = self._sensId2index[sensor_id]

        # Compute column slices
        slices = [0]
        for i in range(2+8*3):
            start = 1+len(self.sensor_ids)+1+i*self.get_tot_bboxes_array_offset()+self.get_bboxes_array_offset(index)
            end = start + self._bbox_array_sizes[index]
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

        # Set bbox dataframe
        self._bbox_dfs[index] = df.dropna()

    def get_frames(self, df):
        """
        Extracts list of frames from the bounding box log file.
        """
        self._frames = np.arange(0.0, df['time'].max() + self.sampling_period, self.sampling_period).round(2)

    ###################################################
    # BBOX LOG QUERIES
    ###################################################
    def get_bboxes(self, vehId):

        # Convert sensing vehicle ID to index
        index = self._sensId2index[self._vehId2sensId[vehId]]

        # Get vehicle bounding boxes at frame
        return self._bbox_dfs[index]
    
    def get_agent_bboxes_at_frame(self, agent, frame):
        """
        Returns a list of BBox objects corresponding to the bounding boxes
        detected by the sensors equipped to a given agent at a given frame,
        along with the ids of their targets.

        Parameters:
            agent (Agent): An given agent object.
            frame (int): An integer specifying a given frame number.

        Returns:
            bboxes ([(int, BBox)]): A list of 2-uples each containing an integer specifying the target id 
            and a bounding box object.
        """

        # Check if agent bboxes are present in log
        target_bboxes = []
        if not agent.id in self._vehId2sensId:
            return target_bboxes

        # Convert sensing agent id to index
        index = self._sensId2index[self._vehId2sensId[agent.id]]

        # Convert frame numeber to time offset
        time = frame/self.sampling_freq

        # Get agent bounding boxes dataframe at frame
        if time not in self._bbox_dfs[index].index:
            return target_bboxes
        df = self._bbox_dfs[index].loc[time]

        # Create bboxes from dataframe rows
        for index, row in df.iterrows():

            # Parse bounding box information from row
            corners, target_id, reference_frame = self.parse_bbox_row(row)

            # Create bounding box
            bbox = BBox(corners, reference_frame)
            assert(bbox.is_world_coordinates())

            # Add bounding box and target tuple to list
            target_bboxes.append((target_id, bbox))

        # Return list of detected targets with their bboxes
        return target_bboxes

    def parse_bbox_row(self, row):
        """
        Parses a dataframe row containing bounding box information about a given 
        target detected by a given agent at a given frame.

        Parameters:
            row (pd.Series): A pd.Series containing information about a bounding box.

        Returns:
            corners (np.ndarray): A (8,3) array representing the corner points of the bounding box in the
                following order: (RBL, FBL, FBR, RBR, RTL, FTL, FTR, RTR).
            target_id (int): An integer specifying the id of the detected target.
            reference_frame (int): An integer identifying the reference frame according to which the corner
                points coordinates are give (0=world, 1=ego agent, 2=sensor).
        """
        
        # Parse target id and reference frame from row
        target_id = int(row.loc[('value', 'id')])
        reference_frame = int(row.loc[('value', 'referenceFrame')])
        
        # Parse corner points from row
        front_bottom_right = np.array([row.loc[('value', 'frontBottomRightX')], row.loc[('value', 'frontBottomRightY')], row.loc[('value', 'frontBottomRightZ')]])
        front_bottom_left = np.array([row.loc[('value', 'frontbottomLeftX')], row.loc[('value', 'frontbottomLeftY')], row.loc[('value', 'frontbottomLeftZ')]])
        front_top_left = np.array([row.loc[('value', 'frontTopLeftX')], row.loc[('value', 'frontTopLeftY')], row.loc[('value', 'frontTopLeftZ')]])
        front_top_right = np.array([row.loc[('value', 'frontTopRightX')], row.loc[('value', 'frontTopRightY')], row.loc[('value', 'frontTopRightZ')]])
        rear_bottom_right = np.array([row.loc[('value', 'rearBottomRightX')], row.loc[('value', 'rearBottomRightY')], row.loc[('value', 'rearBottomRightZ')]])
        rear_bottom_left = np.array([row.loc[('value', 'rearbottomLeftX')], row.loc[('value', 'rearbottomLeftY')], row.loc[('value', 'rearbottomLeftZ')]])
        rear_top_left = np.array([row.loc[('value', 'rearTopLeftX')], row.loc[('value', 'rearTopLeftY')], row.loc[('value', 'rearTopLeftZ')]])
        rear_top_right = np.array([row.loc[('value', 'rearTopRightX')], row.loc[('value', 'rearTopRightY')], row.loc[('value', 'rearTopRightZ')]])
            
        # Compose corners points array
        corners = np.array([rear_bottom_left, front_bottom_left, front_bottom_right, rear_bottom_right, rear_top_left, front_top_left, front_top_right, rear_top_right])

        # Return bounding box info
        return corners, target_id, reference_frame

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
    # MAGIC METHODS
    ###################################################
    def __str__(self):
        return f"BBoxLog(token='{self.token}', path='{self.path}', num_agents={self.num_agents}, num_frames={self.num_frames})"

    def __repr__(self):
        return f"BBoxLog(token='{self.token}', path='{self.path}', num_agents={self.num_agents}, num_frames={self.num_frames})"
