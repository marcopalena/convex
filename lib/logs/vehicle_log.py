import re
import pandas as pd
import numpy as np
from .log import SCANeRDataLog
from ..poses import EgoPose, EgoSpeed

# Order in with the ego pose parameter values appear in the vehicle log file
EGO_POSE_PARAMETERS = ["X", "Y", "Z", "Heading", "Pitch", "Roll"]
EGO_SPEED_PARAMETERS = ["X", "Y", "Z", "Heading", "Pitch", "Roll"]

# Regular expression to extract vehicle ID from log headings
VEHICLE_POS_RE = r"\[([0-9]+)\]\.VehicleUpdate-pos\.([0-9])+"


class VehicleLog(SCANeRDataLog):
    """
    Class to parse and navigate vehicle log files extracted from SCANeR Studio. Each vehicle
    log file records information about the simulated vehicles at each time frame. The following 
    information are parsed:
     - Vehicle ego pose
     - Vehicle ego speed
    """

    def __init__(self, path, sampling_freq):

        # Call super class constructor
        super().__init__(path, sampling_freq) 

        # Parse log file
        self.parse()

    ###################################################
    # PARSING
    ###################################################

    def parse(self):
        """
        Parses a vehicle log file.
        """

        # Read raw vehicle data
        df = pd.read_csv(self.path, sep="\t")

        # Get vehicle ids
        self.get_vehicle_ids(df)

        # Get frames
        self.get_frames(df)

        # Parse vehicle ego poses
        self.parse_ego_poses(df)

        # Parse vehicle ego speeds
        self.parse_ego_speeds(df)

        # Compute valid sampling window
        self.compute_valid_sampling_window(df)


    def compute_valid_sampling_window(self, df):
        """
        Compute the start and end frame number of the valid time window of observations.
        """

        # Select only timeframe and pos columns
        df = df.iloc[:, :3*6*len(self.ids)+11*len(self.ids)+1]
        df = df.drop(df.iloc[:, 1:3*6*len(self.ids)+10*len(self.ids)+1], axis=1)

        # Discard sampling times for which at least a vehicle has no updates
        df = df.set_index(['time']).dropna()

        # Set valid sampling window boundaries
        self.start_frame = round(df.index[0] / self.sampling_period)
        self.end_frame = round(df.index[-1] / self.sampling_period)

    # def estimate_frames_to_skip(self, df):

    #     # Select only timeframe and pos columns
    #     df = df.iloc[:, :3*6*len(self.ids)+11*len(self.ids)+1]
    #     df = df.drop(df.iloc[:, 1:3*6*len(self.ids)+10*len(self.ids)+1], axis=1)
    #     print(df)

    #     # Compute time at which updates normalize
    #     vehicle_update_available = df.set_index(['time']).iloc[:, 1:].isnull().apply(lambda x: all(x), axis=1)
    #     assert vehicle_update_available.is_monotonic_decreasing 

    #     # Compute number of frames to skip
    #     # self.frames_to_skip = round(vehicle_update_available.idxmin() / self.sampling_period)
    #     self.frames_to_skip = round(vehicle_update_available.index[0] / self.sampling_period)
    #     self.start_frame = round(vehicle_update_available.index[0] / self.sampling_period)
    #     self.end_frame = round(vehicle_update_available.index[-1] / self.sampling_period)

    def parse_ego_poses(self, df):
        """
        Parses ego poses as a MultiIndex pd.DataFrame from the vehicle log file. The MultiIndex
        uses a float representing the timeframe as first level and a string representing the
        vehicle ID as second level. Each row reporst the position, (X, Y, Z) coordinates in the 
        global reference frame) and orientation, (Pitch, Roll, Heading) in rad, of a given vehicle 
        at a given time frame. Vehicles missing from a frame have NaN values.
        """

        # Select only timeframe and pos columns
        df = df.iloc[:, :6*len(self.ids)+1]

        # Stack columns by timeframe
        df = df.set_index(['time']).stack(future_stack=True)
        
        # Flatten multi-index and rename columns
        df = df.reset_index().rename(columns={'level_1': 'id', 0: 'value'})

        # Extract vehicle ids for each row
        df['id'] = df['id'].str.extract(pat = r"^\[([0-9]+)\]").astype(int)

        # Add a column reporting the repeating ego pose parameter names
        N=len(df)
        pars = pd.Series(np.tile(EGO_POSE_PARAMETERS, N//len(EGO_POSE_PARAMETERS))).iloc[:N]
        df['par'] = pars

        # Pivot table using timestamp and id as index and the values of parameter as columns
        # df['frame'] = (df['time']/self.sampling_period).astype(int)
        df = df.pivot(index=['time', 'id'], columns='par')

        # Rename columns with ego pose parameters
        df.columns = df.columns.get_level_values(1)

        # Set ego pose dataframe
        self._df_ego_pose = df


    def parse_ego_speeds(self, df):
        """
        Parses ego speeds as a MultiIndex pd.DataFrame from the vehicle log file. The MultiIndex
        uses a float representing the timeframe as first level and a string representing the
        vehicle ID as second level. Each row reports the speed, with (X,Y,Z)) components in m/s 
        and (Pitch, Roll, Heading) components in rad/s, of a given vehicle at a given time frame.
        Vehicles missing from a frame have NaN values.
        """

        # Select only timeframe and speed columns
        df = df.iloc[:, :2*6*len(self.ids)+1]
        df = df.drop(df.iloc[:, 1:6*len(self.ids)+1], axis=1)

        # Stack columns by timeframe
        df = df.set_index(['time']).stack(future_stack=True)

        # Flatten multi-index and rename columns
        df = df.reset_index().rename(columns={'level_1': 'id', 0: 'value'})

        # Extract vehicle ids for each row
        df['id'] = df['id'].str.extract(pat = r"^\[([0-9]+)\]").astype(int)

        # Add a column reporting the repeating ego pose parameter names
        N=len(df)
        pars = pd.Series(np.tile(EGO_SPEED_PARAMETERS, N//len(EGO_SPEED_PARAMETERS))).iloc[:N]
        df['par'] = pars

        # Pivot table using timestamp and id as index and the values of parameter as columsn
        df = df.pivot(index=['time', 'id'], columns='par')

        # Rename columns with ego pose parameters
        df.columns = df.columns.get_level_values(1)

        # Set ego pose dataframe
        self._df_ego_speed = df

    def get_frames(self, df):
        """
        Extracts list of frames from the vehicle log file.
        """
        self._frames = np.arange(0.0, df['time'].max() + self.sampling_period, self.sampling_period).round(2)

    def get_vehicle_ids(self, df):
        """
        Extracts vehicle IDs from the headers of the vehicle log file.
        """
        ids = set()
        for header in df.columns.to_list():
            res = re.match(VEHICLE_POS_RE, header)
            if res is not None:
                ids.add(res.group(1))
        self._ids = sorted(list(ids))
    
    ###################################################
    # PROPERTIES
    ###################################################
    @property
    def ids(self):
        return self._ids
    
    @property
    def num_vehicles(self):
        return len(self.ids)
    
    @property
    def frames(self):
        return self._frames

    @property
    def num_frames(self):
        return len(self._frames)
    
    ###################################################
    # VEHICLE LOG QUERIES
    ###################################################
    def vehicles_at_frame(self, frame):
        time = frame/self.sampling_freq
        return self._df_ego_pose.loc[time].dropna().index.to_list()
    
    def num_vehicles_at_frame(self, frame):
        return len(self.vehicles_at_frame(frame))
    
    def is_vehicle_in_frame(self, id, frame):
        return id in self.vehicles_at_frame(frame)
    
    def get_ego_pose(self, id, frame):
        time = frame/self.sampling_freq
        rotation = self._df_ego_pose.loc[(time, id)][:3].to_numpy()
        translation = self._df_ego_pose.loc[(time, id)][3:].to_numpy()
        return EgoPose(time, translation, rotation)
    
    def get_ego_speed(self, id, frame):
        time = frame/self.sampling_freq
        rotation_speed = self._df_ego_pose.loc[(time, id)][:3].to_numpy()
        translation_speed = self._df_ego_pose.loc[(time, id)][3:].to_numpy()
        return EgoSpeed(time, translation_speed, rotation_speed)
    
    ###################################################
    # MAGIC METHODS
    ###################################################
    def __str__(self):
        return f"VehicleLog(token='{self.token}', path='{self.path}', num_vehicles={self.num_vehicles}, num_frames={self.num_frames})"

    def __repr__(self):
        return f"VehicleLog(token='{self.token}', path='{self.path}', num_vehicles={self.num_vehicles}, num_frames={self.num_frames})"
