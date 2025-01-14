import re
import pandas as pd
import numpy as np
from .log import SCANeRDataLog
from ..poses import EgoPose, EgoSpeed

# Order in with the ego pose parameter values appear in the vehicle log file
EGO_POSE_PARAMETERS = ["X", "Y", "Z", "Heading", "Pitch", "Roll"]
EGO_SPEED_PARAMETERS = ["X", "Y", "Z", "Heading", "Pitch", "Roll"]

# Regular expression to extract pedestrian ID from log headings
PEDESTRIAN_POS_RE = r"\[([0-9]+)\]\.VehicleUpdate-pos\.([0-9])+"

class PedestrianLog(SCANeRDataLog):
    """
    Class to parse and navigate pedestrian log files extracted from SCANeR Studio.
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
        Parses a pedestrian log file.
        """

        # Read raw pedestrian data
        df = pd.read_csv(self.path, sep="\t")

        # Get pedestrian ids
        self.get_pedestrian_ids(df)

        # Get frames
        self.get_frames(df)

        # Parse pedestrian ego poses
        self.parse_ego_poses(df)

        # Parse pedestrian ego speeds
        self.parse_ego_speeds(df)

    def parse_ego_poses(self, df):
        """
        Parses ego poses as a MultiIndex pd.DataFrame from the pedestrian log file.
        """

        # Select only timeframe and pos columns
        df = df.iloc[:, :6*len(self.ids)+1]

        # Stack columns by timeframe
        df = df.set_index(['time']).stack(future_stack=True)
        
        # Flatten multi-index and rename columns
        df = df.reset_index().rename(columns={'level_1': 'id', 0: 'value'})

        # Extract pedestrian ids for each row
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
        Parses ego speeds as a MultiIndex pd.DataFrame from the pedestrian log file.
        """

        # Select only timeframe and speed columns
        df = df.iloc[:, :2*6*len(self.ids)+1]
        df = df.drop(df.iloc[:, 1:6*len(self.ids)+1], axis=1)

        # Stack columns by timeframe
        df = df.set_index(['time']).stack(future_stack=True)

        # Flatten multi-index and rename columns
        df = df.reset_index().rename(columns={'level_1': 'id', 0: 'value'})

        # Extract pedestrian ids for each row
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
        Extracts list of frames from the pedestrian log file.
        """
        self._frames = np.arange(0.0, df['time'].max() + self.sampling_period, self.sampling_period).round(2)

    def get_pedestrian_ids(self, df):
        """
        Extracts pedestrian IDs from the headers of the pedestrian log file.
        """
        ids = set()
        for header in df.columns.to_list():
            res = re.match(PEDESTRIAN_POS_RE, header)
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
    def num_pedestrians(self):
        return len(self.ids)
    
    @property
    def frames(self):
        return self._frames

    @property
    def num_frames(self):
        return len(self._frames)
    
    ###################################################
    # PEDESTRIAN LOG QUERIES
    ###################################################
    def pedestrians_at_frame(self, frame):
        time = frame/self.sampling_freq
        return self._df_ego_pose.loc[time].dropna().index.to_list()
    
    def num_pedestrians_at_frame(self, frame):
        return len(self.pedestrians_at_frame(frame))
    
    def is_pedestrian_in_frame(self, id, frame):
        return id in self.pedestrians_at_frame(frame)
    
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
        return f"PedestrianLog(token='{self.token}', path='{self.path}', num_pedestrian={self.num_pedestrians}, num_frames={self.num_frames})"

    def __repr__(self):
        return f"PedestrianLog(token='{self.token}', path='{self.path}', num_pedestrian={self.num_pedestrians}, num_frames={self.num_frames})"
