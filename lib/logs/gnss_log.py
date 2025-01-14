import re
import pandas as pd
import numpy as np
from .log import SCANeRDataLog

# Regular expression to extract sensor ID from log headings
SENSOR_VEHICLE_ID_RE = r"\[([0-9\-]+)\]\.GPS-vhlId"

class GNSSLog(SCANeRDataLog):
    """
    Class to parse and navigate GNSS log files extracted from SCANeR Studio.
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

        # Get sensor ids
        self.get_sensor_ids(df)

        # Get frames
        self.get_frames(df)

        # Parse vehicle ego poses
        self.parse_gnss_coordinates(df)

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
        self._sensId2index = dict(map(lambda i,j : (i,j) , self._sensor_ids, range(0, len(self.sensor_ids))))
        self._vehId2sensId = dict(map(lambda i,j : (i,j) , self._agent_ids, self._sensor_ids))

    def get_frames(self, df):
        """
        Extracts list of frames from the lidar log file.
        """
        self._frames = np.arange(0.0, df['time'].max() + self.sampling_period, self.sampling_period).round(2)

    def parse_gnss_coordinates(self, df):
        """
        Parses GNSS coordinates from the GNSS data log file.
        """

        # Compute column slices
        slices = [0]                            # Time slice
        start = 1+2*len(self._sensor_ids)
        end = start + 3*len(self._sensor_ids)
        slices.append(slice(start, end))        # Coordinates slice

        # Select bounding boxes columns for the given sensor
        selector = np.r_[*slices]
        df = df.iloc[:, selector]

        # Stack columns by timeframe
        df = df.set_index(['time']).stack(future_stack=True)
        
        # Flatten multi-index and rename columns
        df = df.reset_index().rename(columns={'level_1': 'field', 0: 'value'})
        
        # Extract fields for each row
        df['id'] = df['field'].str.extract(pat = r"^\[([0-9\-]+)\]")
        df['id'] = df['id'].apply(lambda x: self._sensId2vehId[x])
        df['field'] = df['field'].str.extract(pat = r"\[[0-9\-]+\]\.[^\-]+\-([^\.]+)")
        
        # Pivot table using timestamp and id as index and the values of parameter as columns
        df = df.pivot(index=['time', 'id'], columns='field').dropna()        

        # Set ego pose dataframe
        self._df_gnss= df

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
    # GNSS LOG QUERIES
    ###################################################
    
    def get_agent_gnss_at_frame(self, agent, frame):

        # Check if agent bboxes are present in log
        if not agent.id in self._vehId2sensId:
            return None

        # Convert sensing agent id to index
        index = self._sensId2index[self._vehId2sensId[agent.id]]

        # Convert frame number to time offset
        time = frame/self.sampling_freq

        # Extract agent GNSS coordinates at frame
        df = self._df_gnss.loc[(time, agent.id), :]
        altitude = df[('value', 'altitude')]
        latitude = df[('value', 'latitude')]
        longitude = df[('value', 'longitude')]

        # Return GNSS coordinates
        return np.array([altitude, latitude, longitude])
    
    def compute_valid_sampling_window(self, df):
        """
        Compute the start and end frame number of the valid time window of observations.
        """

        # Select only timeframe and time of update columns
        slices = [0]
        start = 1 + 6*len(self._sensor_ids)
        end = start + len(self._sensor_ids)
        slices.append(slice(start, end))
        selector = np.r_[*slices]
        df = df.iloc[:, selector]

        # Discard sampling times for which at least a vehicle has no updates
        df = df.set_index(['time']).dropna()

        # Set valid sampling window boundaries
        self.start_frame = round(df.index[0] / self.sampling_period)
        self.end_frame = round(df.index[-1] / self.sampling_period)
