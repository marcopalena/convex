import os
import re
import cv2
import shortuuid

###################################################
# CONSTANTS
###################################################

DEFAULT_SCANER_SAMPLING_FREQ = 20           # Default frequency at which SCANeR Studio collects data (Hz) 

class Log:
    """
    A class to represent log files from which the data is extracted.
    """
        
    def __init__(self, path):

        # Initialize log
        self.token = shortuuid.uuid()
        self.path = path

class VideoLog(Log):
    """
    A class to represent video log files from which frames are extracted.
    """

    def __init__(self, path, agent):

        # Initialize video log
        super().__init__(path)
        self.agent = agent
        self.vidcap = None

        # Initialize video metadata
        self.init_capture()
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_capture()

    def init_capture(self):
        self.vidcap = cv2.VideoCapture(self.path)

    def end_capture(self):
        self.vidcap.release()
        self.vidcap = None

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"VideoLog(token='{self.token}', path='{self.path}', agent_id={self.agent.id}, agent_type={self.agent.type}, fps={self.fps}, num_frames={self.num_frames})"

    def __repr__(self):
        return f"VideoLog(token='{self.token}', path='{self.path}', agent_id={self.agent.id}, agent_type={self.agent.type}, fps={self.fps}, num_frames={self.num_frames})"


class SCANeRDataLog(Log):
    """
    A class to represent text log files extracted from SCANeR Studio.
    """
    def __init__(self, path, sampling_freq = DEFAULT_SCANER_SAMPLING_FREQ):

        # Initialize SCANeR data log
        super().__init__(path)
        self.sampling_freq = sampling_freq
        self.sampling_period = 1/self.sampling_freq