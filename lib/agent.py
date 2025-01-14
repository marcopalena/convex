from .sensors import SensorModality

class Agent:
    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.calibrated_camera = None
        self.calibrated_lidar = None
        self.calibrated_gnss = None

    def has_lidar_sensor(self):
        return not self.calibrated_lidar is None
    
    def has_camera_sensor(self):
        return not self.calibrated_camera is None
    
    def has_gnss_sensor(self):
        return not self.calibrated_gnss is None

class VehicleAgent(Agent):
    def __init__(self, id, vehicle):
        super().__init__(id, "vehicle")
        self.vehicle = vehicle

    def __repr__(self):
        return f"VehicleAgent(id={self.id})"

class RoadsideAgent(Agent):
    def __init__(self, id, pedestrian):
        super().__init__(id, "rsu")
        self.pedestrian = pedestrian

    def __repr__(self):
        return f"RoadsideAgent(id={self.id})"