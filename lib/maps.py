import shortuuid

class Map:

    def __init__(self, name, description, mask):

        # Initialize map
        self.token = shortuuid.uuid()
        self.name = name
        self.description = description
        self.mask = mask
        self.logs = []

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"Map(token='{self.token}', name='{self.name}', logs={self.logs})"

    def __repr__(self):
        return f"Map(token='{self.token}', name='{self.name}', logs={self.logs})"
