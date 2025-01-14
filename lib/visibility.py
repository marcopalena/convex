import shortuuid

class VisibilityLevel:

    def __init__(self, name, description, min_val, max_val):

        # Initialize visibility level
        self.token = name
        self.description = description
        self.min = min_val
        self.max = max_val

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"VisibilityLevel(token='{self.token}', min={self.min}, max={self.max})"

    def __repr__(self):
        return f"VisibilityLevel(token='{self.token}', min={self.min}, max={self.max})"