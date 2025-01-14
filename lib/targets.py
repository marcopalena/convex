class Target:
    
    def __init__(self, id, is_moving, scaner_type, category):
        self.id = id
        self.is_moving = is_moving
        self.scaner_type = scaner_type
        self.category = category

    def is_static(self):
        return not self.is_moving
    
    def is_dynamic(self):
        return self.is_moving
    
class StaticTarget(Target):

    def __init__(self, id, scaner_type, category):
        super().__init__(id, False, scaner_type, category) 

class DynamicTarget(Target):
    
    def __init__(self, id, scaner_type, category):
        super().__init__(id, True, scaner_type, category) 

class Infrastructure(StaticTarget):

    def __init__(self, id, name, scaner_type, category):
        super().__init__(id, scaner_type, category)
        self.name = name

    def __str__(self):
        return f"Infrastructure(id={self.id}, name='{self.name}', category='{self.category.name}')"

    def __repr__(self):
        return f"Infrastructure(id={self.id}, name='{self.name}', category='{self.category.name}')"

class Vehicle(DynamicTarget):
    
    def __init__(self, id, name, scaner_type, category, dimensions):
        super().__init__(id, scaner_type, category)
        self.name = name
        self.dimensions = dimensions
        self.vulnerable = True if category.supercategory == "vulnerable_vehicle" else False

    def is_vulnerable(self):
        return self.vulnerable

    def __str__(self):
        return f"Vehicle(id={self.id}, name='{self.name}', category='{self.category.name}', vulnerable={self.vulnerable})"

    def __repr__(self):
        return f"Vehicle(id={self.id}, name='{self.name}', category='{self.category.name}', vulnerable={self.vulnerable})"

class Pedestrian(DynamicTarget):

    def __init__(self, id, name, scaner_type, category, visibility):
        super().__init__(id, scaner_type, category) 
        self.name = name
        self.visibility = visibility

    def __str__(self):
        return f"Pedestrian(id={self.id}, name='{self.name}', category='{self.category.name}')"

    def __repr__(self):
        return f"Pedestrian(id={self.id}, name='{self.name}', category='{self.category.name}')"