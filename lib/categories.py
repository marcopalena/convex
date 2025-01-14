import shortuuid

class Category:

    def __init__(self, name, description, type):

        # Initialize category
        self.token = shortuuid.uuid()
        self.name = name
        self.category_tokens = self.name.split(".")
        self.category_tokens += [None] * (3 - len(self.category_tokens))
        self.description = description
        self.type = type

    ###################################################
    # PROPERTIES
    ###################################################

    @property
    def supercategory(self):
        return self.category_tokens[0]
    
    @property
    def category(self):
        return self.category_tokens[1]
    
    @property
    def subcategory(self):
        return self.category_tokens[2]

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"Category(token='{self.token}', name='{self.name}', type={self.type})"

    def __repr__(self):
        return f"Category(token='{self.token}', name='{self.name}', type={self.type})"
    

class Attribute:

    def __init__(self, name, description):

        # Initialize attribute
        self.token = shortuuid.uuid()
        self.name = name
        self.description = description

    ###################################################
    # MAGIC METHODS
    ###################################################

    def __str__(self):
        return f"Attribute(token='{self.token}', name='{self.name}')"

    def __repr__(self):
        return f"Attribute(token='{self.token}', name='{self.name}')"