class RepFold:

    def __init__(self, name, color, separate=False, rule=False):
        self.name = name
        self.color = color
        self.separate = separate
        self.rule = rule

    def setScope(self, scope):
        self.scope = scope

    def get_class(self, id):
        if type(self.rule) == int:
            return id[:self.rule]
        