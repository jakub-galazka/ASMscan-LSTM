class RepFold:

    def __init__(self, name, color=None, cut_type_rule=None):
        self.name = name
        self.color = color
        self.cut_type_rule = cut_type_rule

    def setScope(self, scope):
        self.scope = scope
        