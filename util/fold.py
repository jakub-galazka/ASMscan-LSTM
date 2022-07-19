class Fold:

    def __init__(self, name, cut_type_rule=None):
        self.name = name
        self.cut_type_rule = cut_type_rule

    def setScope(self, scope):
        self.scope = scope
        