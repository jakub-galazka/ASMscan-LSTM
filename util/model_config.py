import json

from config import MODEL_CONFIG_PATH


class ModelConfig:

    def __init__(self, model_id):
        with open(MODEL_CONFIG_PATH % model_id, "r") as f:
            self.model_config = json.load(f)

    def getParam(self, param_key):
        return self.model_config[param_key]
