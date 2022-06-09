from enum import Enum, unique


@unique
class MLProblem(Enum):
    malignancy_prediction = "malignancy"
    nodule_type_prediction = "noduletype"