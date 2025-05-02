import os
from pathlib import Path

import torch

EVENTS_TYPE = "empe"
TASK_TYPE = "b_pene"
TASK_TYPE_TO_TARGET = {1: 0, 3: 1}
PROJECT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent
HEIGHT_3D_MATRIX = 9
WIDTH_3D_MATRIX = 9

HBO = "hbo"
HBR = "hbr"

SAMPLING_POINTS_T = 60

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0 if torch.cuda.is_available() else 0
PIN_MEMORY = True if torch.cuda.is_available() else False
VALID_DATASETS = ("nemo", "aomic")


def get_dataset_key(dataset: str) -> str:
    assert dataset in VALID_DATASETS
    return dataset