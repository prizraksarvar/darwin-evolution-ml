import copy
import math
import os
from random import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from environment.Environment import Environment
from environment.control import Control
from game_score import GameScore
from ml.CustomLogLoss import CustomLogLoss
from ml.RotationSpinalCordNetwork import RotationSpinalCordNetwork
from ml.SpinalCordNetwork import SpinalCordNetwork
from ml.base_learner import BaseLearner

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

activationTreshold = 0.5

modelFileName = 'spinal_cord_model.pth'


class TestEdit(object):
    def __init__(self):
        rotation_model_path = "../models/rotation_spinal_cord_model.pth"
        self.rotation_model = RotationSpinalCordNetwork().to(device)
        if not os.path.isfile(rotation_model_path):
            print(f"FATAL: model not found {rotation_model_path}")
        self.rotation_model.load_state_dict(torch.load(rotation_model_path))
        self.rotation_model.eval()
        print(self.rotation_model.state_dict())

    def done(self):
        print("Done!")

        torch.save(self.rotation_model.state_dict(), "../models/rotation_spinal_cord_model2.pth")
        print(f"Saved PyTorch Model State to models/rotation_spinal_cord_model2.pth")


t = TestEdit()
