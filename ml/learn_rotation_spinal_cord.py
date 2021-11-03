import copy
import math
import os
from random import random

import numpy
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from environment.Environment import Environment
from environment.control import Control
from game_score import GameScore
from ml.CustomLogLoss import CustomLogLoss
from ml.RotationSpinalCordNetwork import RotationSpinalCordNetwork
from ml.SpinalCordNetwork import SpinalCordNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

modelFileName = 'rotation_spinal_cord_model.pth'

activationTreshold = 0.5


class RotationSpinalCordLearner(object):
    environment: Environment
    controls: [Control]
    scores: [GameScore]
    last_score: [GameScore]

    lastXs: [float]
    lastYs: [float]
    lastRewards: [float]
    lastHunger: float

    def __init__(self, environment: Environment, controls: [Control], scores: [GameScore]):
        self.environment = environment
        self.controls = controls
        self.scores = scores
        self.last_score = copy.deepcopy(scores)

        self.lastXs = []
        self.lastYs = []
        self.lastRewards = []
        self.lastHunger = environment.persons[0].hunger

        self.model = RotationSpinalCordNetwork().to(device)
        if os.path.isfile(modelFileName):
            self.model.load_state_dict(torch.load(modelFileName))
        # self.loss_fn = CustomLogLoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.MSELoss(reduction="sum")
        self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-5)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-3)

        self.custom_lay_counter = 1
        self.activation_heatmap = []

        self.model.register_forward_hook(self.get_activation())

        self.epochs = 0
        self.iter = 0

        self.angleDiff = 0
        self.testGameCount = 0

    def gameRestarted(self):
        self.lastHunger = self.environment.persons[0].hunger
        self.angleDiff = 0
        if self.testGameCount > 0:
            self.testGameCount = self.testGameCount - 1
            if self.testGameCount == 0:
                print(f"Test scores: \n Dies: {self.scores[0].die_count - self.last_score[0].die_count}, "
                      f"Got foods: {self.scores[0].get_food_count - self.last_score[0].get_food_count} \n")
            return

        if len(self.lastXs) > 1000:
            # Нужно пропустить несколько кадров
            # self.needSkipLearn = 30
            t = self.epochs
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(self.lastRewards, self.model, self.optimizer)
            # self.test(rewards, self.model)
            self.epochs = self.epochs + 1
            self.lastXs = []
            self.lastYs = []
            self.lastRewards = []
            # self.testGameCount = 1
            self.last_score = copy.deepcopy(self.scores)

    def learnLoop(self):
        person = self.environment.persons[0]
        food = self.environment.foods[0]

        dX = person.x - food.x
        dY = person.y - food.y
        targetAngle = math.degrees(math.atan(dX / dY))
        if person.y > food.y:
            targetAngle = targetAngle - 180
        if targetAngle < 0:
            targetAngle = 360 + targetAngle

        rotateDirection = -1
        angleDiff = targetAngle - person.movementAngle
        if angleDiff < 0:
            angleDiff = 360 + angleDiff
        if angleDiff > 180:
            rotateDirection = 1
            angleDiff = 360 - angleDiff
        if angleDiff == angleDiff:
            rotateDirection = 0

        orig_x = [
            angleDiff,
            1 if rotateDirection > 0 else 0,
            1 if rotateDirection < 0 else 0,
        ]
        X = torch.Tensor(np.array(orig_x)).float()
        X = X.to(device)

        if self.testGameCount > 0:
            return self.test(X)

        self.model.eval()
        # Compute prediction error
        pred = self.model(X)

        self.prediction_to_control(pred)

        pred_y = pred.detach().numpy()
        pred_y = [pred_y[0], pred_y[1]]

        learnSpeed = 0.5
        v2 = - learnSpeed  # * ((angleDiff + 50.0) / (180.0 + 50.0))
        if self.angleDiff > angleDiff:
            v2 = + learnSpeed  # * ((angleDiff + 50.0) / (180.0 + 50.0))
        if self.angleDiff == 0:
            v2 = 0

        v3 = v2

        rewards = [
            self.get_corrected_y(v2, pred_y[0]),
            self.get_corrected_y(v3, pred_y[1]),
        ]

        # rewards = [
        #     1 if v2 > 0 else 0,
        #     1 if v3 > 0 else 0,
        # ]
        # Награждаем пред шаг ибо его действия привели к текущему результату
        if len(self.lastRewards) > 0:
            self.lastRewards[len(self.lastRewards) - 1] = rewards

        self.lastXs.append(orig_x)
        self.lastYs.append(pred_y)
        self.lastRewards.append(pred_y)

        self.angleDiff = angleDiff

    def testLoop(self):
        print("need implement test_loop")
        pass

    def done(self):
        print("Done!")

        torch.save(self.model.state_dict(), modelFileName)
        print(f"Saved PyTorch Model State to {modelFileName}")

    def train(self, rewards, model, optimizer):
        size = len(rewards)
        model.train()

        X = torch.Tensor(np.array(self.lastXs)).float()
        y = torch.Tensor(np.array(self.lastYs)).float()
        rewards = torch.Tensor(np.array(rewards)).float()
        X, y, rewards = X.to(device), y.to(device), rewards.to(device)
        # y_tmp = torch.Tensor(y).long().to(device)

        # Compute prediction error
        pred = model(X)

        loss = self.loss_fn(pred, rewards)

        lossV = loss.item()
        if math.isnan(lossV):
            print("pred: ")
            print(pred)
            # print("prob: ")
            # print(prob)
            # print("selected_probs: ")
            # print(selected_probs)
            exit(1)

        if math.isinf(lossV):
            print("pred: ")
            print(pred)
            # print("prob: ")
            # print(prob)
            # print("selected_probs: ")
            # print(selected_probs)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predY = pred.detach().numpy().mean(axis=0)
        rewardY = rewards.detach().numpy().mean(axis=0)
        lossV = loss.item()
        self.scores[0].loss = lossV
        print(
            f"loss: {lossV:>7f}  [{size:>5d}]\n"
            f"last prediction: [{predY[0]}, {predY[1]}]\n"
            f"last reward: [{rewardY[0]}, {rewardY[1]}]")

    def test(self, X):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            pred = self.model(X)

            self.prediction_to_control(pred)

            # test_loss += self.loss_fn(pred, y).item()
            # n_digits = 3
            # roundedPred = torch.round(pred * 10 ** n_digits) / (10 ** n_digits)
            # roundedY = torch.round(y * 10 ** n_digits) / (10 ** n_digits)
            # correct += (roundedPred.argmax(1) == roundedY).type(torch.float).sum().item()
        # test_loss /= num_batches
        # correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def prediction_to_control(self, pred):
        self.controls[0].moveForward = True
        self.controls[0].moveBack = False
        self.controls[0].rotateLeft = False
        self.controls[0].rotateRight = False

        if pred[0] > pred[1]:
            self.controls[0].rotateLeft = True
        if pred[1] > pred[0]:
            self.controls[0].rotateRight = True

    # Дисконтированная награда
    def get_corrected_y(self, v: float, predY: float, gamma=0.98) -> float:
        running_add = self.calcRewardFunc(v)
        running_add_negative = 2 - running_add

        max_val = 0.60
        min_val = 0.40

        vt = predY
        it = predY
        if it > 0.5:
            vt = it * running_add
        else:
            vt = (it if it >= min_val else min_val) * running_add_negative

        if vt > max_val:
            vt = max_val

        if vt < min_val:
            vt = min_val

        return vt

    def calcRewardFunc(self, v: float) -> float:
        return 1.0 + v

    def get_activation(self):
        def hook(model: nn.Module, input: Tensor, output: Tensor):
            if model.training or len(output) > 1:
                return
            if model.custom_lay_number is None:
                model.custom_lay_number = self.custom_lay_counter
                self.custom_lay_counter = self.custom_lay_counter + 1
            o = output.detach().numpy()[0]
            if self.activation_heatmap[model.custom_lay_number] is None:
                self.activation_heatmap[model.custom_lay_number] = numpy.array([0.0]*len(o))
            v = self.activation_heatmap[model.custom_lay_number]
            v = v + o

        return hook
