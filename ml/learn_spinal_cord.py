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
from ml.SpinalCordNetwork import SpinalCordNetwork
from ml.base_learner import BaseLearner

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

activationTreshold = 0.5


class SpinalCordBaseLearner(BaseLearner):
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

        self.model = SpinalCordNetwork().to(device)
        if os.path.isfile("spinal_cord_model.pth"):
            self.model.load_state_dict(torch.load("spinal_cord_model.pth"))
        # self.loss_fn = CustomLogLoss()
        # self.loss_fn = nn.L1Loss()
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.MSELoss(reduction="sum")
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-3)

        self.epochs = 0
        self.iter = 0

        self.lastDistance = 0
        self.angleDiff = 0
        self.needSkipLearn = 0
        self.lastExpPositive = True
        self.testGameCount = 0

    def gameRestarted(self):
        self.lastHunger = self.environment.persons[0].hunger
        self.lastDistance = 0
        self.angleDiff = 0
        self.needSkipLearn = 0
        self.lastExpPositive = True
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
            self.testGameCount = 1
            self.last_score = copy.deepcopy(self.scores)

    def learnLoop(self):
        person = self.environment.persons[0]
        food = self.environment.foods[0]

        distance = math.sqrt((person.x - food.x) ** 2 + (person.y - food.y) ** 2)

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
        if angleDiff == 0:
            rotateDirection = 0

        targetSpeed = 3
        if angleDiff >= 45 and distance > 100 or angleDiff >= 45 * (distance / 100):
            targetSpeed = -2

        if targetSpeed > 0 and distance < 150:
            targetSpeed = 2 * (distance / 150)

        orig_x = [
            # angleDiff,
            angleDiff / 180 if rotateDirection < 0 else 0,
            angleDiff / 180 if rotateDirection > 0 else 0,
            person.movementSpeed / 3,
            targetSpeed / 3 if targetSpeed > 0 else 0,
            -targetSpeed / 3 if targetSpeed < 0 else 0,
            distance / 500,
            # person.hunger / 100
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
        pred_y = [pred_y[0], pred_y[1], pred_y[2], pred_y[3]]

        learnSpeed = 0.5
        v0 = - learnSpeed * (distance / 500.0)
        if (self.lastDistance > distance or self.lastDistance == 0) and person.movementSpeed != 0:
            v0 = + learnSpeed * ((500.0 - distance) / 500.0)

        if self.needSkipLearn > 0:
            self.needSkipLearn = self.needSkipLearn - 1
            # Почти не наказываем если ошибаться начал только что
            # v0 = v0 * 0.001

        v1 = v0

        # Получаем пред шаг ибо его действия привели к текущему результату
        last_pred_y = pred_y
        if len(self.lastYs) > 0:
            last_pred_y = self.lastYs[len(self.lastYs) - 1]

        rewards = [
            self.get_corrected_y(v0, last_pred_y[0]),
            self.get_corrected_y(v1, last_pred_y[1]),
        ]

        # Награждаем пред шаг ибо его действия привели к текущему результату
        if len(self.lastRewards) > 0:
            self.lastRewards[len(self.lastRewards) - 1] = rewards

        self.lastXs.append(orig_x)
        self.lastYs.append(pred_y)
        self.lastRewards.append(pred_y)

        self.lastHunger = person.hunger
        self.lastDistance = distance
        self.angleDiff = angleDiff

    def testLoop(self):
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

        targetSpeed = 3
        if angleDiff >= 90:
            targetSpeed = -2

        distance = math.sqrt((person.x - food.x) ** 2 + (person.y - food.y) ** 2)

        origX = [
            angleDiff,
            person.movementSpeed / 3,
            targetSpeed / 3 if targetSpeed > 0 else 0,
            -targetSpeed / 3 if targetSpeed < 0 else 0,
            # distance / 500,
            # person.hunger / 100
        ]
        X = torch.Tensor(np.array(origX)).float()
        X = X.to(device)
        self.test(X)

    def done(self):
        print("Done!")

        torch.save(self.model.state_dict(), "spinal_cord_model.pth")
        print("Saved PyTorch Model State to spinal_cord_model.pth")

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
        print(
            f"loss: {lossV:>7f}  [{size:>5d}]\nlast prediction: [{predY[0]}, {predY[1]}, {predY[2]}, {predY[3]}]\nlast reward: [{rewardY[0]}, {rewardY[1]}, {rewardY[2]}, {rewardY[3]}]")

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
        self.controls[0].moveForward = False
        self.controls[0].moveBack = False
        self.controls[0].rotateLeft = False
        self.controls[0].rotateRight = False

        if pred[0] > activationTreshold and pred[1] < pred[0]:
            self.controls[0].moveForward = True
        if pred[1] > activationTreshold and pred[0] < pred[1]:
            self.controls[0].moveBack = True
        if pred[2] > activationTreshold and pred[3] < pred[2]:
            self.controls[0].rotateLeft = True
        if pred[3] > activationTreshold and pred[2] < pred[3]:
            self.controls[0].rotateRight = True
