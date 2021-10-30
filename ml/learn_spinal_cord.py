import math
import os
from random import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from environment.Environment import Environment
from environment.control import Control
from ml.CustomLogLoss import CustomLogLoss
from ml.SpinalCordNetwork import SpinalCordNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class SpinalCordLearner(object):
    environment: Environment
    controls: [Control]

    lastXs: [float]
    lastYs: [float]
    lastRewards: [float]
    lastHunger: float

    def __init__(self, environment: Environment, controls: [Control]):
        self.environment = environment
        self.controls = controls

        self.lastXs = []
        self.lastYs = []
        self.lastRewards = []
        self.lastHunger = environment.persons[0].hunger

        self.model = SpinalCordNetwork().to(device)
        if os.path.isfile("spinal_cord_model.pth"):
            self.model.load_state_dict(torch.load("spinal_cord_model.pth"))
        # self.loss_fn = CustomLogLoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-3)

        self.epochs = 0
        self.iter = 0

        self.lastDistance = 0
        self.angleDiff = 0
        self.lastExpPositive = True

    def gameRestarted(self):
        self.lastHunger = self.environment.persons[0].hunger
        self.lastDistance = 0
        self.angleDiff = 0
        self.lastExpPositive = True

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

        targetSpeed = 3
        if angleDiff >= 45:
            targetSpeed = -2

        distance = math.sqrt((person.x - food.x) ** 2 + (person.y - food.y) ** 2)

        origX = [
            person.movementAngle / 360,
            targetAngle / 360,
            rotateDirection,
            person.movementSpeed / 3,
            targetSpeed / 3,
            # distance / 500,
            # person.hunger / 100
        ]
        X = torch.Tensor(np.array(origX)).float()
        X = X.to(device)

        # Compute prediction error
        pred = self.model(X)

        self.controls[0].moveForward = False
        self.controls[0].moveBack = False
        self.controls[0].rotateLeft = False
        self.controls[0].rotateRight = False

        activationTreshold = 0.5
        if pred[0] > activationTreshold and pred[1] < pred[0]:
            self.controls[0].moveForward = True
        if pred[1] > activationTreshold and pred[0] < pred[1]:
            self.controls[0].moveBack = True
        if pred[2] > activationTreshold and pred[3] < pred[2]:
            self.controls[0].rotateLeft = True
        if pred[3] > activationTreshold and pred[2] < pred[3]:
            self.controls[0].rotateRight = True

        self.lastXs.append(origX)
        predY = pred.detach().numpy()
        predY = [predY[0], predY[1], predY[2], predY[3]]
        self.lastYs.append(predY)

        learnSpeed = 0.1
        v = - learnSpeed * (distance / 500.0)
        expPositive = False
        if self.lastDistance > distance:
            v = + learnSpeed * ((500.0 - distance) / 500.0)
            expPositive = True

        v2 = - learnSpeed * (angleDiff / 180.0)
        if self.angleDiff > angleDiff or angleDiff == 0:
            v2 = + learnSpeed * ((180 - angleDiff) / 180.0)

        # Угол быстро меняется если мы близко к цели и двигаемся быстро, не зависимо от самого поворота
        # if distance < 40 and person.movementSpeed > 0.8:
        #     v2 = 0

        # Если у нас активировалось и влево и вправо одновремено то снижаем
        # if v2 > 0 and angleDiff > 3 and len(self.lastRewards) > 0 \
        #         and self.lastRewards[len(self.lastRewards) - 1][2] > activationTreshold \
        #         and self.lastRewards[len(self.lastRewards) - 1][3] > activationTreshold:
        #     v2 = -v2

        rewards = self.discountCorrectRewards(v, v2, predY)
        if len(self.lastRewards) > 0:
            self.lastRewards[len(self.lastRewards) - 1] = rewards
        self.lastRewards.append(predY)

        if len(self.lastXs) > 1 and self.lastExpPositive != expPositive or len(self.lastXs) > 200:
            self.lastExpPositive = expPositive
            t = self.epochs
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(self.lastRewards, self.model, self.optimizer)
            # self.test(rewards, self.model)
            self.epochs = self.epochs + 1
            self.lastXs = []
            self.lastYs = []
            self.lastRewards = []
        self.lastHunger = person.hunger
        self.lastDistance = distance

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
        y_tmp = torch.Tensor(y).long().to(device)

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

    def test(self, dataloader, model):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += self.loss_fn(pred, y).item()
                n_digits = 3
                roundedPred = torch.round(pred * 10 ** n_digits) / (10 ** n_digits)
                roundedY = torch.round(y * 10 ** n_digits) / (10 ** n_digits)
                correct += (roundedPred.argmax(1) == roundedY).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def discountCorrectRewards(self, v: float, v2: float, predY: [float], gamma=0.98) -> [
        float]:  # Дисконтированная награда
        """ take 1D float array of rewards and compute discounted reward """
        running_add = self.calcRewardFunc(v)
        running_add_negative = 2 - running_add

        running_add2 = self.calcRewardFunc(v2)
        running_add_negative2 = 2 - running_add2

        vt = [0, 0, 0, 0]
        it = predY
        if it[0] > 0.5:
            vt[0] = it[0] * running_add
        else:
            vt[0] = it[0] * running_add_negative
        if it[1] > 0.5:
            vt[1] = it[1] * running_add
        else:
            vt[1] = it[1] * running_add_negative

        if it[2] > 0.5:
            vt[2] = it[2] * running_add2
        else:
            vt[2] = it[2] * running_add_negative2
        if it[3] > 0.5:
            vt[3] = it[3] * running_add2
        else:
            vt[3] = it[3] * running_add_negative2

        max = 0.95
        if vt[0] > max:
            vt[0] = max
        if vt[1] > max:
            vt[1] = max
        if vt[2] > max:
            vt[2] = max
        if vt[3] > max:
            vt[3] = max

        min = 0.05
        if vt[0] < min:
            vt[0] = min
        if vt[1] < min:
            vt[1] = min
        if vt[2] < min:
            vt[2] = min
        if vt[3] < min:
            vt[3] = min

        return vt

    def calcRewardFunc(self, v: float) -> float:
        return 1.0 + v
