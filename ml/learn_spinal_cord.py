import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from environment.Environment import Environment
from environment.control import Control
from ml.SpinalCordNetwork import SpinalCordNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class SpinalCordLearner(object):
    environment: Environment
    controls: [Control]

    lastXs: [float]
    lastYs: [float]
    lastHunger: float

    def __init__(self, environment: Environment, controls: [Control]):
        self.environment = environment
        self.controls = controls

        self.lastXs = []
        self.lastYs = []
        self.lastHunger = environment.persons[0].hunger

        self.model = SpinalCordNetwork().to(device)
        if os.path.isfile("spinal_cord_model.pth"):
            self.model.load_state_dict(torch.load("spinal_cord_model.pth"))
        self.loss_fn = nn.L1Loss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-3)

        self.epochs = 0
        self.iter = 0

    def learnLoop(self):
        person = self.environment.persons[0]
        food = self.environment.foods[0]

        targetAngle = math.degrees(math.atan((person.x - food.x) / (person.y - food.y)))
        if targetAngle < 0:
            targetAngle = 360 - targetAngle

        rotateDirection = 1
        angleDiff = targetAngle - person.movementAngle
        if angleDiff < 0:
            angleDiff = 360 - angleDiff
        if angleDiff < 180:
            rotateDirection = -1
        else:
            angleDiff = angleDiff - 180

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
            distance / 500,
            person.hunger / 100
        ]
        X = torch.Tensor(np.array(origX)).float()
        X = X.to(device)

        # Compute prediction error
        pred = self.model(X)

        self.controls[0].moveForward = False
        self.controls[0].moveBack = False
        self.controls[0].rotateLeft = False
        self.controls[0].rotateRight = False

        if pred[0] > 0.2 and pred[1] < pred[0]:
            self.controls[0].moveForward = True
        if pred[1] > 0.2 and pred[0] < pred[1]:
            self.controls[0].moveBack = True
        if pred[2] > 0.2 and pred[3] < pred[2]:
            self.controls[0].rotateLeft = True
        if pred[3] > 0.2 and pred[2] < pred[3]:
            self.controls[0].rotateRight = True

        self.lastXs.append(origX)
        predY = pred.detach().numpy()
        self.lastYs.append([predY[0], predY[1], predY[2], predY[3]])

        # началась новая игра
        # if person.hunger == 10 and self.lastHunger > 90:
        #     self.lastHunger = person.hunger
        #     self.lastXs = []
        #     self.lastYs = []

        #  or person.hunger > 99.9
        if self.lastHunger > person.hunger:
            v = -4.0 * (distance / 500.0)
            if self.lastHunger > person.hunger and not (person.hunger == 10 and self.lastHunger > 90):
                v = 4

            # maxCount = 400
            # if len(self.lastXs) > maxCount:
            #     self.lastXs = self.lastXs[len(self.lastXs) - maxCount:len(self.lastXs)]
            #     self.lastYs = self.lastYs[len(self.lastYs) - maxCount:len(self.lastYs)]

            # np.argmax(self.lastYs, axis=1)
            rewards = self.discountCorrectRewards(v, self.lastYs)

            if len(self.lastXs) > 1:
                t = self.epochs
                print(f"Epoch {t + 1}\n-------------------------------")
                self.train(rewards, self.model, self.loss_fn, self.optimizer)
                # self.test(rewards, self.model, self.loss_fn)
                self.epochs = self.epochs + 1
            self.lastXs = []
            self.lastYs = []
        self.lastHunger = person.hunger

    def done(self):
        print("Done!")

        torch.save(self.model.state_dict(), "spinal_cord_model.pth")
        print("Saved PyTorch Model State to spinal_cord_model.pth")

    def train(self, rewards, model, loss_fn, optimizer):
        size = len(rewards)
        model.train()

        X = torch.Tensor(np.array(self.lastXs)).float()
        y = torch.Tensor(np.array(self.lastYs)).float()
        rewards = torch.Tensor(np.array(rewards)).float()
        X, y, rewards = X.to(device), y.to(device), rewards.to(device)
        y_tmp = torch.Tensor(y).long().to(device)

        # Compute prediction error
        pred = model(X)

        # print(pred)
        # print(rewards)
        # print(self.lastYs)
        # print(np.arange(len(y)))
        # print(y_tmp)
        # print(pred[np.arange(len(y)), y_tmp])

        # Этап 1 - логарифмируем вероятности действий
        # prob = torch.log(pred[np.arange(len(y)), y])
        prob = torch.log(pred)
        # Этап 2 - отрицательное среднее произведения вероятностей на награду
        selected_probs = rewards * prob
        loss = -selected_probs.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossV = loss.item()
        print(f"loss: {lossV:>7f}  [{size:>5d}]")

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                n_digits = 3
                roundedPred = torch.round(pred * 10 ** n_digits) / (10 ** n_digits)
                roundedY = torch.round(y * 10 ** n_digits) / (10 ** n_digits)
                correct += (roundedPred.argmax(1) == roundedY).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def discountCorrectRewards(self, v: float, indexes: np.array, gamma=0.98) -> [float]:  # Дисконтированная награда
        """ take 1D float array of rewards and compute discounted reward """
        count = len(indexes)
        vals = [0.0] * count
        running_add = v
        for t in reversed(range(count)):
            vt = [0, 0, 0, 0]
            i1 = 0
            if indexes[t][0] < indexes[t][1]:
                i1 = 1
            i2 = 2
            if indexes[t][2] < indexes[t][3]:
                i2 = 3
            vt[i1] = running_add
            vt[i2] = running_add
            vals[t] = vt
            running_add = running_add * gamma

        # discounted_r -= discounted_r.mean()
        # discounted_r /- discounted_r.std()
        return vals
