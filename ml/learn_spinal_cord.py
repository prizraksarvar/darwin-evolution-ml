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


class SpinalCordLearner(BaseLearner):
    environment: Environment
    controls: [Control]
    scores: [GameScore]
    last_score: [GameScore]

    lastXs: [float]
    lastYs: [float]
    lastRewards: [float]
    last_hunger: float

    def __init__(self, environment: Environment, controls: [Control], scores: [GameScore]):
        self.environment = environment
        self.controls = controls
        self.scores = scores
        self.last_score = copy.deepcopy(scores)

        self.lastXs = []
        self.lastYs = []
        self.lastRewards = []
        self.last_hunger = environment.persons[0].hunger

        rotation_model_path = "models/rotation_spinal_cord_model.pth"
        self.rotation_model = RotationSpinalCordNetwork().to(device)
        if not os.path.isfile(rotation_model_path):
            print(f"FATAL: model not found {rotation_model_path}")
        self.rotation_model.load_state_dict(torch.load(rotation_model_path))
        self.rotation_model.eval()

        self.model = SpinalCordNetwork().to(device)
        if os.path.isfile(modelFileName):
            self.model.load_state_dict(torch.load(modelFileName))
        # self.loss_fn = CustomLogLoss()
        self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.MSELoss(reduction="sum")
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-3)

        self.epochs = 0
        self.iter = 0

        self.last_distance = 0
        self.target_speed = 0
        self.last_person_speed = 0
        self.angle_diff = 0
        self.need_skip_learn = 0
        self.last_exp_positive = True

        # TODO: времено включил тестовые игры
        self.test_game_count = 0

    def gameRestarted(self):
        self.last_distance = 0
        self.target_speed = 0
        self.last_person_speed = 0
        self.angle_diff = 0
        self.need_skip_learn = 0
        self.last_exp_positive = True
        if self.test_game_count > 0:
            self.test_game_count = self.test_game_count - 1
            if self.test_game_count == 0:
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
            self.test_game_count = 0  # Пока не работает нужно подумать как сделать
            self.last_score = copy.deepcopy(self.scores)

    def learnLoop(self):
        person = self.environment.persons[0]
        food = self.environment.foods[0]

        distance = math.sqrt((person.x - food.x) ** 2 + (person.y - food.y) ** 2)

        dX = person.x - food.x
        dY = person.y - food.y
        target_angle = math.degrees(math.atan(dX / dY))
        if person.y > food.y:
            target_angle = target_angle - 180
        if target_angle < 0:
            target_angle = 360 + target_angle

        rotate_direction = -1
        angle_diff = target_angle - person.movementAngle
        if angle_diff < 0:
            angle_diff = 360 + angle_diff
        if angle_diff > 180:
            rotate_direction = 1
            angle_diff = 360 - angle_diff
        if angle_diff == 0:
            rotate_direction = 0

        target_speed = 3
        if angle_diff >= 45 and distance > 100 or angle_diff >= 45 * (distance / 100):
            target_speed = -2

        if target_speed > 0 and distance < 150:
            target_speed = 2 * ((distance + 50) / (150 + 50))

        orig_x = [
            self.normilize_target_speed(target_speed) if target_speed > 0 else 0,
            self.normilize_target_speed(-target_speed) if target_speed < 0 else 0,
            person.movementSpeed / 3,
        ]
        X = torch.Tensor(np.array(orig_x)).float()
        X = X.to(device)

        rotation_X = torch.Tensor(np.array([
            self.normilize_angle_diff(angle_diff) if rotate_direction < 0 else 0.0,
            self.normilize_angle_diff(angle_diff) if rotate_direction > 0 else 0.0,
        ])).float()

        rotation_X = rotation_X.to(device)

        if self.test_game_count > 0:
            return self.test(X, rotation_X)

        self.model.eval()
        # Compute prediction error
        pred = self.model(X)

        rotation_pred = self.rotation_model(rotation_X)

        rotation_pred_y = rotation_pred.detach().numpy()
        rotation_pred_y = [rotation_pred_y[0], rotation_pred_y[1]]

        pred_y = pred.detach().numpy()
        pred_y = [pred_y[0], pred_y[1]]

        self.prediction_to_control([pred_y[0], pred_y[1], rotation_pred_y[0], rotation_pred_y[1]])

        ab_target_speed = math.fabs(target_speed)

        learn_speed = 0.5
        v0 = - learn_speed * self.normilize_target_speed(math.fabs(self.target_speed - person.movementSpeed))
        if self.last_person_speed < person.movementSpeed and self.target_speed > self.last_person_speed \
                or self.last_person_speed > person.movementSpeed and self.target_speed < self.last_person_speed:
            v0 = + learn_speed * self.normilize_target_speed(math.fabs(self.target_speed - person.movementSpeed))

        if self.need_skip_learn > 0:
            self.need_skip_learn = self.need_skip_learn - 1
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

        self.last_distance = distance
        self.target_speed = target_speed
        self.last_person_speed = person.movementSpeed
        self.angle_diff = angle_diff

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

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predY = pred.detach().numpy().mean(axis=0)
        rewardY = rewards.detach().numpy().mean(axis=0)
        lossV = loss.item()

        self.scores[0].loss = lossV
        print(
            f"loss: {lossV:>7f}  [{size:>5d}]\nlast prediction: [{predY[0]}, {predY[1]}]\n"
            f"last reward: [{rewardY[0]}, {rewardY[1]}]")

    def test(self, X, rotation_X):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)

            rotation_pred = self.rotation_model(rotation_X)

            rotation_pred_y = rotation_pred.detach().numpy()
            rotation_pred_y = [rotation_pred_y[0], rotation_pred_y[1]]

            pred_y = pred.detach().numpy()
            pred_y = [pred_y[0], pred_y[1]]

            self.prediction_to_control([pred_y[0], pred_y[1], rotation_pred_y[0], rotation_pred_y[1]])

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

    def normilize_target_speed(self, target_speed: float) -> float:
        padding = 0.4
        return (target_speed + padding) / (3.0 + padding)

    def normilize_distance(self, distance: float) -> float:
        padding = 50.0
        return (distance + padding) / (500.0 + padding)

    def normilize_angle_diff(self, angle_diff: float) -> float:
        padding = 80.0
        return (angle_diff + padding) / (180.0 + padding)
