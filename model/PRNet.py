import torch
from torch import nn
import math
import os
import argparse
import cv2
import numpy as np
from net.predictor import resfcn256
from loader.Dataset import TrainData


class PRNet(object):

    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.epochs = opt.epochs
        self.learning_rate = opt.learning_rate
        self.net = resfcn256()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

        #self.weights = cv2.imread("Data/uv-data/weight_mask_final.jpg")  # [256, 256, 3]
        self.weights = cv2.imread("Data/uv-data/map_xgtu.jpg")  # [256, 256, 3]
        self.weights_data = np.zeros([1, 256, 256, 3], dtype=np.float32)

    def setup(self,PATH):
        # load state_dict
        if os.path.exists(PATH):
            self.net.load_state_dict(torch.load(PATH))
        # load weight data
        weights = self.weights
        for i in range(13):
            for j in range(27):
                weights[i + 59, j + 78, :] = 250
        for i in range(13):
            for j in range(27):
                weights[i + 59, j + 153, :] = 250
        # cv2.imshow('weights',weights)
        # cv2.waitKey(0)
        weights_data = self.weights_data
        weights_data[0, :, :, :] = weights
        # [b, h, w, c] to [b, c, h, w]
        weights_data = np.swapaxes(weights_data, 1, 2)
        weights_data = np.swapaxes(weights_data, 1, 3)

        self.weights_data = torch.from_numpy(weights_data)

    def train(self):
        self.optimizer.zero_grad()

    def eval(self):
        self.net.eval()

    def optimize_parameters(self, x, label):
        x = np.array(x)
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x)
        label = np.swapaxes(label, 1, 2)
        label = np.swapaxes(label, 1, 3)
        label = torch.from_numpy(label)
        x_op = self.net.forward(x)
        loss = self.criterion(x_op, label)
        loss = loss.mul(self.weights_data)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, x, label):
        x = np.array(x)
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x)
        x_op = self.net.forward(x)
        label = np.swapaxes(label, 1, 2)
        label = np.swapaxes(label, 1, 3)
        label = torch.from_numpy(label)
        error = self.criterion(x_op, label)
        error = error.mean()
        loss = error.mul(self.weights_data)
        loss = loss.mean()
        x_op = x_op.numpy()
        x_op = np.swapaxes(x_op, 1, 3)
        x_op = np.swapaxes(x_op, 1, 2)
        return x_op, error, loss

    def generate(self, x):
        x = np.array(x)
        # forward without error
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x)
        x = torch.tensor(x, dtype=torch.float32)
        x_op = self.net.forward(x)
        x_op = x_op.detach().numpy()
        x_op = np.swapaxes(x_op, 1, 3)
        x_op = np.swapaxes(x_op, 1, 2)
        return x_op

    def save(self,save_path):
        torch.save(self.net.state_dict(), save_path)
