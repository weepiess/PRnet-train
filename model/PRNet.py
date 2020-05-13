import torch
from torch import nn
import math
import os
import argparse
import cv2
import numpy
from net.predictor import resfcn256
from loader.Dataset import TrainData


class PRNet(object):

    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.epochs = opt.epochs
        self.learning_rate = opt.learning_rate
        self.net = resfcn256(*args, **kwargs)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

        self.weights = cv2.imread("Data/uv-data/weight_mask_final.jpg")  # [256, 256, 3]
        self.weights_data = np.zeros([1, 256, 256, 3], dtype=np.float32)

    def setup(self,PATH):
        # load state_dict
        self.net.load_state_dict(torch.load(PATH))
        # load weight data
        for i in range(13):
            for j in range(27):
                self.weights[i + 59, j + 78, :] = 250
        for i in range(13):
            for j in range(27):
                self.weights[i + 59, j + 153, :] = 250
        # cv2.imshow('weights',weights)
        # cv2.waitKey(0)
        self.weights_data[0, :, :, :] = self.weights
        # [b, h, w, c] to [b, c, h, w]
        self.weights_data = np.swapaxes(self.weights_data, 1, 2)
        self.weights_data = np.swapaxes(self.weights_data, 1, 3)

        self.weights_data = torch.from_numpy(self.weights_data)

    def train(self):
        self.optimizer.zero_grad()

    def eval(self):
        self.net.eval()

    def optimize_parameters(self, x, label):
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x)
        label = np.swapaxes(label, 1, 2)
        label = np.swapaxes(label, 1, 3)
        label = torch.from_numpy(label)
        x_op = self.net(x)
        loss = self.criterion(x_op, label)
        loss = loss.mul(self.weights_data)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, x, label):
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x)
        x_op = self.net(x)
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
        # forward without error
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 1, 3)
        x = torch.from_numpy(x)
        x_op = self.net(x)
        x_op = x_op.numpy()
        x_op = np.swapaxes(x_op, 1, 3)
        x_op = np.swapaxes(x_op, 1, 2)
        return x_op

    def save(self,save_path):
        torch.save(self.net.state_dict(), save_path)
