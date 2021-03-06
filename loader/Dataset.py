import numpy as np
import os
import argparse
import cv2
import random
import math
from datetime import datetime
from skimage.io import imread, imsave
import scipy.io as sio


class TrainData(object):

    def __init__(self, train_data_file):

        self.train_data_file = train_data_file
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)

    def readTrainData(self):
        with open(self.train_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip('\n').split('*')
                self.train_data_list.append(item)
            random.shuffle(self.train_data_list)

    def load_uv_coords(self,path = 'BFM_UV.mat'):
        ''' load uv coords of BFM
        Args:
            path: path to data.
        Returns:
            uv_coords: [nver, 2]. range: 0-1
        '''
        C = sio.loadmat(path)
        uv_coords = C['UV'].copy(order = 'C')
        return uv_coords

    def process_uv(self,uv_coords, uv_h = 256, uv_w = 256):
        uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
        uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
        uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
        return uv_coords

    def getBatch_normal(self, batch_list):
        batch = []
        imgs = []
        labels = []
        for item in batch_list:
            img = cv2.imread(item[0])
            label = np.load(item[1])
            img_array = np.array(img, dtype=np.float32)
            imgs.append(img_array / 256.0 / 1.1)

            label_array = np.array(label, dtype=np.float32)
            labels.append(label_array / 256 / 1.1)
        batch.append(imgs)
        batch.append(labels)

        return batch

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        #step = 0
        face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32)
        triangles = np.loadtxt('./Data/uv-data/triangles.txt').astype(np.int32)
        for item in batch_list:
            # step = step + 1
            # print('read num :',step)
            # print('0: ',item[0])
            if '300W' in item[0]:
                img = cv2.imread(item[0])
                label = np.load(item[1])
                img_array = np.array(img, dtype=np.float32)
                imgs.append(img_array / 256.0 / 1.1)

                label_array = np.array(label, dtype=np.float32)
                labels.append(label_array / 256 / 1.1)
                #print('common')
            else:
                name = item[0].split('/posmap/')[1]
                mode = 0
                if '-0.jpg' in name:
                    mode = 0
                if '-1.jpg' in name:
                    mode = 1
                if '-2.jpg' in name:
                    mode = 2
                #print('name: ',name)
                img = cv2.imread(item[0])
                label = np.load(item[1])
                img_,label_ = synthesize(img,label,face_ind,triangles,mode)
                img_array = np.array(img_, dtype=np.float32)
                imgs.append(img_array / 256.0 / 1.1)

                label_array = np.array(label_, dtype=np.float32)
                labels.append(label_array / 256 / 1.1)
                #print('read down****************')
        batch.append(imgs)
        batch.append(labels)

        return batch

    def __call__(self, batch_num,type_method):
        if (self.index + batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            if type_method == 0:
                batch_data = self.getBatch(batch_list)
            else:
                batch_data = self.getBatch_normal(batch_list)
            self.index += batch_num

            return batch_data
        else:
            self.index = 0
            random.shuffle(self.train_data_list)
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch_normal(batch_list)
            self.index += batch_num

            return batch_data