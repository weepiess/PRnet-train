import numpy as np
from skimage import io, transform
import math
import copy
from PIL import ImageEnhance, ImageOps, ImageFile, Image
import sys
import cv2
import math
from math import cos, sin
from utils.render import render_texture
import scipy.io as sio
import time
import random

def randomColor(image):
    """
    """
    PIL_image = Image.fromarray((image * 255.).astype(np.uint8))
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(PIL_image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
    random_factor = np.random.randint(0, 31) / 10.
    out = np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))
    out = out / 255.
    return out


def getRotateMatrix(angle, image_shape):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0], [math.sin(angle), math.cos(-angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)


def getRotateMatrix3D(angle, image_shape):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, 0, -image_height / 2.], [0, 1, 0, -image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0, 0], [math.sin(-angle), math.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, image_height / 2.], [0, 1, 0, image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, 0, -image_height / 2.], [0, 1, 0, -image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0, 0], [math.sin(angle), math.cos(-angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, image_height / 2.], [0, 1, 0, image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)



def myDot(a, b):
    return np.dot(a, b)


def rotateData(x, y, angle_range=45, specify_angle=None):
    if specify_angle is None:
        angle = np.random.randint(-angle_range, angle_range)
        angle = angle / 180. * np.pi
    else:
        angle = specify_angle
    [image_height, image_width, image_channel] = x.shape
    # move-rotate-move
    [rform, rform_inv] = getRotateMatrix(angle, x.shape)

    # rotate_x = transform.warp(x, rform_inv,
    #                           output_shape=(image_height, image_width))
    rotate_x = cv2.warpPerspective(x, rform, (image_height, image_width))
    rotate_y = y.copy()
    rotate_y[:, :, 2] = 1.
    rotate_y = rotate_y.reshape(image_width * image_height, image_channel)
    # rotate_y = rotate_y.dot(rform.T)
    rotate_y = myDot(rotate_y, rform.T)
    rotate_y = rotate_y.reshape(image_height, image_width, image_channel)
    rotate_y[:, :, 2] = y[:, :, 2]
    # for i in range(image_height):
    #     for j in range(image_width):
    #         rotate_y[i][j][2] = 1.
    #         rotate_y[i][j] = rotate_y[i][j].dot(rform.T)
    #         rotate_y[i][j][2] = y[i][j][2]
    # tex = np.ones((256, 256, 3))
    # from visualize import show
    # show([rotate_y, tex, rotate_x.astype(np.float32)], mode='uvmap')
    return rotate_x, rotate_y


def gaussNoise(x, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, x.shape)
    out = x + noise
    out = np.clip(out, 0., 1.0)
    # cv.imshow("gasuss", out)
    return out


def randomErase(x, max_num=4, s_l=0.02, s_h=0.3, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=1.0):
    [img_h, img_w, img_c] = x.shape
    out = x.copy()
    num = np.random.randint(1, max_num)

    for i in range(num):
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        mask = np.zeros((img_h, img_w))
        mask[top:min(top + h, img_h), left:min(left + w, img_w)] = 1
        if np.random.rand() < 0.25:
            c = np.random.uniform(v_l, v_h)
            out[mask > 0] = c
        elif np.random.rand() < 0.75:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] = c0
            out1 = out[:, :, 1]
            out1[mask > 0] = c1
            out2 = out[:, :, 2]
            out2[mask > 0] = c2
        else:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] *= c0
            out1 = out[:, :, 1]
            out1[mask > 0] *= c1
            out2 = out[:, :, 2]
            out2[mask > 0] *= c2
    return out


def channelScale(x, min_rate=0.6, max_rate=1.4):
    out = x.copy()
    for i in range(3):
        r = np.random.uniform(min_rate, max_rate)
        out[:, :, i] = out[:, :, i] * r
    return out


def prnAugment_torch(x, y, is_rotate=True):
    if is_rotate:
        #if np.random.rand() > 0.5:
        x, y = rotateData(x, y, 90)
    if np.random.rand() > 0.75:
        x = randomErase(x)
    if np.random.rand() > 0.5:
        x = channelScale(x)
    # if np.random.rand() > 0.75:
    #     x = gaussNoise(x)
    return x, y


# def prnAugment_torch(x, y, is_rotate=True):
#     if np.random.rand() > 0.5:
#         x = channelScale(x)
#     return x, y
def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),   cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def rotate(vertices, angles):
    ''' rotate vertices. 
    X_new = R.dot(X). X: 3 x 1   
    Args:
        vertices: [nver, 3]. 
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down 
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices

def load_uv_coords(path = 'BFM_UV.mat'):
    ''' load uv coords of BFM
    Args:
        path: path to data.
    Returns:  
        uv_coords: [nver, 2]. range: 0-1
    '''
    C = sio.loadmat(path)
    uv_coords = C['UV'].copy(order = 'C')
    return uv_coords

def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

def synthesize(image,label,face_ind,triangles,mode):
    t1 = time.clock()
    turn_angle = 0
    if mode == 1:
        turn_angle = random.randint(-30,30)
    if mode == 0:
        turn_angle = random.randint(-10,0)
    if mode == 2:
        turn_angle = random.randint(0,10)

    pitch_angle = random.randint(-10,10)
    input_image = image/255.
    #print('posmapsize: ',label.shape)
    pos_gt = label
    # cv2.imshow('pos_gt',pos_gt)
    # cv2.waitKey(0)
    # ref_texture_gt = cv2.remap(input_image, pos_gt[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # all_colors_gt = np.reshape(ref_texture_gt, [256**2, -1])
    # text_c_gt = all_colors_gt[face_ind, :]*255
    all_vertices_gt = np.reshape(pos_gt, [256**2, -1])
    #print('vsize: ',all_vertices_gt.shape)
    temp = all_vertices_gt
    vertices_gt = all_vertices_gt[face_ind, :]
    # pic_gt = render_texture(vertices_gt.T,text_c_gt.T,triangles.T,256,256,3,image)
    #cv2.imshow('ori',input_image)
    #cv2.waitKey(0)

    vertices_gt[:,0] = np.minimum(np.maximum(vertices_gt[:,0], 0), 256 - 1)  # x
    vertices_gt[:,1] = np.minimum(np.maximum(vertices_gt[:,1], 0), 256 - 1)  # y
    ref_texture_r = cv2.remap(input_image, pos_gt[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    all_colors_r = np.reshape(ref_texture_r, [256**2, -1])
    text_c_r = all_colors_r[face_ind, :]
    #print('angle: ',[pitch_angle,turn_angle])
    rv = rotate(vertices_gt,[pitch_angle,turn_angle,0])
    temp[face_ind,:] = rv
    uv_position_map = np.reshape(temp,[256,256,3])


    pic_gt2 = render_texture(rv.T,text_c_r.T,triangles.T,256,256)
    #cv2.imshow('ori_t',pic_gt2)
    # cv2.waitKey(0)

    # all_vertices_gt_show = np.reshape(uv_position_map, [256**2, -1])
    # print('vsize: ',all_vertices_gt_show.shape)

    # vertices_gt_show = all_vertices_gt[face_ind, :]

    # vertices_gt_show[:,0] = np.minimum(np.maximum(vertices_gt_show[:,0], 0), 256 - 1)  # x
    # vertices_gt_show[:,1] = np.minimum(np.maximum(vertices_gt_show[:,1], 0), 256 - 1)  # y
    # ref_texture_show = cv2.remap(pic_gt2, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # all_colors_show = np.reshape(ref_texture_show, [256**2, -1])
    # text_c_show = all_colors_show[face_ind, :]
    # #rv = rotate(vertices_gt,[0,0,0])
    # #uv_position_map = render_texture(uv_coords.T,rv.T,triangles.T, 256, 256, c = 3)

    # pic_r = render_texture(vertices_gt_show.T,text_c_show.T,triangles.T,256,256)
    # cv2.imshow('img',pic_r)
    # cv2.waitKey(0)
    print(time.clock() - t1)
    return pic_gt2,uv_position_map

if __name__ == '__main__':
    
    from skimage import io

    # x = io.imread('../10105-14-0.jpg') / 255.
    # x = x.astype(np.float32)
    # y = np.load('../10105-14-0.npy')
    # y = y.astype(np.float32)

    # t1 = time.clock()
    # # for i in range(1000):
    # #     xr, yr = prnAugment_torch(x, y)
    # xr, yr = prnAugment_torch(x, y)
    # cv2.imshow('pic',xr)
    # cv2.waitKey(0)
    # print(time.clock() - t1)
    image = cv2.imread('../10173-other_3-1.jpg')
    face_ind = np.loadtxt('../Data/uv-data/face_ind.txt').astype(np.int32)
    triangles = np.loadtxt('../Data/uv-data/triangles.txt').astype(np.int32)
    input_image = image/255.
    
    pos_gt = np.load('../10173-other_3-1.npy')
    pos_gt = np.array(pos_gt).astype(np.float32)
    ref_texture_gt = cv2.remap(input_image, pos_gt[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    all_colors_gt = np.reshape(ref_texture_gt, [256**2, -1])
    text_c_gt = all_colors_gt[face_ind, :]*255
    all_vertices_gt = np.reshape(pos_gt, [256**2, -1])
    vertices_gt = all_vertices_gt[face_ind, :]
    pic_gt = render_texture(vertices_gt.T,text_c_gt.T,triangles.T,256,256,3,image)
    cv2.imshow('ori',pic_gt)
    #cv2.waitKey(0)

    vertices_gt[:,0] = np.minimum(np.maximum(vertices_gt[:,0], 0), 256 - 1)  # x
    vertices_gt[:,1] = np.minimum(np.maximum(vertices_gt[:,1], 0), 256 - 1)  # y
    ref_texture_r = cv2.remap(input_image, pos_gt[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    all_colors_r = np.reshape(ref_texture_r, [256**2, -1])
    text_c_r = all_colors_r[face_ind, :]
    rv = rotate(vertices_gt,[0,30,0])
    rc = rotate(text_c_r,[0,10,0])
    pic_r = render_texture(rv.T,text_c_r.T,triangles.T,256,256)
    cv2.imshow('pic_r',pic_r)
    cv2.waitKey(0)