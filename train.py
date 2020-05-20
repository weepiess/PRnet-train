import numpy as np
import os
import argparse
import torch
import cv2
import random
import math
from net.predictor import resfcn256
from model.PRNet import PRNet
from datetime import datetime
from utils import render
from utils.render import render_texture
from skimage.io import imread, imsave
from utils.write import write_obj_with_colors
from utils.augmentation import synthesize
from loader.Dataset import TrainData
#from loader.eval_3dfaw import eval_meshes as
#from loader.eval_300w import eval_meshes
#from loader.eval_pixel import eval_meshes as
from opt.config import Options
import scipy.io as sio
#from net.api import PRN


def main(args):
    trainConfig = Options()
    opt = trainConfig.get_config()
    #prn = PRN(is_dlib = True) 
    # Some arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    batch_size = opt.batch_size
    epochs = opt.epochs
    train_data_file = args.train_data_file
    model_path = args.model_path
    eval_pixel_file = args.eval_pixel_file
    eval_3DFAW_file = args.eval_3DFAW_file
    eval_300W_file = args.eval_300W_file
    #res_dir = args.res_dir
    #ref_dir = args.ref_dir
    '''
    save_dir = args.checkpoint
    if  not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''
    # Training data
    data = TrainData(train_data_file)
    #eval_pixel = TrainData(eval_pixel_file)
    #eval_3DFAW = TrainData(eval_3DFAW_file)
    #eval_300W = TrainData(eval_300W_file)
    #show_data = TrainData(train_data_file)
    begin_epoch = 0
    # if os.path.exists(model_path + '.data-00000-of-00001'):
    #     begin_epoch = int(model_path.split('_')[-1]) + 1
    #     print('begin: ',begin_epoch)
    '''
    epoch_iters = data.num_data / batch_size
    global_step = tf.Variable(epoch_iters * begin_epoch, trainable=False)
    # Declay learning rate half every 5 epochs
    decay_steps = 5 * epoch_iters
    # learning_rate = learning_rate * 0.5 ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                               decay_steps, 0.5, staircase=True)
    '''
    # Model
    model = PRNet(opt)
    model.setup(model_path)
    model.train()

    # Begining train
    error_f = open('./results/error.txt','w')
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fp_log = open("./logs/log_" + time_now + ".txt","w")
    iters_total_each_epoch = int(math.ceil(1.0 * data.num_data / batch_size))
    print('iters_total_each_epoch: ',iters_total_each_epoch)
    #eval_pixel_batch = eval_pixel(eval_pixel.num_data,1)
    #eval_3DFAW_batch = eval_3DFAW(eval_3DFAW.num_data,1)
    #eval_300W_batch = eval_300W(eval_300W.num_data,1)

    #loss_pixel = model.optmize_parameters(x=eval_pixel_batch[0], label=eval_pixel_batch[1])
    #loss_3DFAW = model.optmize_parameters(x=eval_3DFAW_batch[0], label=eval_3DFAW_batch[1])
    #loss_300W = model.optmize_parameters(x=eval_300W_batch[0], label=eval_300W_batch[1])

    #print('error of Pixel start: ',loss_pixel)
    #('error of 3DFAW start: ',loss_3DFAW)
    #print('error of 300W start: ',loss_300W)
    #error_f.write('error in pixel 1st : '+str(loss_pixel)+' error in 3DFAW 1st : '+str(loss_3DFAW)+'\n')

    image = cv2.imread('./examples/10173-other_3-1.jpg')
    face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32)
    triangles = np.loadtxt('./Data/uv-data/triangles.txt').astype(np.int32)
    input_image = image/255.
    
    pos_gt = np.load('./examples/10173-other_3-1.npy')
    pos_gt = np.array(pos_gt).astype(np.float32)
    ref_texture_gt = cv2.remap(input_image, pos_gt[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    all_colors_gt = np.reshape(ref_texture_gt, [256**2, -1])
    text_c_gt = all_colors_gt[face_ind, :]*255
    all_vertices_gt = np.reshape(pos_gt, [256**2, -1])
    vertices_gt = all_vertices_gt[face_ind, :]
    pic_gt = render_texture(vertices_gt.T,text_c_gt.T,triangles.T,256,256,3,image)
    vertices_gt[:,0] = np.minimum(np.maximum(vertices_gt[:,0], 0), 256 - 1)  # x
    vertices_gt[:,1] = np.minimum(np.maximum(vertices_gt[:,1], 0), 256 - 1)  # y
    ind = np.round(vertices_gt).astype(np.int32)
    col = image[ind[:,1], ind[:,0], :] # n x 3
    imsave('./results/results_gt'+'.png',pic_gt)
    write_obj_with_colors('./results/results_gt'+'.obj',vertices_gt,triangles,col)
    # cv2.imshow('ref',ref_texture_gt)
    # cv2.waitKey(0)
    test_list = []
    for epoch in range(begin_epoch, epochs):
        train_loss_mean = 0
        #show_data_ = show_data(1)

        for iters in range(iters_total_each_epoch):
            if True:#iters % 100 == 0:
                batch = data(batch_size,0)
            else:
                batch = data(batch_size,1)
            loss_res = model.optimize_parameters(x=batch[0], label=batch[1])
            train_loss_mean = train_loss_mean + loss_res
            #summary_str = sess.run(merged_summary_op,feed_dict={x: batch[0], label: batch[1]})
            time_now_tmp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            for_show = 'epoch: '+str(epoch)+' iters: '+str(iters)+' loss: '+str(loss_res)
            log_line = str(loss_res)
            print (for_show)
            fp_log.writelines(log_line + "\n")
        train_loss_mean = train_loss_mean/iters_total_each_epoch
        #eval_pixel_batch_ = eval_pixel(eval_pixel.num_data,1)
        #eval_3DFAW_batch_ = eval_3DFAW(eval_3DFAW.num_data,1)
        #eval_300W_batch_ = eval_300W(eval_300W.num_data,1)

        #text_c = prn.get_colors_from_texture(ref_texture)
        #print('colors: ',text_c)

        #pic = render_texture(cropped_vertices.T,text_c.T,prn.triangles.T,256,256)
        # pos = np.load('./10082-15-1.npy')
        # pos = np.array(pos).astype(np.float32)
        posmap = model.generate(x=input_image[np.newaxis, :,:,:])
        posmap = np.squeeze(posmap)
        posmap = posmap*256*1.1
        cropped_vertices = np.reshape(posmap, [-1, 3]).T
        pos = np.reshape(cropped_vertices.T, [256, 256, 3])
        ref_texture = cv2.remap(input_image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        all_colors = np.reshape(ref_texture, [256**2, -1])
        text_c = all_colors[face_ind, :]*255
        test_list.append(pos)

        all_vertices = np.reshape(pos, [256**2, -1])
        vertices = all_vertices[face_ind, :]

        pic = render_texture(vertices.T,text_c.T,triangles.T,256,256,3,image)
        imsave('./results/result'+str(epoch)+'.png',pic)
        write_obj_with_colors('./results/result'+str(epoch)+'.obj',vertices,triangles,triangles)
        #picture = np.multiply(pic,255)
        #print('pic',picture)
        # cv2.imshow('ref',ref_texture)
        # cv2.waitKey(0)
        # image_pic = tf.image.decode_png(picture, channels=4)
        # image_pic = tf.expand_dims(input_image, 0)
        # ori_pic = tf.image.decode_png(image, channels=4)
        # ori_pic = tf.expand_dims(ori_pic, 0)
        '''
        _, loss_pixel_, _ = model.forward(x=eval_pixel_batch_[0], label=eval_pixel_batch_[1])
        #loss_3DFAW_ = sess.run(error,feed_dict={x: eval_3DFAW_batch_[0], label: eval_3DFAW_batch_[1]})
        _, _, loss_300W_ = model.forward(x=eval_300W_batch_[0], label=eval_300W_batch_[1])
        '''
        '''
        summary =tf.Summary(value=[
            tf.Summary.Value(tag="error_pixel", simple_value=loss_pixel_), 
            #tf.Summary.Value(tag="error_3DFAW", simple_value=loss_3DFAW_),
            tf.Summary.Value(tag="error_300W", simple_value=loss_300W_),
            tf.Summary.Value(tag="train loss", simple_value=train_loss_mean)])
        summary_writer.add_summary(summary, epoch)
        '''
        model.save(save_path='./Data/train_result/256_256_resfcn256' + '_' + str(epoch))
        # Test
        # eval_pixel_batcht = eval_pixel(eval_pixel.num_data,1)
        # eval_3DFAW_batcht = eval_3DFAW(eval_3DFAW.num_data,1)
        # loss_pixel2 = sess.run(loss,feed_dict={x: eval_pixel_batcht[0], label: eval_pixel_batcht[1]})
        # loss_3DFAW2 = sess.run(loss,feed_dict={x: eval_3DFAW_batcht[0], label: eval_3DFAW_batcht[1]})
        #print('error of Pixel: ',loss_pixel2)
        #print('error of 3DFAW: ',loss_3DFAW2)
        #error_f.write('error in pixel: '+str(loss_pixel2)+' error in 3DFAW: '+str(loss_3DFAW2)+'\n')
    '''
    model.eval()
    eval_meshes(test_list, res_dir, ref_dir)
    '''
    fp_log.close()
    error_f.close()

if __name__ == '__main__':

    par = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')
    par.add_argument('--train_data_file', default='/home/jikai/generate_posmap/train_path_afw.txt', type=str, help='The training data file')
    par.add_argument('--checkpoint', default='./checkpoint/', type=str, help='The path of checkpoint')
    par.add_argument('--model_path', default='./Data/net-data/256_256_resfcn256_weight', type=str, help='The path of pretrained model')
    par.add_argument('--eval_pixel_file', default='/media/weepies/Seagate Backup Plus Drive/3DMM/3d-pixel/subt_eva.txt', type=str,
                     help='eval_pixel_file')
    par.add_argument('--eval_3DFAW_file', default='/media/weepies/Seagate Backup Plus Drive/3DMM/3DFAW_posmap/3DFAW_pos_eva.txt', type=str,
                     help='eval_3DFAW_file')
    par.add_argument('--eval_300W_file', default='/media/weepies/Seagate Backup Plus Drive/3DMM/train_path_ibug.txt', type=str,
                     help='eval_300W_file')
    #par.add_argument('--checkpoint', default='./checkpoint/', type=str, help='The path of checkpoint')
    '''
    par.add_argument('--res_dir', default='', type=str, help='eval_mesh_res_dir')
    par.add_argument('--ref_dir', default='', type=str, help='eval_mesh_ref_dir')
    '''

    main(par.parse_args())
