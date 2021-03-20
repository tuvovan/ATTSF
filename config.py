"""
This is the configuration module has all the gobal variables and basic
libraries to be shared with other modules in the same project.

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""
import argparse
import numpy as np
import os
import math
import cv2
import sys
import random
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.utils import Sequence
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error
from tensorflow.python.data.experimental import AUTOTUNE
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras import Model

# results and model name
res_model_name='l5_s512_f0.7_d0.4'
op_phase='train'

# image mini-batch size
img_mini_b = 4

#########################################################################
# READ & WRITE DATA PATHS									            #
#########################################################################
# run on server or local machine
server=False

sub_folder=['source/','target/']

if op_phase=='train':
    dataset_name='_canon_patch'
    dataset_name_add = '_add_patch'
    resize_flag=False
elif op_phase=='test':
    dataset_name='_pixel'
    resize_flag=False
elif op_phase=='valid':
    dataset_name=''
    resize_flag=False
else:
    raise NotImplementedError

# path to save model
path_save_model='ModelCheckpoints/model.h5'
    

# paths to read data
path_read_train = 'dd_dp_dataset'+dataset_name+'/'
path_read_val_test = 'dd_dp_dataset'+dataset_name+'/'
path_read_val_valid = 'dd_dp_dataset_validation_inputs_only/'

# path to write results
path_write='results/res_'+res_model_name+'_dd_dp'+dataset_name+'/'
path_write_comp = 'results_comp/'

os.makedirs(path_write, exist_ok=True)
os.makedirs(path_write_comp, exist_ok=True)
#########################################################################
# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS	    #
#########################################################################
if op_phase=='train':
    total_nb_train = len([path_read_train + 'train_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read_train + 'train_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])
    
    total_nb_val = len([path_read_val_test + 'val_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read_val_test + 'val_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])
    
    # number of training image batches
    nb_train = int(math.ceil(total_nb_train/img_mini_b))
    # number of validation image batches
    nb_val = int(math.ceil(total_nb_val/img_mini_b))
    
elif op_phase=='test':
    total_nb_test = len([path_read_val_test + 'test_c/' + sub_folder[0] + f for f
                    in os.listdir(path_read_val_test + 'test_c/' + sub_folder[0])
                    if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))])

elif op_phase=='valid':
    total_nb_test = len([path_read_val_valid + f for f
                    in os.listdir(path_read_val_valid)
                    if (f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF')) and '_r' in f)])

#########################################################################
# MODEL PARAMETERS & TRAINING SETTINGS									#
#########################################################################

# input image size
img_w = 1680
img_h = 1120

# input patch size
patch_w=560
patch_h=560
# patch_w = 1680
# patch_h = 1120

# mean value pre-claculated
src_mean=0
trg_mean=0

# number of epochs
nb_epoch = 300

# number of input channels
nb_ch_all= 6
# number of output channels
nb_ch=3  # change conv9 in the model and the folowing variable

# color flag:"1" for 3-channel 8-bit image or "0" for 1-channel 8-bit grayscale
# or "-1" to read image as it including bit depth
color_flag=-1
if op_phase == 'valid':
    bit_depth=16
    color_flag = -1
else:
    bit_depth=16
    # color_flag = 1

norm_val=(2**bit_depth)-1

# after how many epochs you change learning rate
scheduling_rate=30

dropout_rate=0.4

# generate learning rate array
lr_=[]
lr_.append(1e-4) #initial learning rate
for i in range(int(nb_epoch/scheduling_rate)):
    lr_.append(lr_[i]*0.5)

train_set, val_set, test_set, comp_set = [], [], [], []

size_set, portrait_orientation_set = [], []

mse_list, psnr_list, ssim_list, mae_list = [], [], [], []

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) 
    return lr

def step_decay_schedule(epoch):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    initial_lr=1e-5
    decay_factor=0.5
    step_size=2
    return initial_lr * (decay_factor ** np.floor(epoch/step_size))

def loss_function(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference, axis=-1)
    ssim = SSIMLoss(y_true, y_pred)

    total_loss = 100*mse + 5*ssim
    return total_loss

def loss_function_2(y_true, y_pred):
    absolute_difference = tf.abs(y_true - y_pred)
    mse = tf.reduce_mean(absolute_difference, axis=-1)
    ssim = SSIMLossMS(y_true, y_pred)

    total_loss = 5*mse + 1*ssim
    return total_loss

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def SSIMLossMS(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))

lr__ = []
lr__.append(1e-5)