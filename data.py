"""
This module has all the functions used for the data manipulation, data
generation, and learning rate scheduler.

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

This code imports the modules and starts the implementation based on the
configurations in config.py module.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""

from config import *
from metrics import *

def check_dir(_path):
    if not os.path.exists(_path):
        try:
            os.makedirs(_path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
def schedule_learning_rate(epoch):
    lr=lr_[int(epoch/scheduling_rate)]
    return lr

def data_random_shuffling(temp_type):
    global train_set, val_set, test_set, comp_set, path_read_train, path_read_val_test
    if temp_type != 'validation':
        if temp_type == 'train':
            path_read = path_read_train
        else:
            path_read = path_read_val_test
        
        images_C_src = [path_read + temp_type + '_c/' + sub_folder[0] + f for f
                        in os.listdir(path_read + temp_type + '_c/' + sub_folder[0])
                        if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
        images_C_src.sort()
        
        images_C_trg = [path_read + temp_type + '_c/' + sub_folder[1] + f for f
                        in os.listdir(path_read + temp_type + '_c/' + sub_folder[1])
                        if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
        images_C_trg.sort()
        
        images_L_src = [path_read + temp_type + '_l/' + sub_folder[0] + f for f
                        in os.listdir(path_read + temp_type + '_l/' + sub_folder[0])
                        if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
        images_L_src.sort()
        
        images_R_src = [path_read + temp_type + '_r/' + sub_folder[0] + f for f
                        in os.listdir(path_read + temp_type + '_r/'  + sub_folder[0])
                        if f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF'))]
        images_R_src.sort()
        
        len_imgs_list=len(images_C_src)
        
        # generate random shuffle index list for all list
        tempInd=np.arange(len_imgs_list)
        random.shuffle(tempInd)
        
        images_C_src=np.asarray(images_C_src)[tempInd]
        images_C_trg=np.asarray(images_C_trg)[tempInd]
        
        images_L_src=np.asarray(images_L_src)[tempInd]
        images_R_src=np.asarray(images_R_src)[tempInd]

        for i in range(len_imgs_list):
            if temp_type =='train':
                train_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                                images_C_trg[i]])
            elif temp_type =='val':
                val_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                                images_C_trg[i]])
            elif temp_type =='test':
                test_set.append([images_C_src[i],images_L_src[i],images_R_src[i],
                                images_C_trg[i]])
            else:
                raise NotImplementedError
    else:
        path_read = 'dd_dp_dataset_validation_inputs_only/'
        images_L_src = [path_read + f for f
                        in os.listdir(path_read)
                        if (f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF')) and '_l' in f)]
        images_L_src.sort()
        
        images_R_src = [path_read + f for f
                        in os.listdir(path_read)
                        if (f.endswith(('.jpg','.JPG', '.png', '.PNG', '.TIF')) and '_r' in f)]
        images_R_src.sort()
        
        len_imgs_list=len(images_L_src)
        
        # generate random shuffle index list for all list
        tempInd=np.arange(len_imgs_list)
        random.shuffle(tempInd)
        
        images_L_src=np.asarray(images_L_src)[tempInd]
        images_R_src=np.asarray(images_R_src)[tempInd]

        for i in range(len_imgs_list):
            comp_set.append([images_L_src[i],images_R_src[i]])
    
def test_generator(num_image):
    in_img_tst = np.zeros((num_image, img_h, img_w, nb_ch_all))
    out_img_gt = np.zeros((num_image, img_h, img_w, nb_ch))

    for i in range(num_image):
        print('Read image: ',i,num_image)
        if resize_flag:
            temp_img_l=cv2.imread(test_set[i][1],color_flag)
            size_set.append([temp_img_l.shape[1],temp_img_l.shape[0]])
            if temp_img_l.shape[0]>temp_img_l.shape[1]:
                portrait_orientation_set.append(True)
                temp_img_l=cv2.rotate(temp_img_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
                in_img_tst[i, :,:,0:3] = (cv2.resize((temp_img_l-src_mean)/norm_val,
                            (img_w,img_h))).reshape((img_h, img_w,nb_ch))
                temp_img_r=cv2.rotate(cv2.imread(test_set[i][2],color_flag), cv2.ROTATE_90_COUNTERCLOCKWISE)
                in_img_tst[i, :,:,3:6] = (cv2.resize((temp_img_r-src_mean)
                                    /norm_val,(img_w,img_h))).reshape((img_h, img_w,nb_ch))
                temp_img_trg=cv2.rotate(cv2.imread(test_set[i][3],color_flag), cv2.ROTATE_90_COUNTERCLOCKWISE)
                out_img_gt[i, :] = (cv2.resize((temp_img_trg-src_mean)
                                    /norm_val,(img_w,img_h))).reshape((img_h, img_w,nb_ch))

            else:
                portrait_orientation_set.append(False)
                in_img_tst[i, :,:,0:3] = (cv2.resize((temp_img_l-src_mean)/norm_val,
                            (img_w,img_h))).reshape((img_h, img_w,nb_ch))
                in_img_tst[i, :,:,3:6] = (cv2.resize((cv2.imread(test_set[i][2],color_flag)-src_mean)
                                    /norm_val,(img_w,img_h))).reshape((img_h, img_w,nb_ch))
                out_img_gt[i, :] = (cv2.resize((cv2.imread(test_set[i][3],color_flag)-src_mean)
                                    /norm_val,(img_w,img_h))).reshape((img_h, img_w,nb_ch))
                
        else:
            in_img_tst[i, :,:,0:3] = ((cv2.imread(test_set[i][1],color_flag)-src_mean)
                                    /norm_val).reshape((img_h, img_w,nb_ch))
            in_img_tst[i, :,:,3:6] = ((cv2.imread(test_set[i][2],color_flag)-src_mean)
                                    /norm_val).reshape((img_h, img_w,nb_ch))
            out_img_gt[i, :] = ((cv2.imread(test_set[i][3],color_flag)-src_mean)
                              /norm_val).reshape((img_h, img_w,nb_ch))
    return in_img_tst, out_img_gt

def validation_generator(num_image):
    in_img_tst = np.zeros((num_image, img_h, img_w, nb_ch_all))

    for i in range(num_image):
        print('Read image: ',i,num_image)
        if resize_flag:
            temp_img_l=cv2.imread(comp_set[i][0],color_flag)
            size_set.append([temp_img_l.shape[1],temp_img_l.shape[0]])
            if temp_img_l.shape[0]>temp_img_l.shape[1]:
                portrait_orientation_set.append(True)
                temp_img_l=cv2.rotate(temp_img_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
                in_img_tst[i, :,:,0:3] = (cv2.resize((temp_img_l-src_mean)/norm_val,
                            (img_w,img_h))).reshape((img_h, img_w,nb_ch))
                temp_img_r=cv2.rotate(cv2.imread(comp_set[i][1],color_flag), cv2.ROTATE_90_COUNTERCLOCKWISE)
                in_img_tst[i, :,:,3:6] = (cv2.resize((temp_img_r-src_mean)
                                    /norm_val,(img_w,img_h))).reshape((img_h, img_w,nb_ch))
            else:
                portrait_orientation_set.append(False)
                in_img_tst[i, :,:,0:3] = (cv2.resize((temp_img_l-src_mean)/norm_val,
                            (img_w,img_h))).reshape((img_h, img_w,nb_ch))
                in_img_tst[i, :,:,3:6] = (cv2.resize((cv2.imread(comp_set[i][1],color_flag)-src_mean)
                                    /norm_val,(img_w,img_h))).reshape((img_h, img_w,nb_ch))
                
        else:
            in_img_tst[i, :,:,0:3] = ((cv2.imread(comp_set[i][0],color_flag)-src_mean)
                                    /norm_val).reshape((img_h, img_w,nb_ch))
            in_img_tst[i, :,:,3:6] = ((cv2.imread(comp_set[i][1],color_flag)-src_mean)
                                    /norm_val).reshape((img_h, img_w,nb_ch))
    return in_img_tst

def generator(phase_gen='train'):
    if phase_gen == 'train':       
        data_set_temp=train_set
        nb_total=total_nb_train
    elif phase_gen == 'val':
        data_set_temp=val_set
        nb_total=total_nb_val
    else:
        raise NotImplementedError
        
    image_counter = 0
    src_ims = np.zeros((img_mini_b, patch_h, patch_w, nb_ch_all))
    trg_ims = np.zeros((img_mini_b, patch_h, patch_w, nb_ch))
    num_iter = 1
    while True:      
        num_iter += 1
        if phase_gen == 'train' and num_iter == nb_train:
            np.random.shuffle(data_set_temp)
        for i in range(0, img_mini_b):
            img_data_src_c = data_set_temp[(image_counter + i) % (nb_total)][0]
            img_data_src_l = data_set_temp[(image_counter + i) % (nb_total)][1]
            img_data_src_r = data_set_temp[(image_counter + i) % (nb_total)][2]
            
            img_data_trg = data_set_temp[(image_counter + i) % (nb_total)][3]
            if resize_flag:
                src_ims[i, :,:,0:3] = (cv2.resize((cv2.imread(img_data_src_l,color_flag)-src_mean)
                                    /norm_val,(patch_w,patch_h))).reshape((patch_h, patch_w,nb_ch))
                src_ims[i, :,:,3:6] = (cv2.resize((cv2.imread(img_data_src_r,color_flag)-src_mean)
                                    /norm_val,(patch_w,patch_h))).reshape((patch_h, patch_w,nb_ch))
                trg_ims[i, :] = (cv2.resize((cv2.imread(img_data_trg,color_flag)-trg_mean)
                                    /norm_val,(patch_w,patch_h))).reshape((patch_h, patch_w,nb_ch))
            else:
                src_ims[i, :,:,0:3] = ((cv2.imread(img_data_src_l,color_flag)-src_mean)
                                      /norm_val).reshape((patch_h, patch_w,nb_ch))
                src_ims[i, :,:,3:6] = ((cv2.imread(img_data_src_r,color_flag)-src_mean)
                                      /norm_val).reshape((patch_h, patch_w,nb_ch))
                trg_ims[i, :] = ((cv2.imread(img_data_trg,color_flag)-trg_mean)
                                /norm_val).reshape((patch_h, patch_w,nb_ch))               
        X, y = random_flip(src_ims, trg_ims)
        X, y = random_rotate(X, y)
        yield (X,y)
        image_counter = (image_counter + img_mini_b) % (nb_total)
        
        
def save_eval_predictions(path_to_save,test_imgaes,predictions,gt_images):
    global mse_list, psnr_list, ssim_list, test_set
    for i in range(len(test_imgaes)):
        mse, psnr, ssim = MSE_PSNR_SSIM((gt_images[i]).astype(np.float64), (predictions[i]).astype(np.float64))
        mae = MAE((gt_images[i]).astype(np.float64), (predictions[i]).astype(np.float64))
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mae_list.append(mae)

        temp_in_img=cv2.imread(test_set[i][0],color_flag)
        if bit_depth == 8:
            temp_out_img=((predictions[i]*norm_val)+src_mean).astype(np.uint8)
            temp_gt_img=((gt_images[i]*norm_val)+src_mean).astype(np.uint8)
        elif bit_depth == 16:
            temp_out_img=((predictions[i]*norm_val)+src_mean).astype(np.uint16)
            temp_gt_img=((gt_images[i]*norm_val)+src_mean).astype(np.uint16)
        img_name=((test_set[i][0]).split('/')[-1]).split('.')[0]
        if resize_flag:
            if portrait_orientation_set[i]:
                temp_out_img=cv2.resize(cv2.rotate(temp_out_img,cv2.ROTATE_90_CLOCKWISE),(size_set[i][0],size_set[i][1]))
                temp_gt_img=cv2.resize(cv2.rotate(temp_gt_img,cv2.ROTATE_90_CLOCKWISE),(size_set[i][0],size_set[i][1]))
            else:
                temp_out_img=cv2.resize(temp_out_img,(size_set[i][0],size_set[i][1]))
                temp_gt_img=cv2.resize(temp_gt_img,(size_set[i][0],size_set[i][1]))
        cv2.imwrite(path_to_save+str(img_name)+'_i.png',temp_in_img)
        cv2.imwrite(path_to_save+str(img_name)+'_p.png',temp_out_img)
        cv2.imwrite(path_to_save+str(img_name)+'_g.png',temp_gt_img)
        print('Write image: ',i,len(test_imgaes))

def save_eval_comp(path_to_save,test_imgaes,predictions):
    global comp_set
    for i in range(len(test_imgaes)):
        bit_depth = 8
        norm_val = (2 ** bit_depth) - 1
        temp_out_img=((predictions[i]*norm_val)+src_mean).astype(np.uint8)
        img_name=((comp_set[i][0]).split('/')[-1]).split('.')[0]
        if resize_flag:
            if portrait_orientation_set[i]:
                temp_out_img=cv2.resize(cv2.rotate(temp_out_img,cv2.ROTATE_90_CLOCKWISE),(size_set[i][0],size_set[i][1]))
            else:
                temp_out_img=cv2.resize(temp_out_img,(size_set[i][0],size_set[i][1]))
        cv2.imwrite(path_to_save+str(img_name)[:-2]+'_g.png',temp_out_img)
        print('Write image: ',i,len(test_imgaes))

class DataGenerator():
    def __init__(self, batch_size, subset='train', shuffle=True):
        self.subset = subset
        if subset == 'train':
            self.images_dir = "dd_dp_dataset_canon_patch/train_c"
            self.data_ids = np.array([str(i) for i in sorted(os.listdir(os.path.join(self.images_dir, sub_folder[1]))) if 'png' in i])
        elif subset == 'valid':
            self.images_dir = "dd_dp_dataset_canon_patch/val_c"
            self.data_ids = np.array([str(i) for i in sorted(os.listdir(os.path.join(self.images_dir, sub_folder[1]))) if 'png' in i])
        elif subset == 'test':
            self.images_dir = "dd_dp_dataset_pixel/test_c"
            self.data_ids = np.array([str(i) for i in sorted(os.listdir(os.path.join(self.images_dir, sub_folder[1]))) if 'png' in i])
        else:
            raise ValueError("subset must be 'train', 'valid' or 'test'")

        self.indices = np.arange(len(self.data_ids)).astype(np.uint32)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.data_ids) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        indexes = self.data_ids[inds]

        num_image = len(indexes)
        if self.subset == 'test':
            num_image = len(self.data_ids)
            src_ims = np.zeros((num_image, img_h, img_w, nb_ch_all))
            trg_ims = np.zeros((num_image, img_h, img_w, nb_ch))
        else:
            src_ims = np.zeros((num_image, patch_h, patch_w, nb_ch_all))
            trg_ims = np.zeros((num_image, patch_h, patch_w, nb_ch))
        for i in range(0, num_image):
            img_data_src_l = os.path.join(self.images_dir, sub_folder[0], self.data_ids[i]).replace(self.subset + '_c', self.subset + '_l')
            img_data_src_r = os.path.join(self.images_dir, sub_folder[0], self.data_ids[i]).replace(self.subset + '_c', self.subset + '_r')
            img_data_trg = os.path.join(self.images_dir, sub_folder[1], self.data_ids[i])
            # print(img_data_src_l, img_data_src_r, img_data_trg)
            if resize_flag:
                src_ims[i, :,:,0:3] = (cv2.resize((cv2.imread(img_data_src_l,color_flag)-src_mean)
                                    /norm_val,(patch_w,patch_h))).reshape((patch_h, patch_w,nb_ch))
                src_ims[i, :,:,3:6] = (cv2.resize((cv2.imread(img_data_src_r,color_flag)-src_mean)
                                    /norm_val,(patch_w,patch_h))).reshape((patch_h, patch_w,nb_ch))
                trg_ims[i, :] = (cv2.resize((cv2.imread(img_data_trg,color_flag)-trg_mean)
                                    /norm_val,(patch_w,patch_h))).reshape((patch_h, patch_w,nb_ch))
            else:
                if self.subset == 'test':
                    src_ims[i, :,:,0:3] = ((cv2.imread(img_data_src_l,color_flag)-src_mean)
                                      /norm_val).reshape((img_h, img_w,nb_ch))
                    src_ims[i, :,:,3:6] = ((cv2.imread(img_data_src_r,color_flag)-src_mean)
                                        /norm_val).reshape((img_h, img_w,nb_ch))
                    trg_ims[i, :] = ((cv2.imread(img_data_trg,color_flag)-trg_mean)
                                    /norm_val).reshape((img_h, img_w,nb_ch)) 
                else:
                    src_ims[i, :,:,0:3] = ((cv2.imread(img_data_src_l,color_flag)-src_mean)
                                        /norm_val).reshape((patch_h, patch_w,nb_ch))
                    src_ims[i, :,:,3:6] = ((cv2.imread(img_data_src_r,color_flag)-src_mean)
                                        /norm_val).reshape((patch_h, patch_w,nb_ch))
                    trg_ims[i, :] = ((cv2.imread(img_data_trg,color_flag)-trg_mean)
                                    /norm_val).reshape((patch_h, patch_w,nb_ch)) 
        if self.shuffle:
            src_ims, trg_ims = random_flip(src_ims, trg_ims)
            src_ims, trg_ims = random_rotate(src_ims, trg_ims)
        # print(type(trg_ims))
        return src_ims, trg_ims

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def random_crop(lr_img, hr_img, hr_crop_size=128):
    lr_crop_size = hr_crop_size
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w
    hr_h = lr_h

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)

def scaling(lr_img, hr_img):
    lr_img = tf.cast(lr_img, tf.float32)
    hr_img = tf.cast(hr_img, tf.float32)
    lr_img = lr_img / norm_val
    hr_img = hr_img / norm_val
    return lr_img, hr_img