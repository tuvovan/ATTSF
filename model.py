import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend
from tensorflow.keras.activations import *
from tensorflow_addons.layers import AdaptiveAveragePooling2D


class Net(Model):
    def __init__(self, config):
        super(Net, self).__init__()

        self.attention_filter = config.attention_filter
        self.filter = config.filter
        self.encoder_kernel = config.encoder_kernel
        self.decoder_kernel = config.decoder_kernel
        self.triple_pass_filter = config.triple_pass_filter

    def attention_network(self, I_l, I_h):
        h,w,c = list([I_l.shape[1], I_l.shape[2], I_l.shape[3]])
        concat = tf.concat([I_l, I_h], axis=-1)
        lay1 = Conv2D(self.attention_filter, self.encoder_kernel, padding='same', activation='relu')(concat)
        lay1 = Conv2D(self.attention_filter, self.encoder_kernel, padding='same', activation='relu')(lay1)
        out = Conv2D(c, self.encoder_kernel, padding='same', activation='sigmoid')(lay1)
        return out

    def CA(self, X, i):
        c = list(X.shape)[-1]
        if i == 1:
            cc = c
        else:
            cc = c/8
        gap = GlobalAveragePooling2D()(X)
        d = tf.reshape(gap, shape=(-1,1,1,c))
        d1 = ReLU()(Conv2D(filters=cc, kernel_size=(1,1), kernel_initializer = 'he_normal')(d))
        d_bid = sigmoid(Conv2D(filters=c, kernel_size=(1,1), kernel_initializer = 'he_normal')(d1))

        return X*d_bid

    def SA(self, X):
        gap = tf.reduce_max(X, axis=-1)
        gap = tf.expand_dims(gap, axis=-1)
        gmp = tf.reduce_mean(X, axis=-1)
        gmp = tf.expand_dims(gmp, axis=-1)
        
        ff = Concatenate(axis=-1)([gap, gmp])

        f = Conv2D(1, kernel_size=(1,1), kernel_initializer = 'he_normal')(ff)
        f = sigmoid(f)

        return X * f

    def dual_attention(self, X, i):
        c = list(X.shape)[-1]
        M = Conv2D(c, kernel_size=(3,3), padding='same', kernel_initializer = 'he_normal')(X)
        M = ReLU()(M)
        M = Conv2D(c, kernel_size=(3,3), padding='same', kernel_initializer = 'he_normal')(M)

        ca = self.CA(M, i)
        sa = self.SA(M)

        # ca_X = tf.multiply(ca, X)
        # sa_X = tf.multiply(sa, X)

        concat = Concatenate(axis=-1)([ca, sa])

        concat2 = Conv2D(c, kernel_size=(1,1), kernel_initializer = 'he_normal')(concat)

        return Add()([X, concat2])

    def attention_mask(self, I):
        h,w,c = list([I.shape[1], I.shape[2], I.shape[3]])
        lay1 = Conv2D(c, self.encoder_kernel, padding='same', activation='relu')(I)
        lay2 = Conv2D(c, self.encoder_kernel, padding='same', activation='sigmoid')(lay1)
        out = tf.math.multiply(I, lay2)
        return out

    def adaptive_interpolation(self, required_size, img):
        h, w, c = list([img.shape[1], img.shape[2], img.shape[3]])

        pool_size = (int(required_size[0]/img.shape[1]), int(required_size[1]/img.shape[2]))
        img = UpSampling2D(size=pool_size)(img) 
        img = Conv2D(c, self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(img)
        return img

    def encoder_1_1(self, X, i=1):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        # X = MaxPooling2D()(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1

    def encoder_1_2(self, X, i=2):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        X1 = MaxPooling2D()(X1)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1

    def encoder_1_3(self, X, i=4):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        X1 = MaxPooling2D()(X1)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1

    def encoder_2_1(self, X, i=1):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        # X = MaxPooling2D()(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1

    def encoder_2_2(self, X, i=2):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        X1 = MaxPooling2D()(X1)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1

    def encoder_2_3(self, X, i=4):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        X1 = MaxPooling2D()(X1)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1

    def encoder_last_1(self, X, i=8):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        X1 = MaxPooling2D()(X1)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1 

    def encoder_last_2(self, X, i=8):
        # X = self.attention_mask(X)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X)
        X1 = MaxPooling2D()(X1)
        X1 = Conv2D(int(self.filter*i), self.encoder_kernel, padding='same', kernel_initializer = 'he_normal', activation='relu')(X1)
        return X1    

    def decoder_1(self, X, i=4):
        X = UpSampling2D(size= (2,2))(X)
        X = Conv2D(int(self.filter*i), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        X = Conv2D(int(self.filter*i), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        return X

    def decoder_2(self, X, i=2):
        X = UpSampling2D(size= (2,2))(X)
        X = Conv2D(int(self.filter*i), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        X = Conv2D(int(self.filter*i), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        return X

    def decoder_3(self, X, i=1):
        X = UpSampling2D(size= (2,2))(X)
        X = Conv2D(int(self.filter*i), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        X = Conv2D(int(self.filter*i), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        return X

    def decoder_last(self, X):
        # X = UpSampling2D(size= (2,2))(X)
        X = Conv2D(self.filter, self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        X = Conv2D(self.filter, self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(X)
        X = LeakyReLU()(X)
        return X


    def triplepass(self, T0):
        T1 = Conv2D(self.triple_pass_filter, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer = 'he_normal')(T0)
        T1 = ReLU()(T1)

        T2 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer = 'he_normal')(T0)
        T2 = ReLU()(T2)

        T3 = Conv2D(self.triple_pass_filter, kernel_size=(5,5), strides=(1,1), padding='same', kernel_initializer = 'he_normal')(T0)
        T3 = ReLU()(T3)

        T3_ = tf.concat([T1, T2, T3], axis=-1)

        T4 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer = 'he_normal')(T3_)
        T4 = ReLU()(T4)
        T5 = Add()([T4, T0])

        T5 = Conv2D(self.triple_pass_filter, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer = 'he_normal')(T5)
        T5 = ReLU()(T5)
        return T5

    def global_non_local(self, X):
        h, w , c = list(X.shape)[1], list(X.shape)[2], list(X.shape)[3]
        theta = Conv2D(256, kernel_size=(1,1), padding='same', kernel_initializer = 'he_normal')(X)
        theta = ReLU()(theta)
        theta_rsh = Reshape((h*w, 256))(theta)

        phi = Conv2D(256, kernel_size=(1,1), padding='same', kernel_initializer = 'he_normal')(X)
        phi = ReLU()(phi)
        phi_rsh = Reshape((256, h*w))(phi)

        g = Conv2D(256, kernel_size=(1,1), padding='same', kernel_initializer = 'he_normal')(X)
        g = ReLU()(g)
        g_rsh = Reshape((h*w, 256))(g)

        theta_phi = tf.matmul(theta_rsh, phi_rsh)
        theta_phi = tf.keras.layers.Softmax()(theta_phi)

        theta_phi_g = tf.matmul(theta_phi, g_rsh)
        theta_phi_g = Reshape((h, w, 256))(theta_phi_g)

        theta_phi_g = Conv2D(self.triple_pass_filter, kernel_size=(1,1), padding='same', kernel_initializer = 'he_normal')(theta_phi_g)
        theta_phi_g = ReLU()(theta_phi_g)
        out = Add()([theta_phi_g, X])

        return out

    def main_model(self, X):
        ## attention network
        X_i = X[:,:,:,:3]
        X_r = X[:,:,:,3:]

        # X_i = Conv2D(self.filter, (1,1), padding='same', kernel_initializer='he_normal')(X_i)
        # X_r = Conv2D(self.filter, (1,1), padding='same', kernel_initializer='he_normal')(X_r)

        X_1_masked = self.dual_attention(X_i, i=1)
        X_2_masked = self.dual_attention(X_r, i=1)

        # mask1 = self.attention_network(X_i, X_r)
        # mask2 = self.attention_network(X_r, X_i)
        # X_1_masked = tf.math.multiply(mask1, X_i)
        # X_2_masked = tf.math.multiply(mask2, X_r)

        X_i_64 = self.encoder_1_1(X_1_masked)
        X_r_64 = self.encoder_2_1(X_2_masked)

        X_1_1_masked = self.dual_attention(X_i_64, i=2)
        X_2_1_masked = self.dual_attention(X_r_64, i=2)

        # mask1_1 = self.attention_network(X_i_64, X_r_64)
        # mask2_1 = self.attention_network(X_r_64, X_i_64)
        # X_1_1_masked = tf.math.multiply(mask1_1, X_i_64)
        # X_2_1_masked = tf.math.multiply(mask2_1, X_r_64)


        X_i_128 = self.encoder_1_2(X_1_1_masked)
        X_r_128 = self.encoder_2_2(X_2_1_masked)

        X_1_2_masked = self.dual_attention(X_i_128, i=2)
        X_2_2_masked = self.dual_attention(X_r_128, i=2)

        # mask1_2 = self.attention_network(X_i_128, X_r_128)
        # mask2_2 = self.attention_network(X_r_128, X_i_128)
        # X_1_2_masked = tf.math.multiply(mask1_2, X_i_128)
        # X_2_2_masked = tf.math.multiply(mask2_2, X_r_128)

        X_i_256 = self.encoder_1_3(X_1_2_masked)
        X_r_256 = self.encoder_2_3(X_2_2_masked)

        X_1_3_masked = self.dual_attention(X_i_256, i=2)
        X_2_3_masked = self.dual_attention(X_r_256, i=2)

        # mask1_3 = self.attention_network(X_i_256, X_r_256)
        # mask2_3 = self.attention_network(X_r_256, X_i_256)
        # X_1_3_masked = tf.math.multiply(mask1_3, X_i_256)
        # X_2_3_masked = tf.math.multiply(mask2_3, X_r_256)


        X_i_512 = self.encoder_last_1(X_1_3_masked)
        X_r_512 = self.encoder_last_2(X_2_3_masked)

        X_1_4_masked = self.dual_attention(X_i_512, i=2)
        X_2_4_masked = self.dual_attention(X_r_512, i=2)

        # mask1_4 = self.attention_network(X_i_512, X_r_512)
        # mask2_4 = self.attention_network(X_r_512, X_i_512)
        # X_1_4_masked = tf.math.multiply(mask1_4, X_i_512)
        # X_2_4_masked = tf.math.multiply(mask2_4, X_r_512)

        encoder_cat = tf.concat([X_1_4_masked, X_2_4_masked], axis=-1)
        # encoder_cat = Dropout(0.2)(encoder_cat)
        # encoder_last = tf.nn.depth_to_space(encoder_cat, 2)
        encoder_last = Conv2D(self.triple_pass_filter, kernel_size=(3,3), padding='same', kernel_initializer = 'he_normal', activation='relu')(encoder_cat)
        encoder_last = Conv2D(self.triple_pass_filter, kernel_size=(3,3), padding='same', kernel_initializer = 'he_normal', activation='relu')(encoder_last)
        encoder_last = self.dual_attention(encoder_last, i=2)

        ## upper path ##
        tpl_out = self.triplepass(encoder_last)
        tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        # tpl_out = self.triplepass(tpl_out)
        
        # tpl_out = self.triplepass(tpl_out)

        ## lower path ##
        glb_out = AdaptiveAveragePooling2D(output_size=(7,7))(encoder_last)
        glb_out = self.global_non_local(glb_out)
        required_size = [encoder_last.shape[1], encoder_last.shape[2]]
        glb_out = self.adaptive_interpolation(required_size, glb_out)

        # glb_out = self.adapmaxpooling(encoder_last, 16)
        # glb_out = self.global_non_local(glb_out)
        # required_size = [encoder_last.shape[1], encoder_last.shape[2]]
        # glb_out = self.adapupsampling(required_size, glb_out)


        ## cat ##
        merger = tf.concat([tpl_out, glb_out, X_1_4_masked, X_2_4_masked], axis=-1)
        merger = Conv2D(512, self.decoder_kernel, padding='same',  kernel_initializer = 'he_normal', activation='relu')(merger)
        merger = Conv2D(256, self.decoder_kernel, padding='same',  kernel_initializer = 'he_normal', activation='relu')(merger)

        # merger = Dropout(0.2)(merger)

        O_256 = self.decoder_1(merger)
        O_256 = tf.concat([X_1_3_masked, X_2_3_masked, O_256], axis=-1)
        O_256 = Conv2D(int(self.filter*4), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(O_256)
        O_256 = LeakyReLU()(O_256)

        O_128 = self.decoder_2(O_256)
        O_128 = tf.concat([X_1_2_masked, X_2_2_masked, O_128], axis=-1)
        O_128 = Conv2D(int(self.filter*2), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(O_128)
        O_128 = LeakyReLU()(O_128)

        O_64 = self.decoder_3(O_128)
        O_64 = tf.concat([X_1_1_masked, X_2_1_masked, O_64], axis=-1)
        O_64 = Conv2D(int(self.filter*1), self.decoder_kernel, padding='same', kernel_initializer = 'he_normal')(O_64)
        O_64 = LeakyReLU()(O_64)

        O_64 = self.decoder_last(O_64)
        O_3 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer = 'he_normal')(O_64)
        O_3 = Conv2D(3, (3,3), padding='same', activation='relu', kernel_initializer = 'he_normal')(O_3)

        O_9 = tf.concat([X_1_masked, X_2_masked, O_3], axis=-1)
        O_10 = Conv2D(9, (3,3), padding='same', activation='relu', kernel_initializer = 'he_normal')(O_9)
        out = Conv2D(3, (1,1), padding='same', activation='sigmoid', kernel_initializer = 'he_normal')(O_10)

        return out