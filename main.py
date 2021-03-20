"""
This is the main module for linking different components of the CNN-based model
proposed for the task of image defocus deblurring based on dual-pixel data. 

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
from model import *
from config import *
from data import *

check_dir(path_write)

def train(configure):
    if op_phase=='train':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        data_random_shuffling('train')
        data_random_shuffling('val')
        model_x = Net(configure)

        in_data = Input(batch_shape=(None, patch_h, patch_w, nb_ch_all))
        
        model = Model(inputs=in_data, outputs=model_x.main_model(in_data))
        model.summary()
        model.compile(optimizer = Adam(lr = lr__[0]), loss = loss_function_2)
        
        # training callbacks
        model_checkpoint = ModelCheckpoint(path_save_model, monitor='val_loss',
                                verbose=1, save_best_only=True)

        l_r_scheduler_callback = LearningRateScheduler(schedule=lr_scfn, verbose=True) 
        
        history = model.fit_generator(generator('train'), nb_train, nb_epoch,
                            validation_data=generator('val'),
                            validation_steps=nb_val,callbacks=[model_checkpoint,
                            l_r_scheduler_callback])
        
        np.save(path_write+'train_loss_arr',history.history['loss'])
        np.save(path_write+'val_loss_arr',history.history['val_loss'])

    elif op_phase=='test':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        data_random_shuffling('test')


        model_x = Net(configure)
        input_size = (img_h, img_w, nb_ch_all)
        input_test = Input(input_size)
        output_test=model_x.main_model(input_test)
        model = Model(inputs = input_test, outputs = output_test)

        model.load_weights('ModelCheckpoints/defocus_deblurring_dp_l5_s512_f0.7_d0_64_14_dual_attention_big_100_psnr.h5')
        img_mini_b = 1

        test_imgaes, gt_images = test_generator(total_nb_test)
        predictions = model.predict(test_imgaes,img_mini_b,verbose=1)
                                
        save_eval_predictions(path_write,test_imgaes,predictions,gt_images)
        
        print('PSNR: ', np.mean(np.asarray(psnr_list)))
        print('MSE: ', np.mean(np.asarray(mse_list)))
        print('SSIM: ', np.mean(np.asarray(ssim_list)))

        np.save(path_write+'mse_arr',np.asarray(mse_list))
        np.save(path_write+'psnr_arr',np.asarray(psnr_list))
        np.save(path_write+'ssim_arr',np.asarray(ssim_list))
        np.save(path_write+'mae_arr',np.asarray(mae_list))
        np.save(path_write+'final_eval_arr',[np.mean(np.asarray(mse_list)),
                                            np.mean(np.asarray(psnr_list)),
                                            np.mean(np.asarray(ssim_list)),
                                            np.mean(np.asarray(mae_list))])

    elif op_phase=='valid':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        data_random_shuffling('validation')

        model_x = Net(configure)
        input_size = (img_h, img_w, nb_ch_all)
        input_test = Input(input_size)
        output_test=model_x.main_model(input_test)
        model = Model(inputs = input_test, outputs = output_test)
        model.load_weights('ModelCheckpoints/weights.h5')
        img_mini_b = 1

        test_imgaes = validation_generator(total_nb_test)
        import time
        t = time.time()
        predictions = model.predict(test_imgaes,img_mini_b,verbose=1)
        print((time.time()-t)/total_nb_test)
                                
        save_eval_comp(path_write_comp,test_imgaes,predictions)

if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()

	# Input Parameters

	parser.add_argument('--filter', type=int, default= 64)
	parser.add_argument('--attention_filter', type=int, default= 64)
	parser.add_argument('--kernel', type=int, default= 3)
	parser.add_argument('--encoder_kernel', type=int, default= 3)
	parser.add_argument('--decoder_kernel', type=int, default= 3)
	parser.add_argument('--triple_pass_filter', type=int, default= 512)

	configure = parser.parse_args()

	train(configure)