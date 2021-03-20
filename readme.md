# Deep HDR Imaging
Solution of Defocus Deblurring Challenge - [Attention! Stay Focus! (ATTSF)](https://competitions.codalab.org/competitions/28049#results) - NTIRE 2021
## Content
- [Deep-HDR-Imaging](#attention-stay-focus)
- [Getting Started](#getting-started)
- [Running](#running)
- [References](#references)
- [Citations](#citation)

## Getting Started

- Clone the repository

### Prerequisites

- Tensorflow 2.2.0+
- Tensorflow_addons
- Python 3.6+
- Keras 2.3.0
- PIL
- numpy


## Running
### Training 
- Preprocess
    - Download the [training data](https://ln2.sync.com/dl/66bc64370/u7hy9v4a-qrdjtr8z-xvwtpi2t-7fc2h7yv)

    - Unzip the file

- Train ATTSF 
    - change ```op_phase='train'``` in ```config.py```
    ```
    python main.py
    ```

- Test ATTSF
    - change ```op_phase='valid'``` in ```config.py```
    ```
    python main.py
    ```
## Usage
### Training
```
usage: main.py [-h] [--images_path IMAGES_PATH] [--test_path TEST_PATH]
               [--lr LR] [--gpu GPU] [--num_epochs NUM_EPOCHS] 
               [--train_batch_size TRAIN_BATCH_SIZE]
               [--display_ep DISPLAY_EP] [--checkpoint_ep CHECKPOINT_EP]
               [--checkpoints_folder CHECKPOINTS_FOLDER]
               [--load_pretrain LOAD_PRETRAIN] [--pretrain_dir PRETRAIN_DIR]
               [--filter FILTER] [--kernel KERNEL]
               [--encoder_kernel ENCODER_KERNEL]
               [--decoder_kernel DECODER_KERNEL]
               [--triple_pass_filter TRIPLE_PASS_FILTER]
```

```
optional arguments: -h, --help                show this help message and exit
                    --images_path             training path
                    --lr                      LR
                    --gpu                     GPU
                    --num_epochs              NUM of EPOCHS
                    --train_batch_size        training batch size
                    --display_ep              display result every "x" epoch
                    --checkpoint_ep           save weights every "x" epoch
                    --checkpoints_folder      folder to save weight
                    --load_pretrain           load pretrained model
                    --pretrain_dir            pretrained model folder
                    --filter                  default filter
                    --kernel                  default kernel
                    --encoder_kernel          encoder filter size
                    --decoder_kernel          decoder filter size
                    --triple_pass_filter      number of filter in triple pass
```

### Testing
- Download the weight [here](https://drive.google.com/file/d/1OjJYirwRa8cLGzzdRYRkjq_1FokyI80V/view?usp=sharing) and put it to the folder ```ModelCheckpoints```

#### Result
        Left image         |       Right Image         |        Output
![](results/0501_l.png)    | ![](results/0501_r.png)   | ![](results/0501_g.png)
![](results/0500_l.png)    | ![](results/0500_r.png)   | ![](results/0500_g.png) 
:-------------------------:|:-------------------------:|:-------------------------:
## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/tuvovan/NHDRRNet/blob/master/LICENSE) file for details

## References
[1] Deep HDR Imaging via A Non-Local Network - TIP 2020 [link](https://ieeexplore.ieee.org/document/8989959)

[3] Training and Testing dataset - [link](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)

## Citation
```
    @ARTICLE{8989959,  author={Q. Yan and L. Zhang and Y. Liu and Y. Zhu and J. Sun and Q. Shi and Y. Zhang},  
    journal={IEEE Transactions on Image Processing},   
    title={Deep HDR Imaging via A Non-Local Network},   
    year={2020},  
    volume={29},  
    number={},  
    pages={4308-4322},}
```
## Acknowledgments
- This work based on the paper mentioned above with few modification:
    - the fixed size of the adaptive average pooling (16 instead of 32 as assigned in the paper)
    - the number of triple pass module is defined as 10 to match the number of 32M as stated in the paper.
- Any ideas on updating or misunderstanding, please send me an email: <vovantu.hust@gmail.com>
- If you find this repo helpful, kindly give me a star.

# Update: I have just released my work on HDR imaging using Attention non-local network. Please check as follow: https://github.com/tuvovan/ANL-HDRI
