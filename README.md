# A-Network-for-Video-Quality-Mapping
Pytorch Implementation of "Dense Recurrent Residual U-Net with Spatial Attention for Video Quality Mapping"

# Environment
Python 3.7.1\
Pytorch 1.0.1\
TensorboardX (for visualization of loss, PSNR and SSIM)

# Train/Test
Codes for train and test are in train_test.py together.

If you want to start a new training,\
'''Python
python train_test.py --cuda --restart
'''

Or you can continue the previous training,\
'''Python 
train_test.py --cuda
'''

If you want to test your model,\
'''Python 
train_test.py --cuda --eval True
'''

