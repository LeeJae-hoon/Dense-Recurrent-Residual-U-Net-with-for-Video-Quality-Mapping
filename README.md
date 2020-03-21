# A-Network-for-Video-Quality-Mapping
Pytorch Implementation of "Dense Recurrent Residual U-Net with Spatial Attention for Video Quality Mapping"

# Environment
Python 3.7.1\
Pytorch 1.0.1\
TensorboardX (for visualization of loss, PSNR and SSIM)

# Train/Test
Codes for train and test are in train_test.py together.

If you want to start a new training,\
<pre>
<code>
python train_test.py --cuda --restart
</code>
</pre>

Or you can continue the previous training,\
<pre>
<code>
train_test.py --cuda
</code>
</pre>

If you want to test your model,\
<pre>
<code>
train_test.py --cuda --eval True
</code>
</pre>

# Contact
If you have any question, please contact jh0729jh@naver.com
