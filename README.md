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

# References
[1] M. Tassano, J. Delon, and T. Veit, "FastDVDnet: Towards Real-Time Video Denoising Without Explicit Motion Estimation," arXiv preprint arXiv:1907.01361, 2019.\
[2] O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation," International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, pp. 234-241, 2015.\
[3] M. Z. Alom, C. Yakopcic, T. M. Taha, and V. K. Asari, "Nuclei Segmentation with Recurrent Residual Convolutional Neural Networks based U-Net (R2U-Net)," NAECON 2018-IEEE National Aerospace and Electronics Conference, IEEE, pp. 228-233, 2018.\
[4] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, B. Glocker, and D. Rueckert, "Attention u-net: Learning where to look for the pancreas," arXiv preprint arXiv:1804.03999, 2018.\
[5] M. Liang, and X. Hu, "Recurrent convolutional neural network for object recognition," Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3367-3375, 2015.

# Acknowledgement
Thanks for [z-bingo](https://github.com/z-bingo/FastDVDNet), [m-tassano](https://github.com/m-tassano/fastdvdnet) and [LeeJunHyun](https://github.com/LeeJunHyun/Image_Segmentation) for sharing their codes.

# Contact
If you have any question, please contact jh0729jh@naver.com
