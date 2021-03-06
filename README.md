# Dense Recurrent Residual U-Net with Spatial Attention for Video Quality Mapping
Pytorch Implementation of "Dense Recurrent Residual U-Net for Video Quality Mapping"

# Environment
Python 3.7.1\
Pytorch 1.0.1\
TensorboardX (for visualization of loss, PSNR and SSIM)

# Training
If you want to train your own model,
<pre>
<code>
python train.py
</code>
</pre>

### Notes
- The path to the corrupted training dataset has to be stored at the argument "training_source".
- The path to the ground truth training datset has to be store at the argument "training_target".
- Likewise, the path to other dataset such as validation source/ target has to be stored at the matching argument.
- If you want to continue the training, set 'restart' argument to 'True' and put the pretrained model in the folder named 'models'.

# Testing
If you want to test a model,
<pre>
<code>
python test.py
</code>
</pre>

### Notes
- If you want to use my pretrained model, click [here](https://drive.google.com/open?id=1qJh1lgADqUO8PImGfndj71ZJE9aM5_g2) to download.
- Put the pretrained model in the folder named 'models'.
- The path to the test dataset has to be stored at the "test_source" argument. 
- Check the "save_path" argument. It keeps the path of the directory where you want to save test results.


# References
[1] M. Tassano, J. Delon, and T. Veit, "FastDVDnet: Towards Real-Time Video Denoising Without Explicit Motion Estimation," arXiv preprint arXiv:1907.01361, 2019.\
[2] O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation," International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, pp. 234-241, 2015.\
[3] M. Z. Alom, C. Yakopcic, T. M. Taha, and V. K. Asari, "Nuclei Segmentation with Recurrent Residual Convolutional Neural Networks based U-Net (R2U-Net)," NAECON 2018-IEEE National Aerospace and Electronics Conference, IEEE, pp. 228-233, 2018.\
[4] M. Liang, and X. Hu, "Recurrent convolutional neural network for object recognition," Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3367-3375, 2015.

# Acknowledgement
Thanks for [z-bingo](https://github.com/z-bingo/FastDVDNet), [m-tassano](https://github.com/m-tassano/fastdvdnet) and [LeeJunHyun](https://github.com/LeeJunHyun/Image_Segmentation) for sharing their codes.

# Contact
If you have any question, feel free to contact dlwogns0729@gmail.com
