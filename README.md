
# 🖼️Image Outpainting Using Wasserstein Generative Adversarial Network with Gradient Penalty🎨

We are excited to share that our research has been successfully published in the [IEEE ICCMC Conference](https://ieeexplore.ieee.org/document/9753713) held on 03/22. This paper serves as a significant milestone in our ongoing work and contributes to the body of knowledge in this field.

**Abstract**: In the realm of AI-driven image generation, Image Inpainting has garnered significant attention for completing missing data within images. Conversely, the field of Image Outpainting, which involves extending images beyond their borders, remains relatively unexplored.

In Image Outpainting, the key challenge lies in establishing spatial correlation between the generated image and the ground truth image. Training instability of Generative Adversarial Networks (GANs) can hinder this process. Wasserstein GAN (WGAN) with Gradient Penalty (WGAN-GP) provides a solution to this issue.

Our proposed model leverages the WGAN-GP algorithm and deep convolutional neural networks for image outpainting using a dataset of natural images. Our findings demonstrate that WGAN-GP surpasses GAN in various aspects, making it a promising approach for image outpainting.


## Dataset

We have collected the dataset of 4300 images from google 
images and kaggle datasets of natural landscapes. Due to a 
lack of computational resources, we have opted to train and 
test our model on an image size of 128x128 rather than taking 
higher resolution images like 512x512. Out of the 4300 
images, 60 of them were for testing and the other 4240 
images were fed to the model for training. Our dataset 
initially consisted of a wide variety of natural landscapes 
ranging from mountains to lake views, sea, and farmlands, 
but was of different sizes each. So we first resized all of the 
images to 128x128 size for easy training.



## Proposed System 

![App Screenshot](https://github.com/tarundirector/Image-Outpainting-Using-Wasserstein-Generative-Adversarial-Network-with-Gradient-Penalty/assets/85684655/0c21515e-6a20-4704-a16d-b8eaf66ff87e)

Our project involves several key stages to achieve our goals:

**A. Collecting Dataset**
We have collected the dataset of 4300 images from Google Images and Kaggle datasets of natural landscapes. Due to a lack of computational resources, we have opted to train and test our model on an image size of 128x128 rather than taking higher resolution images like 512x512. Out of the 4300 images, 60 of them were for testing, and the other 4240 images were fed to the model for training. Our dataset initially consisted of a wide variety of natural landscapes ranging from mountains to lake views, sea, and farmlands, but was of different sizes each. So we first resized all of the images to 128x128 size for easy training.

**B. Image Preprocessing**
Before feeding images to the model, we first go through the preprocessing stage. We have followed a similar preprocessing pipeline as presented by [7]. First, the images are normalized to a {0,1} 128x128x3. After normalizing we mask the image. Masking basically center crops the given image. The mask we applied was the same as defined in [7] mask M ∈ {0, 1} 128×128 such that Mij = 1 − 1[32 ≤ j < 96], this mask is for images of size 128x128. One can change the mask according to the size of the images in the training dataset.

**C. Training Phase**
We have followed a training pipeline similar to the one presented in [7]. The training of the model is divided into three phases. The first phase (18% of total iterations) involves training the generator network alone. The next 2% of the training iterations will involve training the critic network alone. The remaining iterations will include training of generators and critic network adversarial manners. We have adopted the Wasserstein GAN model with Critic (C) and Generator (G) networks. In each iteration, we create a small sample mini-batch of training data. Each image I in the batch goes through the preprocessing pipeline. This processed image I’ is fed to the generator, which generates an outpainted image Io = G(I’) ∈ {0,1}128x128x3. We then run the critic network to classify ground truth (I) and outpainted image (Io). The losses are computed according to the loss function discussed in the next section, and the parameters are updated according to the training pipeline.

**D. Testing the Model**
In this stage, unknown images from different sources like google are gathered and fed to the WGAN architecture, and the results of the generated image are analyzed. We analyzed the images generated by WGAN by comparing its results with the images generated by GAN.




## Network Architechture
We have opted for a shallower convolutional network for the generator and discriminator as proposed by [7], as a deeper and more complex network would cost us computationally, which is not feasible for us at this point in time. The generator network consists of an encoder-decoder structure and dilated convolutions to expand the convolutional kernels without making generator computation expensive. Dilated convolution also helps generators to generate more realistic images.

As for the critic network, we have tried to make the network deeper compared to the generator network, as the more complex the critic network, the better the generator can learn to generate the image. The critic network and generator network are shown in Fig 3 and Fig 4.

![App Screenshot](https://github.com/tarundirector/Image-Outpainting-Using-Wasserstein-Generative-Adversarial-Network-with-Gradient-Penalty/assets/85684655/ddc7b30d-06f8-4703-ae5d-39de15313851)

![App Screenshot](https://github.com/tarundirector/Image-Outpainting-Using-Wasserstein-Generative-Adversarial-Network-with-Gradient-Penalty/assets/85684655/aeb7499f-f3d6-45b5-bb85-879aac25fddc)


## Results

We trained our critic-generator architecture on the given dataset of natural landscapes in a Three-Phase training pipeline as mentioned before, where total iterations N= 212000, phase 1 = 38160 iterations; phase 2 = 4240 iterations, and phase 3 = 169600 iterations. Using a batch size of 16, we trained the architecture for 800 epochs on the Tesla T4 GPU of Nvidia provided by Google Colab.

To improve the quality of the generated images, after training, we blended the outpainted image with the unmasked portion of the actual image. The results after this process are shown in Fig 6. We also analyzed the MSE losses of training and test images and plotted the loss in a graph shown in Fig 5.

The graph involves the following things:
- Phase 1: The red region depicts the first phase, which is training the generator based on the MSE loss. As we can see in the red region, there are drastic changes seen in the training MSE loss (shown by Red Line) and validation MSE loss (shown by orange lines).
- Phase 2: This phase involves training the critic model. As we can see the changes in MSE values are almost negligible as the generator is not learning in this phase.
- Phase 3: This involves the adversarial training of critic and generator so you can see the MSE values fluctuating although it is negligible to some extent as seen in the graph.

![App Screenshot](https://github.com/tarundirector/Image-Outpainting-Using-Wasserstein-Generative-Adversarial-Network-with-Gradient-Penalty/assets/85684655/d0ea863c-f72a-4d18-8cbc-0dfbbaeea212)

From Table 1, you can see the images generated using WGAN-GP and GAN algorithm respectively. As mentioned earlier, the smaller the RMSE value, the better is the picture quality of the generated image, and as we can see WGAN-GP algorithm has a lesser RMSE value compared to GAN, we can say WGAN-GP provides better results although there is still room for improvement in the quality of generated images.

![App Screenshot](https://github.com/tarundirector/Image-Outpainting-Using-Wasserstein-Generative-Adversarial-Network-with-Gradient-Penalty/assets/85684655/d8d0e440-264f-478c-86bf-471b6286dc13)

![App Screenshot](https://github.com/tarundirector/Image-Outpainting-Using-Wasserstein-Generative-Adversarial-Network-with-Gradient-Penalty/assets/85684655/fd4df3d0-bbdc-4f6e-ba44-850be5f1ca77)

## Conclusion

We have successfully implemented image outpainting using WGAN-GP. The results from the training were fairly realistic but could be improved further with a large and diverse dataset and a more complex critic-generator network. The three-phase training has proven to help reach the optimal critic and generator losses that, in turn, improve the quality of generated images.

## Acknowledgements

 - [1] Basile Van Hoorick, Image Outpainting and Harmonization using Generative Adversarial Networks,2019, arXiv :1912.10960 
- [2] L. Zhang, J. Wang and J. Shi, "Multimodal Image Outpainting with Regularized Normalized Diversification," 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), 2020, pp. 3422-3431, doi: 10.1109/WACV45572.2020.9093636.
- [3] QINGGUO XIAO, GUANGYAO LI, AND QIAOCHUAN CHEN, Image Outpainting: Hallucinating Beyond the Image, IEEE Access 8 (2020),doi: 10.1109/ACCESS.2020.3024861
- [4] Bhadoriya, Shailendra & Aggarwal, Nainish & Jain, Udit & Jaiswal, Hrithik, Outpainting Images and Videos using GANs, International Journal of Computer Trends and Technology,2020, doi: 10.14445/22312803/IJCTT-V68I5P107
- [5] Y. Wang, X. Tao, X. Shen and J. Jia, "Wide-Context Semantic Image Extrapolation," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1399-1408, doi: 10.1109/CVPR.2019.00149.
- [6] Z. Yang, J. Dong, P. Liu, Y. Yang and S. Yan, "Very Long Natural Scenery Image Prediction by Outpainting," 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 10560-10569, doi: 10.1109/ICCV.2019.01066.
- [7] Mark Sabini, Gili Rusak: “Painting Outside the Box: Image Outpainting with GANs”, 2018; arXiv:1808.08483.
- [8] Cheng, Yen-Chi & Lin, Chieh & Lee, Hsin-Ying & Ren, Jian & Tulyakov, Sergey & Yang, Ming-Hsuan. In&Out: Diverse Image Outpainting via GAN Inversion, (2021), arXiv:2104.00675
- [9] K. Kim, Y. Yun, K. -W. Kang, K. Kong, S. Lee and S. -J. Kang, "Painting Outside as Inside: Edge Guided Image Outpainting via Bidirectional Rearrangement with Progressive Step Learning, 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 2121-2129, doi: 10.1109/WACV48630.2021.00217.
- [10] Martin Arjovsky, Soumith Chintala, Léon Bottou:“Wasserstein GAN”, 2017; arXiv:1701.07875
- [11] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville: “Improved Training of Wasserstein GANs”, 2017; arXiv:1704.00028.
- [12] Saxena, Sanjay & Sharma, Shiru & Sharma, Neeraj. (2016). Parallel Image Processing Techniques, Benefits and Limitations. Research Journal of Applied Sciences, Engineering and Technology. 12. 223-238. 10.19026/rjaset.12.2324. 
- [13] Jacob, I. Jeena, and P. Ebby Darney. “Design of Deep Learning Algorithm for IoT Application by Image based Recognition.” Journal of ISMAC 3, no. 03 (2021): 276-290, doi: //doi.org/10.36548/jismac.2021.3.008
