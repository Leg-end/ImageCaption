# ImageCaption
This a image-to-caption model based on 'show and tell'https://github.com/tensorflow/models/tree/master/research/im2txt/im2txt written for learning purposes

The program structure is base on Tensorflow's 'nmt'https://github.com/tensorflow/nmt
# DataSet
MCOCO 2014
# Pre-trained model
Google inception-v3

# Environment
tensorflow-gpu: 1.9.0
CUDA: 9.0.176
CUDNN: 7.3.0
GPU: Nvida RTX-2070

# Model Structure
![image](https://user-images.githubusercontent.com/31924601/221394525-086c9f95-e07b-420b-83ff-945b14c4d660.png)

# Optimization Assumption
Max Likelihood Assumption

![image](https://user-images.githubusercontent.com/31924601/221394793-02151a82-8a39-41ea-97e9-6b7fe8f939cb.png)

![image](https://user-images.githubusercontent.com/31924601/221394800-8798f22d-b296-43a4-aba0-9de8a295a792.png)
