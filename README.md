# Cycle GAN for Image Denoising and Trajectory Reconstruction

## Project Overview
This project involves training a Cycle GAN, composed of 2 generators and 2 discriminators, with the goal of denoising images and reconstructing segmented trajectories. The project utilizes two datasets:

1. **MNIST Dataset**: Used for the image denoising task, with 1000 training examples, 1000 validation examples, and 1000 test examples.
2. **Trajectory Dataset**: Used for the trajectory reconstruction task, with 5000 training examples, 500 validation examples, and 500 test examples.

## Network Architecture
The network architecture consists of:

- **Input**: Images of size (28,28,1) or (128,128,1)
- **Initial Block**: Conv2D, Normalization, ReLU
- **Downsampling**: Two Conv2D layers, Normalization, ReLU
- **ResNet Blocks**: 6 or 9 blocks
- **Upsampling**: Two Conv2DTranspose layers, Normalization, ReLU
- **Output**: Conv2D, 'tanh' activation

The discriminator architecture consists of:
- **Input**: Images of size (28,28,1) or (128,128,1)
- **Conv Layers**: Conv2D, LeakyReLU, Dropout (twice)
- **Final Output**: Flatten, Dense with 1 output

## Losses
The model is trained using the following losses:
- 2 Losses for the generators
- 2 Losses for the discriminators
- Cycle consistency loss
- Identity loss

## Hyperparameters
The hyperparameters used for the two tasks are:

**MNIST Parameters**:
- Batch size: 1
- Learning rate: 2E-4 (Generators), 1E-4 (Discriminators)
- Number of epochs: 2
- Cycle lambda: 10
- 
**Trajectory Parameters**:
- Batch size: 1
- Learning rate: 1E-4 (Generators), 1E-4 (Discriminators)
- Number of epochs: 2
- Cycle lambda: 10
## Results and Analysis
The initial results for the MNIST dataset were not satisfactory due to using a large batch size. With a batch size of 1, the model achieved remarkable results after just one epoch.

For the trajectory task, the results were not as remarkable as those obtained with the MNIST dataset, but they were not entirely disappointing either. The difficulty was in creating the function that segments the trajectory and finding the right parameter configuration.

## Conclusion
This project was a valuable learning experience, as it was the first time I implemented a Cycle GAN. Despite the initial difficulties, I am happy to have chosen this project, as it aligns with my studies in artificial intelligence. The process was both interesting and fun.
