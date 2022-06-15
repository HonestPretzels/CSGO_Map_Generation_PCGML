# Neural Network Code

This folder contains all the code necessary to run the neural networks. Train and test are split up into different files. At the time of writing this the predictor is still a work in progress. Many of the network architectures I used required the dataset to be split accross several files in order to load everything into memory.

## Autoencoder
This has 3 different files:
1. train_autoencoder.py - This file is used for training the autoencoder
2. test_autoencoder.py - This file lets you visualize the output of the autoencoder
3. encoder.py - This file can be used to generate an encoded latent-space from the input data