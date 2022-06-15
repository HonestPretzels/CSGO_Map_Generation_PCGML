# Utility Files

## File Merger
fileMerger.py concatenates numpy files with the same shape and saves them into a single file

## File Resizer

fileResizer.py takes one large numpy file and splits it into several smaller files. This is sometimes needed in order to load the datasets into memory for the neural networks.

## Map Resizer
map_resizer.py takes larger image files and scales them down to lower resolution versions of themselves. This was used to reduce minimap image size.

## Reshape
reshape.py is a catch all file that I used to reshape numpy files as needed. The code in here changed all the time based on what I needed at any given moment.

## Test Loader
test_loader.py is very similar to reshape.py in that the code changed all the time depending on the need. This file is used to load numpy files, confirm their shape, check what the values are, visualize them etc.

## T-SNE Visualization
tSneVisualizer.py is used to perform tsne visualization on the latent space. This was used to see how well the autoencoder encoded different aspects of the input by visualizing the latent space clustering.