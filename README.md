# CSGO Map Encoding and Generation
Contact: tom.c.maurer@gmail.com or tmaurer@ualberta.ca

This repo contains my current work done towards generating csgo maps for balanced matchmaking using pcgml techniques. Csgo is short for Counter Strike: Global Offensive, a tactical first person shooter developed by Valve.

## The high level goal is:
- Encode map and player position data using an autoencoder
- Feed these encodings to a predictor network which predicts a teams win chance
- Alter the map in the encoding to bias the win chance towards a particular team

## Dataset
No dataset is included inside this repo due to size constraints (My current dev dataset is roughly 500GB) however the code to create your own dataset is included. You will need an API key for faceit.com in order to scrape the csgo demos files. FACEIT is a third party matchmaking service available for free at the time of writing this. It includes several different hubs in addition to it's matchmaking. I chose to use demos from the Mythic Hubs as they include matches of various skill levels and only use maps in the current competitive map pool.

## Code
The Code in this repo is primarily written in python (specifically version 3.9), but some of the demo processing is written in golang. The python code is split into 3 sections: Dataset generation, Neural Networks, Utility. 

Dataset generation is the only section that also requires you use golang as I used a library to parse the csgo demos after scraping them. The code in this section is used for scraping csgo demo files, analyzing them, and processing them into a variety of formats.

Neural Networks is where all the code lives that trains or tests the various networks we run. They are all built in python using keras.

Utility contains all the remaining code. Most of this relates to visualization or file management.

## Research Progress
As this is primarily intended as a research repo for academic purposes I will include a separate document which details the steps I have taken so far in this project in the hopes that it helps future collaborators understand the work done so far.

## Installation
Unfortunately I failed to document all the installation steps I took when I was installing this a year ago so I will do my best to list the necessary steps and packages but this may not be fully complete. All the packages should be installable using ```go get``` or ```pip install```.

### Programs:
- Go 1.16
- Python 3.9
- Faceit API access

### Go Packages
- gonpy
- demoinfocs-golang (github.com/markus-wa/demoinfocs-golang/v2)

### Python Packages
- Pillow
- Matplotlib
- Numpy
- Keras
- Tensorflow
- Pandas
- Seaborn
- Sklearn
- Requests
- gzip
- dotenv

