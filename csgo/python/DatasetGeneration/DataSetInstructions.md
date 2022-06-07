# Dataset Generation Pipeline

1. Run FaceitLogScraper.py to get demo and meta data files
2. Run demoParser.go to get sequences of seconds and round win data
3. Run demo_parse_loader.py to get 30 second splits (data (num_splits, 30_seconds, 10_players, 23_data_points)), maps assigned to each split (maps (num_splits)), round winners (scores (num_splits)) for each split,
   and breakpoints that show which splits belong to which game (breakspoints (num_games, num_splits_per_round))
4. Run  splits_to_images.py to convert to full data arrays
5. Run layer_reduce.py to convert to reduced format. Use False to create data and True to create labels. This will be the dataset used to train the autoencoder
6. Pass the reduced layers to zeroRemover.py to generate remove 30 seconds splits with no players in them.
6. Run test_autoencoder.py to encode a dataset into a latent space