# Data Lists & Data Configuration

This folder contains text files defining the training and validation splits for the experiments.

## Data List Format
The `.txt` files in this directory (e.g., inside `deap/`) contain lists of filenames that correspond to the data files.

**Example Entry:**
`15_36_3.04_5.04_1.9.mat`

## Data Storage & Configuration
The actual `.mat` data files should be stored in a data directory (e.g., `../data`). 

**File Naming Convention:**
`{SubjectID}_{TrialID}_{Valence}_{Arousal}_{Dominance}.mat`
(e.g., `15_36_3.04_5.04_1.9.mat`)

> **Configuration**: The path to the data directory is specified in the configuration files (e.g., `configs/ssl_example.cfg` or `configs/train_example.cfg`). You can modify the data path there.
