# config.py

# Paths
DATA_DIR = r'D:\Ai courses\DL mostafa saad\project\data set\Brain Tumor Data Set\Brain Tumor Data Set'
CSV_PATH = r'D:\Ai courses\DL mostafa saad\project\data set\metadata_rgb_only.csv'

# Image settings
IMAGE_SIZE = (224, 224)

# Training settings
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = False

# Hyperparameters
params = {
    "initial_filters": 32,
    "num_fc1": 256,
    "num_classes": 2,
    "dropout_rate": 0.5
}



# Model settings
NUM_CLASSES = 2  

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")