import os

ROOT_FOLDER_PATH = "/home/johnathon/Desktop/test_dir"
RAW_IMAGE_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "raw_images")
MASKED_IMAGES_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "masked_images")
SAVED_MODEL_FOLDER = "/home/johnathon/Desktop/multi_segmentation/saved_model"

########################## train.py ##############################
MEAN_BATCH_TRAINING_LOSS_TEXT_FILE_PATH = "/home/johnathon/Desktop/multi_segmentation/mean_batch_training_loss.txt"
SAVED_SERIAL_MODEL_FOLDER = "/home/johnathon/Desktop/multi_segmentation/saved_serial_model"
BEST_WEIGHTS_MODEL_FOLDER_PATH = "/home/johnathon/Desktop/multi_segmentation/best_model_weights"
LOWEST_MEAN_BATCH_LOSS_TEXT_FILE_PATH = "/home/johnathon/Desktop/multi_segmentation/lowest_mean_batch_loss.txt"
LOWEST_VALIDATION_LOSS_TEXT_FILE_PATH = "/home/johnathon/Desktop/multi_segmentation/lowest_validation_loss.txt"

BATCH_SIZE = 2
NUM_OF_WORKERS = 2

############################ test.py ##############################
SAVED_MODEL_NAME = "epoch_52_model.pt"