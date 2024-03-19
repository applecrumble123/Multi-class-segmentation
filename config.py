import os

#ROOT_FOLDER_PATH = "/home/johnathon/Desktop/test_dir"
#RAW_IMAGE_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "raw_images")
#MASKED_IMAGES_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "masked_images")
ROOT_FOLDER_PATH = "/home/johnathon/Desktop/"
RAW_IMAGE_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "red_set_2/image_red_2")
MASKED_IMAGES_FOLDER_PATH = os.path.join(ROOT_FOLDER_PATH, "red_set_2/mask_red_2")
SAVED_MODEL_FOLDER = "/home/johnathon/Desktop/multiclass_segmentation/saved_model"

########################## train.py ##############################
MEAN_BATCH_TRAINING_LOSS_TEXT_FILE_PATH = "/home/johnathon/Desktop/multiclass_segmentation/mean_batch_training_loss.txt"
SAVED_SERIAL_MODEL_FOLDER = "/home/johnathon/Desktop/multiclass_segmentation/saved_serial_model"
BEST_WEIGHTS_MODEL_FOLDER_PATH = "/home/johnathon/Desktop/multiclass_segmentation/best_model_weights"
LOWEST_MEAN_BATCH_LOSS_TEXT_FILE_PATH = "/home/johnathon/Desktop/multiclass_segmentation/lowest_mean_batch_loss.txt"
LOWEST_VALIDATION_LOSS_TEXT_FILE_PATH = "/home/johnathon/Desktop/multiclass_segmentation/lowest_validation_loss.txt"

EPOCH_NUM = 3000
BATCH_SIZE = 4
NUM_OF_WORKERS = 2

############################ test.py ##############################
#SAVED_MODEL_NAME = "epoch_257_model.pt"
SAVED_MODEL_NAME = "segmentation_model_epoch_last.pth"
