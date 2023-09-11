import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from PIL import Image
import cv2
from typing import Optional, Callable, Tuple, Any
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

from SQNet import SQNet
from UNet import UNet
from loss import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, FocalLoss2d, LDAMLoss, ProbOhemCrossEntropy2d, LovaszSoftmax, CrossEntropy2d
from AdamW import AdamW
from RAdam import RAdam
from dice_loss import DiceLoss

from PIL import Image

import random

from sklearn.metrics import accuracy_score



device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

"""
print("Is there a GPU available: "),
print(torch.cuda.device_count())
"""

root_folder_path = "/home/johnathon/Desktop/test_dir"
raw_image_folder_path = os.path.join(root_folder_path, "raw_images")
masked_images_folder_path = os.path.join(root_folder_path, "masked_images")

list_of_annotated_images = [0, 10, 100, 1000, 1020, 1040, 1050, 1060, 110, 1100, 
 1130, 1150, 1160, 1170, 1180, 120, 1200, 1220, 1230, 
 1240, 1250, 1260, 1270, 1280, 1290, 130, 1300, 1310, 
 1320, 1330, 1350, 1360, 1370, 1390, 140, 1400, 1420, 
 1430, 1470, 1490, 150, 1500, 1550, 1560, 160, 1600, 
 1620, 1630, 1640, 1660]

################################### Split into train, val, test ###################################

raw_image_path_list = []
masked_images_path_list = []

for i in range(len(os.listdir(raw_image_folder_path))):
    img_path = os.path.join(raw_image_folder_path, str(list_of_annotated_images[i]) + ".png")
    raw_image_path_list.append(img_path)

    mask_img_path = os.path.join(masked_images_folder_path, str(list_of_annotated_images[i]) + "_mask.png")
    masked_images_path_list.append(mask_img_path)

# raw images path
train_raw_image_path_list = raw_image_path_list[:30]
val_raw_image_path_list = raw_image_path_list[30:40]
test_raw_image_path_list = raw_image_path_list[40:]
#print(train_raw_image_list)
#print(val_raw_image_list)
#print(test_raw_image_list)

# masked images path
train_masked_image_path_list = masked_images_path_list[:30]
val_masked_image_path_list = masked_images_path_list[30:40]
test_masked_image_path_list = masked_images_path_list[40:]



# sky, land, sea, ship, buoy, other
CLASS_LIST = ['sky', 'land', 'sea', 'ship', 'buoy', 'other', 'background']
CLASS_LIST_LABEL = [1, 2, 3, 4, 5, 6]
PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], 
           [0, 0, 128], [128, 0, 128], [0, 128, 128], [0, 0, 0]]


def rgb_to_one_hot_encoded_mask(rgb_mask, colormap):

    # shape --> [H,W,C] [960, 1280, 3]
    target = torch.from_numpy(rgb_mask)

    # reshape mask 
    # shape --> [C,H,W] [3,960, 1280]
    target = target.permute(2, 0, 1).contiguous()

    # map each colour to a class indice 
    # class indice starts from 1
    # to create a one-hot encoding structure with class indices
    mapping = {tuple(c): t for c, t in zip(PALETTE, range(1, len(PALETTE)+1))}

    # creates an empty mask array with no channels
    # class indices start from 1
    mask = torch.empty(rgb_mask.shape[0], rgb_mask.shape[1], dtype=torch.long)
    
    # each pixel position in the mask represents the class indices
    for k in mapping:
        # Get all indices for current class
        idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)  # Check that all channels match
        mask[validx] = torch.tensor(mapping[k], dtype=torch.long)

    #print(mask)
    #print(mask.shape)

    # creates a stack of mask for each class
    list_of_mask = []

    # convert tensor mask to numpy array for manipulation
    array_mask = np.array(mask)

    for i in range(1,len(CLASS_LIST)+ 1):
        # for each class, keep the class indice and change all other class indices to 0
        new_mask = np.where(array_mask==i, i, 0)
        list_of_mask.append(new_mask)
        
    #print(np.array(list_of_mask).shape)
    #print(list_of_mask)

    # convert numpy array to tensor again
    # to be used as a label where a one-hot encoding structure is used with class indices
    # shape --> [C,H,W] [7, 960, 1280]
    new_tensor_mask = torch.tensor(np.array(list_of_mask))
    #print(new_tensor_mask.shape)

   
    # output_mask --> contain all the individual binary (True/False) mask 
    # height and width is the same as the mask
    # channel is the number of classes
    # if 7 classes --> H,W,7
    output_mask = []

    for i, colour in enumerate(colormap):
        #print(rgb_mask)

        # for each individual colour, match it against the rgb_mask
        # colour_map --> binary mask
        # individual mask for individual classes
        colour_map = np.all(np.equal(rgb_mask, colour), axis = -1)

        # black images for classes that do not exist in the image
        #cv2.imwrite("/home/johnathon/Desktop/{}.png".format(i), colour_map*255)
        #print(colour_map)
        # colour_map * 255 --> black and white binary mask
        output_mask.append(colour_map)
        #break
    
    # one hot encoded mask with True and False --> can be converted to 0 and 1 when convert to .long()
    # structure of mask must be the same as the data when input into the model
    # output_mask shape --> [C,H,W] [7, 960, 1280]
    #print(np.array(output_mask).shape)

    # stack the output mask
    # one hot encoded mask with True and False --> can be converted to 0 and 1 when convert to .long()
    # structure of mask must be the same as the data when input into the model
    # shape --> [H,W,C] [960, 1280, 7]
    re_arranged_output_mask = np.stack(output_mask, axis=-1)
    #print(re_arranged_output_mask.shape)
   

    # single channel mask
    # shape --> [H,W] [960, 1280]
    # each pixel is the labelled
    # class indice starts from 0
    grayscale_mask = np.argmax(re_arranged_output_mask, axis=-1)
    #print(grayscale_mask)

    # convert each pixel class to a shade of grayscale
    #grayscale_mask = (grayscale_mask/len(CLASS_LIST))*255
    
    # expand dimension to show the image
    # shape --> [H,W,C], [960, 1280, 1]
    #grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)
    return np.array(new_tensor_mask), grayscale_mask

# create the dataset structure for dataloader
class ImageDataset(Dataset):
    def __init__(self, images, masks, transform=None):

        self.images = images
        self.masks= masks
        self.transform = transform

    
    # get the length of total dataset
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
            
        img = self.images[index]
        mask = self.masks[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img).convert("RGB")
        mask = Image.open(mask).convert("RGB")

        # convert mask to grayscale
        #mask = Image.open(mask).convert("L")
        
        if self.transform is not None:
            img = self.transform(img)
            # convert to tensor
            img = transforms.ToTensor()(img)

            # mask still in pil format
            mask = self.transform(mask)
            #print(np.array(mask))
            #print(np.array(mask).shape)

            # outout mask is in numpy array
            encoded_mask , _ = rgb_to_one_hot_encoded_mask(np.array(mask), colormap = PALETTE)

        if self.transform is None:
            # resize to 256
            #img = img.resize((256, 256), Image.Resampling.LANCZOS)
            # convert PIL to tensor
            img = transforms.ToTensor()(img)
            #mask = transforms.ToTensor()(mask)
            encoded_mask, _ = rgb_to_one_hot_encoded_mask(np.array(mask), colormap = PALETTE)
        return img, encoded_mask


#--------- Implement Data Transformations ----------------
class TrainDataTransforms(object):

    # do not resize the image as resizing the mask will result in changing the pixel colour of the mask, thus affecting class labels of each pixel
    def __init__(self,
                 # image size
                 #input_height: int = 256
                 ):

        # track the variables internally
        #self.input_height = input_height

        # can apply to the same image and will get a different result every single time
        # apply to only PIL images
        data_transforms = [
            #transforms.RandomResizedCrop(self.input_height),
            # p refers to the probability
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomGrayscale(p=0.2)
        ]

        # transforms.Compose just clubs all the transforms provided to it.
        # So, all the transforms in the transforms.Compose are applied to the input one by one.
        self.train_transform = transforms.Compose(data_transforms)

    #  The __call__ method enables Python programmers to write classes where the instances behave like functions and can be called like a function
    # sample refers to an input image
    def __call__(self, sample):

        # call the instance self.train_transform and make it behave like a function
        # use the train_transform as specified in the initialization
        transform = self.train_transform
        img = transform(sample)
        return img


train_transformed_dataset = ImageDataset(images = train_raw_image_path_list, masks=train_masked_image_path_list, transform=TrainDataTransforms())
train_dataloader = DataLoader(train_transformed_dataset, batch_size=1, shuffle=True, num_workers=1)

val_transformed_dataset = ImageDataset(images = val_raw_image_path_list, masks=val_masked_image_path_list, transform=None)
val_dataloader = DataLoader(val_transformed_dataset, batch_size=1, shuffle=True, num_workers=1)

test_transformed_dataset = ImageDataset(images = test_raw_image_path_list, masks=test_masked_image_path_list, transform=None)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=1, shuffle=True, num_workers=1)



checkpoints_folder = "/home/johnathon/Desktop/multi_segmentation/saved_model"

sqnet_model = SQNet(classes=len(CLASS_LIST))
#sqnet_model = nn.DataParallel(sqnet_model, device_ids=list(range(torch.cuda.device_count())))
sqnet_model.eval()

# get the model weights
state_dict = torch.load(os.path.join(checkpoints_folder, "epoch_5_model.pt"), map_location=torch.device('cpu'))

# load the model weights
sqnet_model.load_state_dict(state_dict)

model = sqnet_model.to(device)

"""
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
"""

# freeze all the layers in the model
for param in model.parameters():
    param.requires_grad = False

for batch_idx, (data, targets) in enumerate(train_dataloader):
        #print(batch_idx)
        
        data = data.to(device)
        #print(data.shape)
        #print(targets)

        targets = targets.long().to(device)
        #print(targets)
        #print(targets.shape)
        
        # get predictions --> logits
        # shape --> [batch_size,C,H,W] [1, 7, 960, 1280]
        predictions = sqnet_model(data)
        #print(predictions)
        #print(predictions.shape)
        
        # shape --> [C,H,W] [1, 960, 1280]
        # each pixel represents a class
        new_predictions = torch.argmax(predictions, dim=1)
        #print(new_predictions.shape)
        
        
        ground_truth = torch.argmax(targets, dim=1)
        #print(ground_truth)

       
        # shape --> [H,W,C] [960, 1280, 1]
        # to stack to get 3 channels
        new_predictions = torch.reshape(new_predictions, (new_predictions.shape[1],new_predictions.shape[2], new_predictions.shape[0]))
        #print(new_predictions)
        #print(new_predictions.shape)

        # get 3 channels so that can map each pixel class to the colour
        three_channels_prediction = torch.cat([new_predictions, new_predictions, new_predictions], dim=2)
        #print(three_channels_prediction)
        #print(three_channels_prediction.shape)

        # map each colour to a class indice 
        # class indice starts from 1
        # to create a one-hot encoding structure with class indices
        mapping = {tuple(c): t for c, t in zip(PALETTE, range(len(PALETTE)))}

        # convert the tensors to numpy array
        mask_array = np.array(three_channels_prediction.cpu())

        # create a an empty mask
        new_mask = np.zeros(three_channels_prediction.shape)

        # map each class indices to each colour
        for index, color in enumerate(mapping.keys()):
            #print(index, color)
            if index < len(mapping.keys()):
                # bring the array to the cpu first
                new_mask[np.all(np.array(new_predictions.cpu())== index, axis=-1)] = color
        
        #print(new_mask)
        #print(new_mask.shape)
        
        # check the output of the mask
        #cv2.imwrite("/home/johnathon/Desktop/test.jpg", new_mask)

        # flatten the ground truth and prediction to check the accuracy
        y_ground_truth = np.array(torch.flatten(ground_truth).cpu())
        y_predict = np.array(torch.flatten(new_predictions).cpu())
        acc = accuracy_score(y_ground_truth, y_predict)
        print("Accuracy is {}%".format(acc * 100))
        




      
        break