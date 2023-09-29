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
import glob

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

################################### Split into train, val, test ###################################

"""
In this solution we created a list of files in a folder using globe() function. 
Then passed the list to filter() function to select only files from the list and skip dictionaries etc. 
For this we passed the os.path.isfile() function as an argument to the filter() function. 
Then we passed the list of files to the sorted() function, which returned a list of files in directory sorted by name. 
"""
raw_image_path_list = sorted(filter(os.path.isfile, glob.glob(raw_image_folder_path + '/*')))
masked_images_path_list = sorted(filter(os.path.isfile, glob.glob(masked_images_folder_path + '/*')))

#print(raw_image_path_list)
#print(masked_images_path_list)

# raw images path
train_raw_image_path_list = raw_image_path_list[:30]
val_raw_image_path_list = raw_image_path_list[30:40]
# test_raw_image_path_list = raw_image_path_list[40:]
test_raw_image_path_list = raw_image_path_list[49:]
print(test_raw_image_path_list)
#print(train_raw_image_list)
#print(val_raw_image_list)
#print(test_raw_image_list)

# masked images path
train_masked_image_path_list = masked_images_path_list[:30]
val_masked_image_path_list = masked_images_path_list[30:40]
# test_masked_image_path_list = masked_images_path_list[40:]
test_masked_image_path_list = masked_images_path_list[49:]
print(test_masked_image_path_list)



# sky, land, sea, ship, buoy, other
CLASS_LIST = ['background', 'sky', 'land', 'sea', 'ship', 'buoy', 'other']
PALETTE = [[0, 0, 0],[128, 0, 0], [0, 128, 0], [128, 128, 0], 
           [0, 0, 128], [128, 0, 128], [0, 128, 128]]


def rgb_to_one_hot_encoded_mask(rgb_mask):

    # shape --> [H,W,C] [960, 1280, 3]
    target = torch.from_numpy(rgb_mask)

    # reshape mask 
    # shape --> [C,H,W] [3,960, 1280]
    target = target.permute(2, 0, 1).contiguous()

    # map each colour to a class indice 
    # class indice starts from 1
    # to create a one-hot encoding structure with class indices
    mapping = {tuple(c): t for c, t in zip(PALETTE, range(0,len(PALETTE)))}

    # empty tensor to concatenate tensor masks of objects with class indices encoding
    # create an empty tensor of zeros of size (1,H,W) --> 1 is class indice 0, which is the background class --> (0, 0,0,0)
    class_indices_mask_list = torch.zeros((1, rgb_mask.shape[0], rgb_mask.shape[1]))

    # empty tensor to concatenate tensor masks of objects with one hot encoding
    # create an empty tensor of zeros of size (1,H,W) --> 1 is class indice 0, which is the background class --> (0, 0,0,0)
    one_hot_encoding_mask_list = torch.zeros((1, rgb_mask.shape[0], rgb_mask.shape[1]))
    
    # each pixel position in the mask represents the class indices
    # k --> pixel colour
    # mapping[k] --> class indices
    for k in mapping:

        # if the class indice == 0, pass
        # skip the background class 
        if mapping[k] == 0:
            
            pass

        else:

            # Get all indices for current class
            # e.g. class label 3
            # tensor_array --> [3]
            # tensor_array.unsqueeze(1) --> [3,1]
            # tensor_array.unsqueeze(1).unsqueeze(2) --> [3,1,1]
            idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))

            # Check that all channels match
            validx = (idx.sum(0) == 3)

            # where the value is 1 (true), replace it with class indice and the rest is kept as 0
            class_indice_mask = torch.where(validx == 1, mapping[k], 0)
            #mask[validx] = torch.tensor(mapping[k], dtype=torch.long)

            # convert the validx to torch integer and then concatenate to the empty tensor
            # .unsqueeze(0) --> a increase the dimension of the tensor by 1 at position 0
            # 0 --> concatenate at position 1
            # concatenate to the tensors of zero, which is the background class
            class_indices_mask_list = torch.cat((class_indices_mask_list, class_indice_mask.unsqueeze(0)),0)

            one_hot_encoding_mask_list = torch.cat((one_hot_encoding_mask_list, validx.unsqueeze(0)),0)

    # print(class_indices_mask_list)
    # print(class_indices_mask_list.shape)
    # print(one_hot_encoding_mask_list)
    # print(one_hot_encoding_mask_list.shape)

    # shape --> [H,W,C] [960, 1280, 7]
    re_arranged_class_indices_mask = class_indices_mask_list.permute(1, 2, 0).contiguous()
    # print(re_arranged_one_hot_encoding_mask)
    # print(re_arranged_one_hot_encoding_mask.shape)

    # single channel mask
    # shape --> [H,W] [960, 1280]
    # each pixel is the labelled class
    # class indice starts from 0
    # argmax returns the indices of the maximum value of all elements in the input tensor
    # If there are multiple maximal values then the indices of the first maximal value are returned.
    grayscale_mask = torch.argmax(re_arranged_class_indices_mask, axis=-1)
    #print(grayscale_mask)
    
    # convert each pixel class to a shade of grayscale
    #grayscale_mask = (grayscale_mask/len(CLASS_LIST))*255
    
    # expand dimension to show the image
    # shape --> [H,W,C], [960, 1280, 1]
    #grayscale_mask = grayscale_mask.unsqueeze(-1)
    #print(grayscale_mask.shape)

    
    return class_indices_mask_list, grayscale_mask


""" 
img_pil = Image.open("/home/johnathon/Desktop/test_dir/masked_images/0_mask.png").convert("RGB")
pimg = np.array(img_pil)
#print(pimg)
print(pimg.shape)
output_mask , grayscale_mask = rgb_to_one_hot_encoded_mask(pimg, colormap = PALETTE)
"""


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
            encoded_mask , _ = rgb_to_one_hot_encoded_mask(np.array(mask))

        if self.transform is None:
            # resize to 256
            #img = img.resize((256, 256), Image.Resampling.LANCZOS)
            # convert PIL to tensor
            print(np.array(img))
            print(np.array(img).shape)
            #print(np.array(img).shape)
            img = transforms.ToTensor()(img)
            print(img)
            print(img.shape)
            #mask = transforms.ToTensor()(mask)
            encoded_mask, _ = rgb_to_one_hot_encoded_mask(np.array(mask))
        return img, encoded_mask


test_transformed_dataset = ImageDataset(images = test_raw_image_path_list, masks=test_masked_image_path_list, transform=None)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=1, shuffle=True, num_workers=1)

checkpoints_folder = "/home/johnathon/Desktop/multi_segmentation/saved_model"

sqnet_model = SQNet(classes=len(CLASS_LIST))
#sqnet_model = nn.DataParallel(sqnet_model, device_ids=list(range(torch.cuda.device_count())))
sqnet_model.eval()

# get the model weights
state_dict = torch.load(os.path.join(checkpoints_folder, "epoch_22_model.pt"), map_location=torch.device('cpu'))

# load the model weights
sqnet_model.load_state_dict(state_dict)

quantized_model = torch.quantization.quantize_dynamic(sqnet_model)



""" # Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
"""

# freeze all the layers in the model
for param in sqnet_model.parameters():
    param.requires_grad = False


for batch_idx, (data, targets) in enumerate(test_dataloader):
        print(batch_idx)
        
        # shape --> [batch_size, C, H, W] [1, 3, 960, 1280]
        # data = data.to(device)
        #print(data)
        #print(data.shape)
        #print(targets)

        # targets = targets.long().to(device)
        targets = targets.long()
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
        print(new_predictions)
        print(new_predictions.size())
        
        
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
            print(index, color)
            #if index < len(mapping.keys()):
                #print(index)
                # bring the array to the cpu first
            new_mask[np.all(np.array(new_predictions.cpu())== index, axis=-1)] = color
        
        #print(new_mask)
        #print(new_mask.shape)
        
        # check the output of the mask
        cv2.imwrite("/home/johnathon/Desktop/test2.jpg", new_mask)
        
        

        # flatten the ground truth and prediction to check the accuracy
        y_ground_truth = np.array(torch.flatten(ground_truth).cpu())
        y_predict = np.array(torch.flatten(new_predictions).cpu())
        acc = accuracy_score(y_ground_truth, y_predict)
        print("Accuracy is {}%".format(acc * 100))
        
        #break
         

 