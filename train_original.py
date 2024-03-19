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


##################################### Create dataset class ###############################

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

"""
# Testing the function
img_pil = Image.open("/home/johnathon/Desktop/test_dir/masked_images/0_mask.png").convert("RGB")
pimg = np.array(img_pil)
#print(pimg)
#print(pimg.shape)
output_mask , grayscale_mask = rgb_to_one_hot_encoded_mask(pimg, colormap = PALETTE)
#print(output_mask.shape)
#print(grayscale_mask)
#print(grayscale_mask.shape)
#cv2.imshow("test",encoded_image)
#cv2.waitKey(0)
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

"""
# only use for resizing or transforming the data to tensors when evaluation
# -------- evaluation of the the data transformation ---------
class EvalDataTransform(object):
    # track these parameters internally
    def __init__(self, input_height: int = 256):

        self.input_height = input_height

        # convert to tensor
        eval_data_transforms = [transforms.Resize([self.input_height])]

        self.eval_transforms = transforms.Compose(eval_data_transforms)

    # take the same input image used in the training, "sample"
    def __call__(self, sample):

        # call the instance self.test_transforms and make it behave like a function
        transform = self.eval_transforms

        img = transform(sample)  # first version

        return img
"""

train_transformed_dataset = ImageDataset(images = train_raw_image_path_list, masks=train_masked_image_path_list, transform=TrainDataTransforms())
train_dataloader = DataLoader(train_transformed_dataset, batch_size=2, shuffle=True, num_workers=1)

val_transformed_dataset = ImageDataset(images = val_raw_image_path_list, masks=val_masked_image_path_list, transform=None)
val_dataloader = DataLoader(val_transformed_dataset, batch_size=2, shuffle=True, num_workers=1)

test_transformed_dataset = ImageDataset(images = test_raw_image_path_list, masks=test_masked_image_path_list, transform=None)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=2, shuffle=True, num_workers=1)


# SQNet model
sqnet_model = SQNet(classes=len(CLASS_LIST)).to(device)
#sqnet_model = nn.DataParallel(sqnet_model, device_ids=list(range(torch.cuda.device_count())))
#LDAMLoss_loss_fn = LDAMLoss(cls_num_list=CLASS_LIST_LABEL).to(device) # can use cross entropy loss
#loss_fn = nn.CrossEntropyLoss().to(device)
loss_fn = DiceLoss().to(device)
optimizer = RAdam(sqnet_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
#optimizer = torch.optim.SGD(sqnet_model.parameters(), lr=0.01, weight_decay=1e-6)


def get_mean_of_list(L):
    return sum(L) / len(L)

# validation during model training
def _validate(model, dataloader):
    # validation steps
    with torch.no_grad():
        model.eval()

        valid_loss = 0.0
        counter = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            targets = targets.long().to(device)
            predictions = sqnet_model(data)

            loss = loss_fn(predictions, targets)
            valid_loss += loss.item()
            counter += 1
        valid_loss = valid_loss / counter
    model.train()
    return valid_loss

# deletes file if the file exist
# text file is used to log down epoch loss
file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'mean_batch_loss.txt')
if os.path.isfile(file):
    os.remove(file)

best_mean_batch_loss = np.inf

# create a tensorboard
writer = SummaryWriter()

#"""
for epoch_counter in tqdm(range(100), desc='Epoch progress'):

    epoch_losses_train = []
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        data = data.to(device)
        #print(data.shape)
        #print(targets)
        targets = targets.long().to(device)
        #targets = targets.float().unsqueeze(1).to(device)
        #print(targets)
        #print(targets.shape)
    
       
        predictions = sqnet_model(data)
        #print(predictions.shape)
        #predictions = torch.argmax(predictions, dim=1)
        #predictions = predictions.type(torch.cuda.FloatTensor)
        #targets = targets.type(torch.cuda.FloatTensor)
        #print(predictions, targets)
        #print(predictions.shape)
    
        #target = torch.argmax(targets.long(), dim=1)
        
        loss = loss_fn(predictions, targets)
        epoch_losses_train.append(loss.to(device).data.item())
        #print(loss)
        loss.backward()
        optimizer.step()
        #print(loss)
        #loss = LDAMLoss_loss_fn(predictions, targets)
        #print(loss)
    #print(epoch_losses_train)
    valid_loss = _validate(sqnet_model, val_dataloader)
    mean_batch_loss_training = get_mean_of_list(epoch_losses_train)

    # add to tensorboard
    writer.add_scalar("Loss/train", mean_batch_loss_training, epoch_counter)

    print("Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter, mean_batch_loss_training, valid_loss))
    model_path = os.path.join("/home/johnathon/Desktop/multi_segmentation/saved_model", 'epoch_{}_model.pt'.format(epoch_counter))
    torch.save(sqnet_model.state_dict(), model_path)
    file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'mean_batch_loss.txt')
    
    with open(file, 'a') as filetowrite:
        filetowrite.write(
            "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                            mean_batch_loss_training,
                                                                                            valid_loss))
        filetowrite.write("\n")

    if mean_batch_loss_training < best_mean_batch_loss:
        # save the model weights
        best_mean_batch_loss = mean_batch_loss_training
        if len(os.listdir("/home/johnathon/Desktop/multi_segmentation/best_model_weights/")) >=1:
            for files in os.listdir("/home/johnathon/Desktop/multi_segmentation/best_model_weights/"):
                file_path = os.path.join("/home/johnathon/Desktop/multi_segmentation/best_model_weights/", files)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        torch.save(sqnet_model.state_dict(), "/home/johnathon/Desktop/multi_segmentation/best_model_weights/epoch_{}_model.pt".format(epoch_counter))
        file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'best_mean_batch_loss.txt')
        with open(file, 'w') as filetowrite:
            filetowrite.write(
                "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                            best_mean_batch_loss,
                                                                                            valid_loss))

    #print(mean_batch_loss_training)
        #break
    #break

writer.flush()
writer.close()

#"""


"""
############################################# Extract Target Class definitions ############################
# sky, land, sea, ship, buoy, other
CLASS_LIST = ['sky', 'land', 'sea', 'ship', 'buoy', 'other']
PALETTE = [(128, 0, 0), (0, 128, 0), (128, 128, 0), 
           (0, 0, 128), (128, 0, 128), (0, 128, 128)]


# label colour code
code2id = {v:k for k,v in enumerate(PALETTE)}
id2code = {k:v for k,v in enumerate(PALETTE)}

#print(code2id)

# label name code
name2id = {v:k for k,v in enumerate(CLASS_LIST)}
id2name = {k:v for k,v in enumerate(CLASS_LIST)}

#print(name2id)
"""

"""
############## Define functions for one hot encoding rgb labels, and decoding encoded predictions ##############

def rgb_to_onehot(rgb_image, colormap = id2code):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    # image plus class
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image
"""

"""
def onehot_to_rgb(onehot, colormap = id2code):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in colormap.keys():
        output[single_layer==k] = colormap[k]
    return np.uint8(output)
"""

