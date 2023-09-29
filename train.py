import numpy as np
import matplotlib.pyplot as plt
import os
import random
import re
from PIL import Image
import cv2
from typing import Optional, Callable, Tuple, Any
from tqdm import tqdm
import glob

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
test_raw_image_path_list = raw_image_path_list[40:]
#print(train_raw_image_list)
#print(val_raw_image_list)
#print(test_raw_image_list)

# masked images path
train_masked_image_path_list = masked_images_path_list[:30]
val_masked_image_path_list = masked_images_path_list[30:40]
test_masked_image_path_list = masked_images_path_list[40:]

# sky, land, sea, ship, buoy, other
CLASS_LIST = ['background','sky', 'land', 'sea', 'ship', 'buoy', 'other']
PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], 
           [0, 0, 128], [128, 0, 128], [0, 128, 128]]


##################################### Create dataset class ###############################

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
# Testing the function
img_pil = Image.open("/home/johnathon/Desktop/test_dir/masked_images/0_mask.png").convert("RGB")
pimg = np.array(img_pil)
#print(pimg)
#print(pimg.shape)
#class_indices_mask_list , grayscale_mask = rgb_to_one_hot_encoded_mask(pimg, colormap = PALETTE)
#print(class_indices_mask_list.shape)
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
        #mask = transforms.ToPILImage()(mask)
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
            img = transforms.ToTensor()(img)
            #mask = transforms.ToTensor()(mask)
            encoded_mask, _ = rgb_to_one_hot_encoded_mask(np.array(mask))
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
train_dataloader = DataLoader(train_transformed_dataset, batch_size=2, shuffle=True, num_workers=3)

val_transformed_dataset = ImageDataset(images = val_raw_image_path_list, masks=val_masked_image_path_list, transform=None)
val_dataloader = DataLoader(val_transformed_dataset, batch_size=2, shuffle=True, num_workers=3)

test_transformed_dataset = ImageDataset(images = test_raw_image_path_list, masks=test_masked_image_path_list, transform=None)
test_dataloader = DataLoader(test_transformed_dataset, batch_size=2, shuffle=True, num_workers=3)


# SQNet model
sqnet_model = SQNet(classes=len(CLASS_LIST)).to(device)
#sqnet_model = nn.DataParallel(sqnet_model, device_ids=list(range(torch.cuda.device_count())))
#LDAMLoss_loss_fn = LDAMLoss(cls_num_list=CLASS_LIST_LABEL).to(device) # can use cross entropy loss
#loss_fn = nn.CrossEntropyLoss().to(device)
loss_fn = DiceLoss().to(device)
optimizer = RAdam(sqnet_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
#optimizer = torch.optim.SGD(sqnet_model.parameters(), lr=0.01, weight_decay=1e-6)


serial_module = torch.jit.script(sqnet_model)

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
file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'mean_batch_training_loss.txt')
if os.path.isfile(file):
    os.remove(file)

lowest_mean_batch_loss = np.inf
lowest_valid_loss = np.inf

# create a tensorboard
writer = SummaryWriter()


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

        #traced_script_module_predictions = torch.jit.trace(sqnet_model, data)
        
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
    serial_module_model_path = "/home/johnathon/Desktop/multi_segmentation/saved_model/epoch_{}_serial_module_model.pt".format(epoch_counter)
    torch.save(sqnet_model.state_dict(), model_path)
    serial_module.save(serial_module_model_path)

    file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'mean_batch_training_loss.txt')
    
    with open(file, 'a') as filetowrite:
        filetowrite.write(
            "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                            mean_batch_loss_training,
                                                                                            valid_loss))
        filetowrite.write("\n")

    if mean_batch_loss_training < lowest_mean_batch_loss:
        # save the model weights
        lowest_mean_batch_loss = mean_batch_loss_training
        
        if len(os.listdir("/home/johnathon/Desktop/multi_segmentation/best_model_weights/")) >=1:
            for files in os.listdir("/home/johnathon/Desktop/multi_segmentation/best_model_weights/"):
                file_path = os.path.join("/home/johnathon/Desktop/multi_segmentation/best_model_weights/", files)
                if os.path.isfile(file_path) and "lowest_mean_batch_training_loss" in file_path:
                    os.remove(file_path)
        
        torch.save(sqnet_model.state_dict(), "/home/johnathon/Desktop/multi_segmentation/best_model_weights/epoch_{}_lowest_mean_batch_training_loss_model.pt".format(epoch_counter))
        
        serial_module_model_path = "/home/johnathon/Desktop/multi_segmentation/best_model_weights/epoch_{}_serial_module_lowest_mean_batch_training_loss_model.pt".format(epoch_counter)
        serial_module.save(serial_module_model_path)
        
        file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'lowest_mean_batch_loss.txt')
        with open(file, 'w') as filetowrite:
            filetowrite.write(
                "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                            lowest_mean_batch_loss,
                                                                                            valid_loss))


    if valid_loss < lowest_valid_loss:

        lowest_valid_loss = valid_loss

        if len(os.listdir("/home/johnathon/Desktop/multi_segmentation/best_model_weights/")) >=1:
            for files in os.listdir("/home/johnathon/Desktop/multi_segmentation/best_model_weights/"):
                file_path = os.path.join("/home/johnathon/Desktop/multi_segmentation/best_model_weights/", files)
                if os.path.isfile(file_path) and "lowest_validation_loss" in file_path:
                    os.remove(file_path)
        
        torch.save(sqnet_model.state_dict(), "/home/johnathon/Desktop/multi_segmentation/best_model_weights/epoch_{}_lowest_validation_loss_model.pt".format(epoch_counter))
        
        serial_module_model_path = "/home/johnathon/Desktop/multi_segmentation/best_model_weights/epoch_{}_serial_module_lowest_validation_loss_model.pt".format(epoch_counter)
        serial_module.save(serial_module_model_path)

        file = os.path.join("/home/johnathon/Desktop/multi_segmentation/", 'lowest_validation_loss.txt')
        with open(file, 'w') as filetowrite:
            filetowrite.write(
                "Epoch:{} ------ Mean Batch Loss ({}) ------ Validation_loss: ({})".format(epoch_counter,
                                                                                            lowest_valid_loss,
                                                                                            valid_loss))

    #print(mean_batch_loss_training)
        #break
    #break

writer.flush()
writer.close()




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

