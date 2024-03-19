import torch
import numpy as np
import os
from SQNet import SQNet

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

root_folder_path = "/home/johnathon/Desktop/multi_segmentation/best_model_weights"
pytorch_model_path = os.path.join(root_folder_path, "epoch_22_model.pt")

sqnet_model = SQNet(classes=7)

state_dict = torch.load(pytorch_model_path, map_location=torch.device('cpu'))

sqnet_model.load_state_dict(state_dict)

#model = sqnet_model.to(device)

# freeze all the layers in the model
for param in sqnet_model.parameters():
    param.requires_grad = False

quantized_model = torch.quantization.quantize_dynamic(sqnet_model, dtype=torch.qint8)

quantized_model.eval()
sqnet_model.eval()

