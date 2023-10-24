import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import cv2
from PIL import Image
from torchvision import transforms

#from vision_transformer_pytorch import VisionTransformer
from CLIP.clip import clip
from CLIP.clip import model

#from torchvision.datasets import CIFAR100
from tqdm import tqdm
from sklearn import preprocessing
import itertools

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Load the model
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load('ViT-L/14', device, jit=False)

def get_attention_map(img, weights, get_mask=False):
    
    att_mat = weights
    #print("original weights")
    #print(att_mat)
    #print(att_mat.shape)

    att_mat = torch.stack(att_mat).squeeze(1)
    #print("stack and squeeze on the second dimension")
    #print(att_mat)
    #print(att_mat.shape)

    # Average the attention weights across all heads.
    # att_mat = torch.mean(att_mat, dim=1)
    # print("mean")
    # print(att_mat)
    # print(att_mat.shape)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    #print("identity and re-normalize")
    #print(aug_att_mat)
    #print(aug_att_mat.shape)

    # # Recursively multiply the weight matrices
    # joint_attentions = torch.zeros(aug_att_mat.size())  
    # joint_attentions[0] = aug_att_mat[0]

    # for n in range(1, aug_att_mat.size(0)):
    #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = aug_att_mat[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))

    #print(v)
    #print(v.shape)
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask_only = cv2.resize(mask / mask.max(), img.size) # mask only

    #print(mask)
    #print(mask.shape)

    # if get_mask:
    #     result = cv2.resize(mask / mask.max(), img.size)
    # else:        
    #     mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    #     result = (mask * img).astype("uint8")
    #print(result)
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    blended = (mask * img).astype("uint8")

    return mask_only, blended

def plot_attention_map(original_img, mask_only, blended, save_dir, filename, predicted_label, diversity, lu_labels):
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16,4))
    fig.suptitle('City: ' + lu_labels + '   DI value: ' + diversity + '   Prediction: ' + predicted_label)

    ax1.set_title('Original Image')
    ax2.set_title('Attention Map (Last Layer)')
    ax3.set_title('Attention Mask')
    _ = ax1.imshow(original_img, extent=[0, 400, 0, 300])
    _ = ax2.imshow(blended, extent=[0, 400, 0, 300])
    _ = ax3.imshow(mask_only, extent=[0, 400, 0, 300])

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    plt.savefig(save_dir + filename, dpi=200)


save_dir = '/home/ai4sg/meiliu/CLIP/CLIP/landuse/heatmaps/'
src_dir = "/home/ai4sg/meiliu/CLIP/CLIP/landuse/NewYork_target/"

#res_file = open("prompt_ensemble_acc_allpolygons.txt",'r')  #prompt_ensemble_acc_allpolygons_test
res_file = open("prompt_ensemble_acc_allpolygons_test.txt",'r')
res_file_lines = res_file.readlines()

for line in res_file_lines:
    #line example: 
    # format: img, predicted_label, diversity, num_lu_detected, bool_correct, multi_lu_list
    # img format: 40.903867_-73.865892_513d7967fdc9f035870064b4_NewYork.JPG

    line = line.rstrip('\n')
    tokens = line.split(',')
    filename = tokens[0]
    predicted_label = tokens[1]
    diversity = tokens[2]
    lu_labels = tokens[-1]

    print(filename)

    img = Image.open(src_dir + filename)
    img_preproc = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        image_features, weights = model.encode_image(img_preproc)

    # result1 = get_attention_map(img, weights)
    mask_only, blended = get_attention_map(img, weights)
    plot_attention_map(img, mask_only, blended, save_dir, filename, predicted_label, diversity, lu_labels)


# result1 = get_attention_map(img1, True)
# plot_attention_map(img1, result1)