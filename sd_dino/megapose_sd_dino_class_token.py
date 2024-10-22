import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torchvision import transforms
from sd_dino.extractor_sd import load_model, process_features_and_mask, get_mask
from sd_dino.utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA as sklearnPCA
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import cv2
import time
from qrpca.decomposition import svdpca, qrpca  
from gpu_pca import IncrementalPCAonGPU


MASK = True
VER = "v1-5"
PCA = False
CO_PCA = True
PCA_DIMS = [256, 256, 256]
SIZE =960
EDGE_PAD = False
ANNO = False

DINOV2 = True
MODEL_SIZE = 'base'
DRAW_DENSE = 1
DRAW_SWAP = 1
TEXT_INPUT = False
SEED = 42
TIMESTEP = 100

img_size = 840 if DINOV2 else 244
normalization_mean = (0.485, 0.456, 0.406) if DINOV2 else (0.5, 0.5, 0.5)
normalization_std = (0.229, 0.224, 0.225) if DINOV2 else (0.5, 0.5, 0.5)
n_components=4 # the first component is to seperate the object from the background


DIST = 'l2'

# def transform_to_img(img_torch):
#     img = img_torch.permute(0, 3, 1, 2)/255
#     aspect_ratio = img.shape[-2]/img.shape[-1]
#     new_height = int(img_size*aspect_ratio)
#     padding_size = img_size-new_height
#     transform = transforms.Compose([
#         transforms.Resize(new_height),
#         transforms.Pad((0, padding_size//2, 0, padding_size//2), fill=0, padding_mode='constant')
#     ])
#     prep_img = transform(img)
#     return prep_img

def compute_cls_tokens(model, aug, extractor, views_batch, target_shape, only_dino):
    FUSE_DINO = True
    img_size = 840 if DINOV2 else 244
    model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
    
    model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pca = sklearnPCA(n_components=n_components)
    # pca = svdpca(n_component_ratio=n_components,device=device)
    # pca = qrpca(n_component_ratio=n_components,device=device)
    # pca = IncrementalPCAonGPU(n_components=n_components)
    batch_list = []
    for idx in range(len(views_batch)):
        input_imgs = views_batch[idx].to(device)
        normalize = transforms.Normalize(mean=normalization_mean, std=normalization_std)
        normalized_imgs = normalize(input_imgs)

        
        with torch.no_grad():
            if FUSE_DINO:
                img_desc_dino = extractor.extract_descriptors(normalized_imgs, layer, facet, include_cls=True)
                img_desc_dino = img_desc_dino[:, :, 1:, :]
                cls_token = img_desc_dino[:, :, 0, :]

            if DIST == 'l1' or DIST == 'l2':
                if FUSE_DINO:
                    img_desc_dino = img_desc_dino / img_desc_dino.norm(dim=-1, keepdim=True)
                    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

            batch_list.append((cls_token.squeeze().cpu().numpy()))
            
    return batch_list