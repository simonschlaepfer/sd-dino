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
import pickle


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
# n_components=4 # the first component is to seperate the object from the background


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

def compute_features(model, aug, extractor, views_batch, target_shape, only_dino, use_prefit_pca=False, n_components=4, add_cls_token=False):
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

    pca_load_time = time.time()
    if use_prefit_pca:
        with open('/cluster/home/simschla/master_thesis/NOM-Diffusion/pca/pca_dino_model_501_7.pkl', 'rb') as file:
            pca = pickle.load(file)
    else:
        pca = sklearnPCA(n_components=n_components)
    # print("pca load time", time.time()-pca_load_time)

    if add_cls_token:
        with open('/cluster/home/simschla/master_thesis/NOM-Diffusion/pca/pca_class_token_model_501_6.pkl', 'rb') as file:
            pca_cls = pickle.load(file)

    # pca = svdpca(n_component_ratio=n_components,device=device)
    # pca = qrpca(n_component_ratio=n_components,device=device)
    # pca = IncrementalPCAonGPU(n_components=n_components)
    feature_batch_list = []
    cls_token_batch_list = []
    for idx in range(len(views_batch)):
        batch_start_time = time.time()
        input_imgs = views_batch[idx].to(device)
        normalize = transforms.Normalize(mean=normalization_mean, std=normalization_std)
        normalized_imgs = normalize(input_imgs)

        
        with torch.no_grad():
            if FUSE_DINO:
                start_extract_time = time.time()
                img_desc_dino = extractor.extract_descriptors(normalized_imgs, layer, facet, include_cls=True)
                img_desc_dino = img_desc_dino[:, :, 1:, :]
                cls_token = img_desc_dino[:, :, 0, :]
                end_extract_time = time.time()
                # print("Total feature extraction:", end_extract_time-start_extract_time)

            start_postprocess_time = time.time()
            if DIST == 'l1' or DIST == 'l2':
                if FUSE_DINO:
                    img_desc_dino = img_desc_dino / img_desc_dino.norm(dim=-1, keepdim=True)
                    cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

            if add_cls_token:
                cls_token_reduced = pca_cls.transform(cls_token.squeeze().cpu().numpy())[:, :n_components-1]
                cls_token_reduced =  np.expand_dims(cls_token_reduced, axis=1)
            else:
                cls_token_reduced = None
            
            mask = (input_imgs.mean(1)>0).to(torch.float32).to(device)
            mask_pca_start_time = time.time()
            feature_imgs, cls_token_img = vis_pca_mask_all(img_desc_dino, mask, target_shape, use_prefit_pca, pca, n_components, cls_token_reduced)
            mask_pca_end_time = time.time()
            # print("mask pca time:", mask_pca_end_time-mask_pca_start_time)
            feature_img_list = [feature_imgs[i] for i in range(feature_imgs.shape[0])]
            feature_batch_list.append(feature_img_list)
            if add_cls_token:
                cls_token_batch_list.append(cls_token_img)
        batch_end_time = time.time()
        # print("feature creation per sample:", batch_end_time-batch_start_time)
    return feature_batch_list, cls_token_batch_list


def vis_pca_mask_all(feature, mask, target_shape, use_prefit_pca, pca, n_components, cls_token_reduced):
    num_patches = int(math.sqrt(feature.shape[-2]))
    channel_dim = feature.shape[-1]
    feature = feature.squeeze(1)
    feature_reshaped = feature.permute(0,2,1).reshape(feature.shape[0], -1, num_patches, num_patches)
    reshape_start_time = time.time()
    resized_mask = F.interpolate(mask.unsqueeze(1), size=(num_patches, num_patches), mode='nearest')
    reshape_end_time = time.time()
    # print("reshape time", reshape_end_time-reshape_start_time)
    feature_upsampled = feature_reshaped * resized_mask.repeat(1,feature_reshaped.shape[1],1,1)
    feature_processed=feature_upsampled.reshape(feature_upsampled.shape[0], channel_dim,-1).permute(0,2,1)
    feature_processed_stacked=feature_processed.reshape(-1,feature_processed.shape[-1])
    
    start_time = time.time()

    if not use_prefit_pca:
        feature1_n_featuren = pca.fit_transform(feature_processed_stacked.cpu().numpy())[:, 1:n_components] # shape (7200,3)
    else:
        feature1_n_featuren = pca.transform(feature_processed_stacked.cpu().numpy())[:, 1:n_components]
    # feature1_n_featuren = pca.fit_transform(feature_processed_stacked)
    # pca.fit(feature_processed_stacked)
    # feature1_n_featuren = pca.transform(feature_processed_stacked)
    # feature1_n_featuren = feature1_n_featuren.cpu().numpy()
    end_time = time.time()
    # print("Dimension reduction PCA:", end_time-start_time)
    features_reduced=feature1_n_featuren.reshape(feature_processed.shape[0],feature_processed.shape[1],n_components-1)

    min_vals = features_reduced.min(axis=1)
    max_vals = features_reduced.max(axis=1)
    diff = max_vals - min_vals
    diff[diff == 0] = 1 # Avoid division by zero in case of constant columns
    features_reduced = (features_reduced - min_vals[:, np.newaxis, :]) / diff[:, np.newaxis, :]
    if cls_token_reduced is not None:
        cls_token_reduced = (cls_token_reduced - min_vals[:, np.newaxis, :]) / diff[:, np.newaxis, :]
        cls_token_clipped = np.clip(cls_token_reduced, 0, 1)
        cls_token_clipped = np.median(cls_token_clipped, axis=0)
        # cls_token_clipped = np.tile(cls_token_clipped, (9, 1, 1))
        cls_token_clipped = cls_token_clipped[:, np.newaxis, :]
        cls_token_img = (cls_token_clipped*255).astype(np.uint8)
        cls_token_img = np.repeat(cls_token_img, target_shape[1], axis=1)
        cls_token_img = np.repeat(cls_token_img[:, :, np.newaxis, :], target_shape[0], axis=2)

    feature_resized = features_reduced.reshape(features_reduced.shape[0],num_patches,num_patches, n_components-1)*resized_mask.squeeze(1).unsqueeze(-1).repeat(1,1,1,n_components-1).cpu().numpy()
    # feature_img_square = (feature_img_square*255).to(torch.uint8)
    # feature_img = pad_image(feature_img_square, target_shape)

    feature_img = (feature_resized*255).astype(np.uint8)
    feature_imgs_list = []
    pad_start_time = time.time()
    for img in feature_img:
        feature_img_padded = pad_image(img, target_shape)
        feature_imgs_list.append(feature_img_padded)
    pad_end_time = time.time()
    # print("pad time", pad_end_time-pad_start_time)

    # features_reduced=feature1_n_featuren.reshape(feature_processed.shape[0],feature_processed.shape[1],n_components)

    # feature_after_pca_list = [features_reduced[0], features_reduced[1], features_reduced[2], features_reduced[3], features_reduced[4], features_reduced[5], features_reduced[6], features_reduced[7], features_reduced[8]]

    # feature_imgs_list = []
    # for feature_after_pca, mask in zip(feature_after_pca_list, mask):
    #     resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest') # 60 x 60
    #     for show_channel in range(n_components):
    #         if show_channel==0:
    #             continue
    #         # min max normalize the feature map
    #         feature_after_pca[:, show_channel] = (feature_after_pca[:, show_channel] - feature_after_pca[:, show_channel].min()) / (feature_after_pca[:, show_channel].max() - feature_after_pca[:, show_channel].min())
    #     feature_resized = feature_after_pca[:, 1:4].reshape(num_patches,num_patches, 3)*resized_mask.cpu().squeeze().unsqueeze(-1).numpy()
    #     feature_img = (feature_resized*255).astype(np.uint8)
    #     feature_img_padded = pad_image(feature_img, target_shape)
    #     feature_imgs_list.append(feature_img_padded)

    feature_img = np.stack(feature_imgs_list)
    
    if cls_token_reduced is not None:
        return feature_img, cls_token_img
    else:
        return feature_img, None

# def pad_image(img, target_shape):
#     img = img.permute(0, 3, 1, 2)
#     target_height = target_shape[1]
#     target_width = target_shape[0]
#     padding_size = target_width-target_height
#     transform = transforms.Compose([
#         transforms.Resize(target_height),
#         transforms.Pad((padding_size//2, 0, padding_size//2, 0), fill=0, padding_mode='constant')
#     ])
#     prep_img = transform(img)
#     final_img = prep_img.permute(0,2,3,1).numpy()
#     return final_img

def pad_image(img, target_shape):
    target_height = target_shape[1]
    target_width = target_shape[0]
 
    # Calculate the scale factor while maintaining the aspect ratio
    scale_factor = max(target_width / img.shape[1], target_height / img.shape[0])
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))

    # Resize the image
    resized_image = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)

    # unpad image
    unpad_height = abs(target_height - resized_image.shape[1]) // 2
    unpad_width = abs(target_width - resized_image.shape[0]) // 2

    unpadded_image = resized_image[unpad_height:resized_image.shape[0] - unpad_height, unpad_width:resized_image.shape[1] - unpad_width]

    return unpadded_image