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

DIST = 'l2'

def transform_to_img(img_torch):

    to_pil = transforms.ToPILImage()
    img = to_pil(img_torch)
    img_input = resize(img, SIZE, resize=True, to_pil=True, edge=EDGE_PAD)
    img = resize(img, img_size, resize=True, to_pil=True, edge=EDGE_PAD)
    return img_input, img

def compute_features(model, aug, extractor, views_batch, target_shape, only_dino):
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
    stride = 14 if DINOV2 else 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # indiactor = 'v2' if DINOV2 else 'v1'
    # model_size = model_type.split('vit')[-1]
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

    batch_list = []
    for idx in range(len(views_batch)):
        start_prepare_time = time.time()
        rgb = views_batch[idx][0]
        object_views = views_batch[idx][1:]

        img_input_list = []
        img_list = []
        img_input_src, img_src = transform_to_img(rgb)
        img_input_list.append(img_input_src)
        img_list.append(img_src)
        for view in object_views:
            img_input_trg, img_trg = transform_to_img(view)
            img_input_list.append(img_input_trg)
            img_list.append(img_trg)

        end_prepare_time = time.time()
        # print("Preparation time", end_prepare_time-start_prepare_time)
        
        with torch.no_grad():
            if not only_dino:
                feature_list = []
                for img_input in img_input_list:
                    features = process_features_and_mask(model, aug, img_input, input_text=None,  mask=False, raw=True)
                    feature_list.append(features)

                processed_features_list = co_pca_all(feature_list)
                
                img_desc_list = [] 
                for processed_features in processed_features_list:
                    img_desc = processed_features.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                    img_desc_list.append(img_desc)
    

            if FUSE_DINO:
                img_desc_dino_list = []
                start_extract_time = time.time()
                for img_idx, img in enumerate(img_list):
                    start_time = time.time()
                    img_batch = extractor.preprocess_pil(img) # 1x3x840x840
                    img_desc_dino = extractor.extract_descriptors(img_batch.to(device), layer, facet)
                    end_time = time.time()
                    # print("Single feature extraction:", end_time-start_time)
                    img_desc_dino_list.append(img_desc_dino)
                end_extract_time = time.time()
                # print("Total feature extraction:", end_extract_time-start_extract_time)

            start_postprocess_time = time.time()
            if DIST == 'l1' or DIST == 'l2':
                if FUSE_DINO and not only_dino:
                    img_desc_norm_list = []
                    for img_desc in img_desc_list:
                        img_desc = img_desc / img_desc.norm(dim=-1, keepdim=True)
                        img_desc_norm_list.append(img_desc.cpu())
                if FUSE_DINO:
                    img_desc_dino_norm_list = []
                    for img_desc_dino in img_desc_dino_list:
                        img_desc_dino = img_desc_dino / img_desc_dino.norm(dim=-1, keepdim=True)
                        img_desc_dino_norm_list.append(img_desc_dino.cpu())
            
            if FUSE_DINO and not only_dino:
                # cat two features together
                img_desc_cat_list = [torch.cat((img_desc, img_desc_dino), dim=-1) for img_desc, img_desc_dino in zip(img_desc_norm_list, img_desc_dino_norm_list)]
            if only_dino:
                img_desc_cat_list = img_desc_dino_norm_list

            mask_list = []
            for img in img_list:
                mask = torch.Tensor(resize(img, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask_list.append(mask.cpu())

            end_process_time = time.time()
            # print("Postprocess time:", end_process_time-start_postprocess_time)
            start_time = time.time()
            feature_imgs_list = vis_pca_mask_all(img_desc_cat_list, mask_list, target_shape)
            end_time = time.time()
            # print("Vis pca mask call:", end_time-start_time)
            batch_list.append(feature_imgs_list)
    return batch_list


def co_pca_all(features_list):
    s5_list = []
    s4_list = []
    s3_list = []
    s5_size = None
    s4_size = None
    s3_size = None
    for features in features_list:
        s5_size = features['s5'].shape[-1]
        s4_size = features['s4'].shape[-1]
        s3_size = features['s3'].shape[-1]
        s5 = features['s5'].reshape(features['s5'].shape[0], features['s5'].shape[1], -1)
        s4 = features['s4'].reshape(features['s4'].shape[0], features['s4'].shape[1], -1)
        s3 = features['s3'].reshape(features['s3'].shape[0], features['s3'].shape[1], -1)
        s5_list.append(s5)
        s4_list.append(s4)
        s3_list.append(s3)

    processed_features1 = {}
    processed_features2 = {}
    processed_features3 = {}
    processed_features4 = {}
    processed_features5 = {}
    processed_features6 = {}
    processed_features7 = {}
    processed_features8 = {}
    processed_features9 = {}

    # Define the target dimensions
    target_dims = {'s5': PCA_DIMS[0], 's4': PCA_DIMS[1], 's3': PCA_DIMS[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [s5_list, s4_list, s3_list]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        split_features_shape = features.shape[-1] // len(s5_list)

        # Split the features
        processed_features1[name] = features[:, :, :split_features_shape] # Bx(d)x(t_x)
        processed_features2[name] = features[:, :, split_features_shape:2*split_features_shape] # Bx(d)x(t_y)
        processed_features3[name] = features[:, :, 2*split_features_shape:3*split_features_shape] # Bx(d)x(t_y)
        processed_features4[name] = features[:, :, 3*split_features_shape:4*split_features_shape] # Bx(d)x(t_y)
        processed_features5[name] = features[:, :, 4*split_features_shape:5*split_features_shape] # Bx(d)x(t_y)
        processed_features6[name] = features[:, :, 5*split_features_shape:6*split_features_shape] # Bx(d)x(t_y)
        processed_features7[name] = features[:, :, 6*split_features_shape:7*split_features_shape] # Bx(d)x(t_y)
        processed_features8[name] = features[:, :, 7*split_features_shape:8*split_features_shape] # Bx(d)x(t_y)
        processed_features9[name] = features[:, :, 8*split_features_shape:9*split_features_shape] # Bx(d)x(t_y)

    processed_features_list = [processed_features1, processed_features2, processed_features3, processed_features4, processed_features5, processed_features6, processed_features7, processed_features8, processed_features9]

    features_gether_s4_s5_list = []
    for processed_features in processed_features_list:
        processed_features['s5']=processed_features['s5'].reshape(processed_features['s5'].shape[0], -1, s5_size, s5_size)
        processed_features['s4']=processed_features['s4'].reshape(processed_features['s4'].shape[0], -1, s4_size, s4_size)
        processed_features['s3']=processed_features['s3'].reshape(processed_features['s3'].shape[0], -1, s3_size, s3_size)

        # Upsample s5 spatially by a factor of 2
        processed_features['s5'] = F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]), mode='bilinear', align_corners=False)

        # Concatenate upsampled_s5 and s4 to create a new s5
        processed_features['s5'] = torch.cat([processed_features['s4'], processed_features['s5']], dim=1)

        # Set s3 as the new s4
        processed_features['s4'] = processed_features['s3']

        # Remove s3 from the features dictionary
        processed_features.pop('s3')

        # current order are layer 8, 5, 2
        features_gether_s4_s5 = torch.cat([processed_features['s4'], F.interpolate(processed_features['s5'], size=(processed_features['s4'].shape[-2:]), mode='bilinear')], dim=1)
        features_gether_s4_s5_list.append(features_gether_s4_s5)

    return features_gether_s4_s5_list

def vis_pca_mask_all(feature_list, mask_list, target_shape):
    feature_processed_list = []
    for feature, mask in zip(feature_list, mask_list):
        num_patches = int(math.sqrt(feature.shape[-2]))
        feature = feature.squeeze() # shape (3600,768*2)
        channel_dim = feature.shape[-1]
        feature_reshaped = feature.permute(1,0).reshape(-1,num_patches,num_patches).cuda() # 1500 x 60 x 60
        resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda() # 60 x 60
        feature_upsampled = feature_reshaped * resized_mask.repeat(feature_reshaped.shape[0],1,1)
        feature_processed=feature_upsampled.reshape(channel_dim,-1).permute(1,0)
        feature_processed_list.append(feature_processed)
    
    n_components=4 # the first component is to seperate the object from the background
    start_time = time.time()
    pca = sklearnPCA(n_components=n_components)
    end_time = time.time()
    # print("Dimension reduction PCA:", end_time-start_time)
    feature1_n_featuren = torch.cat(feature_processed_list,dim=0) # shape (7200,768*2)
    feature1_n_featuren = pca.fit_transform(feature1_n_featuren.cpu().numpy()) # shape (7200,3)
    split_shape = feature_processed_list[0].shape[0]
    feature1 = feature1_n_featuren[:split_shape,:] # shape (3600,3)
    feature2 = feature1_n_featuren[split_shape:2*split_shape,:] # shape (3600,3)
    feature3 = feature1_n_featuren[2*split_shape:3*split_shape,:] # shape (3600,3)
    feature4 = feature1_n_featuren[3*split_shape:4*split_shape,:] # shape (3600,3)
    feature5 = feature1_n_featuren[4*split_shape:5*split_shape,:] # shape (3600,3)
    feature6 = feature1_n_featuren[5*split_shape:6*split_shape,:] # shape (3600,3)
    feature7 = feature1_n_featuren[6*split_shape:7*split_shape,:] # shape (3600,3)
    feature8 = feature1_n_featuren[7*split_shape:8*split_shape,:] # shape (3600,3)
    feature9 = feature1_n_featuren[8*split_shape:9*split_shape,:] # shape (3600,3)

    feature_after_pca_list = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]

    feature_imgs_list = []
    for feature_after_pca, mask in zip(feature_after_pca_list, mask_list):
        resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest') # 60 x 60
        for show_channel in range(n_components):
            if show_channel==0:
                continue
            # min max normalize the feature map
            feature_after_pca[:, show_channel] = (feature_after_pca[:, show_channel] - feature_after_pca[:, show_channel].min()) / (feature_after_pca[:, show_channel].max() - feature_after_pca[:, show_channel].min())
        feature_resized = feature_after_pca[:, 1:4].reshape(num_patches,num_patches, 3)*resized_mask.cpu().squeeze().unsqueeze(-1).numpy()
        feature_img = (feature_resized*255).astype(np.uint8)
        feature_img_padded = pad_image(feature_img, target_shape)
        feature_imgs_list.append(feature_img_padded)

    return feature_imgs_list

def pad_image(img, target_shape):
    target_height = target_shape[1]
    target_width = target_shape[0]

    # Calculate the scale factor while maintaining the aspect ratio
    scale_factor = min(target_width / img.shape[1], target_height / img.shape[0])
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))

    # Resize the image
    resized_image = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Center the resized image onto the new black image
    start_x = (target_width - new_size[0]) // 2
    start_y = (target_height - new_size[1]) // 2
    padded_image[start_y:start_y + new_size[1], start_x:start_x + new_size[0]] = resized_image
    return padded_image