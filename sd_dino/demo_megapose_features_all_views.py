import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from extractor_sd import load_model, process_features_and_mask, get_mask
from utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
import matplotlib.pyplot as plt
import sys
from extractor_dino import ViTExtractor
from sklearn.decomposition import PCA as sklearnPCA
import math
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import cv2

MASK = True
VER = "v1-5"
PCA = False
CO_PCA = True
PCA_DIMS = [256, 256, 256]
SIZE =960
EDGE_PAD = False
ANNO = False

FUSE_DINO = 1
ONLY_DINO = 0
DINOV2 = True
MODEL_SIZE = 'base'
DRAW_DENSE = 1
DRAW_SWAP = 1
TEXT_INPUT = False
SEED = 42
TIMESTEP = 100

DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'
if ONLY_DINO:
    FUSE_DINO = True


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP, decoder_only=False)

def compute_pair_feature(model, aug, save_path, files, category, mask=False, dist='cos', real_size=960):
    if type(category) == str:
        category = [category]
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
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.model.patch_embed.patch_size[0] if DINOV2 else extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    
    input_text = "a photo of "+category[-1][0] if TEXT_INPUT else None

    current_save_results = 0

    N = len(files) // 2
    pbar = tqdm(total=N)
    result = []

    # Load image 1
    img1 = Image.open(files[0]).convert('RGB')
    img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    # Load image 2
    img2 = Image.open(files[1]).convert('RGB')
    img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    # Load image 3
    img3 = Image.open(files[2]).convert('RGB')
    img3_input = resize(img3, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img3 = resize(img3, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    # Load image 4
    img4 = Image.open(files[3]).convert('RGB')
    img4_input = resize(img4, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img4 = resize(img4, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    # Load image 5
    img5 = Image.open(files[4]).convert('RGB')
    img5_input = resize(img5, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img5 = resize(img5, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    if 'Anno' in files[0] or ANNO:
        Anno=True
    else:
        Anno=False

    with torch.no_grad():
        # not covered
        if not CO_PCA:
            if not ONLY_DINO:
                img1_desc = process_features_and_mask(model, aug, img1_input, input_text=input_text, mask=False, pca=PCA).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
                img2_desc = process_features_and_mask(model, aug, img2_input, category[-1], input_text=input_text,  mask=mask, pca=PCA).reshape(1,1,-1, num_patches**2).permute(0,1,3,2)
            if FUSE_DINO:
                img1_batch = extractor.preprocess_pil(img1)
                img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet)
                img2_batch = extractor.preprocess_pil(img2)
                img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)
        # covered
        else:
            if not ONLY_DINO:
                features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text,  mask=False, raw=True) # s2, s3, s4, s5
                features2 = process_features_and_mask(model, aug, img2_input, input_text=input_text,  mask=False, raw=True)
                features3 = process_features_and_mask(model, aug, img3_input, input_text=input_text,  mask=False, raw=True)
                features4 = process_features_and_mask(model, aug, img4_input, input_text=input_text,  mask=False, raw=True)
                features5 = process_features_and_mask(model, aug, img5_input, input_text=input_text,  mask=False, raw=True)
                processed_features1, processed_features2, processed_features3, processed_features4, processed_features5 = co_pca_all(features1, features2, features3, features4, features5, PCA_DIMS) # 1x768x60x60
                img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2) # 1, 1, 3600, 768
                img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                img3_desc = processed_features3.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                img4_desc = processed_features4.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
                img5_desc = processed_features5.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
            if FUSE_DINO:
                img1_batch = extractor.preprocess_pil(img1) # 1x3x840x840
                img1_desc_dino = extractor.extract_descriptors(img1_batch.to(device), layer, facet) # 1, 1, 3600, 768
                img2_batch = extractor.preprocess_pil(img2)
                img2_desc_dino = extractor.extract_descriptors(img2_batch.to(device), layer, facet)
                img3_batch = extractor.preprocess_pil(img3)
                img3_desc_dino = extractor.extract_descriptors(img3_batch.to(device), layer, facet)
                img4_batch = extractor.preprocess_pil(img4)
                img4_desc_dino = extractor.extract_descriptors(img4_batch.to(device), layer, facet)
                img5_batch = extractor.preprocess_pil(img5)
                img5_desc_dino = extractor.extract_descriptors(img5_batch.to(device), layer, facet)
            
        if dist == 'l1' or dist == 'l2':
            # normalize the features
            img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
            img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
            img3_desc = img3_desc / img3_desc.norm(dim=-1, keepdim=True)
            img4_desc = img4_desc / img4_desc.norm(dim=-1, keepdim=True)
            img5_desc = img5_desc / img5_desc.norm(dim=-1, keepdim=True)
            if FUSE_DINO:
                img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)
                img3_desc_dino = img3_desc_dino / img3_desc_dino.norm(dim=-1, keepdim=True)
                img4_desc_dino = img4_desc_dino / img4_desc_dino.norm(dim=-1, keepdim=True)
                img5_desc_dino = img5_desc_dino / img5_desc_dino.norm(dim=-1, keepdim=True)

        if FUSE_DINO and not ONLY_DINO:
            # cat two features together
            img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)  # 1, 1, 3600, 1536
            img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)
            img3_desc = torch.cat((img3_desc, img3_desc_dino), dim=-1)
            img4_desc = torch.cat((img4_desc, img4_desc_dino), dim=-1)
            img5_desc = torch.cat((img5_desc, img5_desc_dino), dim=-1)

        if ONLY_DINO:
            img1_desc = img1_desc_dino
            img2_desc = img2_desc_dino
            img3_desc = img3_desc_dino
            img4_desc = img4_desc_dino
            img5_desc = img5_desc_dino

        if DRAW_DENSE:
            if not Anno:
                # mask1 = get_mask(model, aug, img1, category[0])
                # mask2 = get_mask(model, aug, img2, category[-1])
                mask1 = torch.Tensor(resize(img1, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)  # 840x840
                mask2 = torch.Tensor(resize(img2, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask3 = torch.Tensor(resize(img3, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask4 = torch.Tensor(resize(img4, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask5 = torch.Tensor(resize(img5, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
            if Anno:
                mask1 = torch.Tensor(resize(img1, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask2 = torch.Tensor(resize(img2, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask3 = torch.Tensor(resize(img3, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask4 = torch.Tensor(resize(img4, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
                mask5 = torch.Tensor(resize(img5, img_size, resize=True, to_pil=False, edge=EDGE_PAD).mean(-1)>0).to(device)
            if ONLY_DINO or not FUSE_DINO:
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                img3_desc = img3_desc / img3_desc.norm(dim=-1, keepdim=True)
                img4_desc = img4_desc / img4_desc.norm(dim=-1, keepdim=True)
                img5_desc = img5_desc / img5_desc.norm(dim=-1, keepdim=True)
            
            img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
            img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches) # 1x1536x60x60
            img3_desc_reshaped = img3_desc.permute(0,1,3,2).reshape(-1, img3_desc.shape[-1], num_patches, num_patches)
            img4_desc_reshaped = img4_desc.permute(0,1,3,2).reshape(-1, img4_desc.shape[-1], num_patches, num_patches)
            img5_desc_reshaped = img5_desc.permute(0,1,3,2).reshape(-1, img5_desc.shape[-1], num_patches, num_patches)
            src_color_map, dense_output_2, dense_output_3, dense_output_4, dense_output_5 = find_nearest_patchs_all(mask1, mask2, mask3, mask4, mask5, img1, img2, img3, img4, img5, img1_desc_reshaped, img2_desc_reshaped, img3_desc_reshaped, img4_desc_reshaped, img5_desc_reshaped, mask=mask)

            if not os.path.exists(f'{save_path}/{category[0]}'):
                os.makedirs(f'{save_path}/{category[0]}')
            fig_colormap, axes = plt.subplots(1, 5, figsize=(40, 8))
            for ax, img in zip(axes, [src_color_map, dense_output_2, dense_output_3, dense_output_4, dense_output_5]):
                ax.axis('off')
                ax.imshow(img)

            # Save the figure
            fig_colormap.savefig(f'{save_path}/{category[0]}/colormap.png')
            plt.close(fig_colormap)
        
        if DRAW_SWAP:
            if not DRAW_DENSE:
                mask1 = get_mask(model, aug, img1, category[0])
                mask2 = get_mask(model, aug, img2, category[-1])
                mask3 = get_mask(model, aug, img3, category[-1])
                mask4 = get_mask(model, aug, img4, category[-1])
                mask5 = get_mask(model, aug, img5, category[-1])
    

            if (ONLY_DINO or not FUSE_DINO) and not DRAW_DENSE:
                img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
                img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
                img3_desc = img3_desc / img3_desc.norm(dim=-1, keepdim=True)
                img4_desc = img4_desc / img4_desc.norm(dim=-1, keepdim=True)
                img5_desc = img5_desc / img5_desc.norm(dim=-1, keepdim=True)
                
            img1_desc_reshaped = img1_desc.permute(0,1,3,2).reshape(-1, img1_desc.shape[-1], num_patches, num_patches)
            img2_desc_reshaped = img2_desc.permute(0,1,3,2).reshape(-1, img2_desc.shape[-1], num_patches, num_patches)
            img3_desc_reshaped = img3_desc.permute(0,1,3,2).reshape(-1, img3_desc.shape[-1], num_patches, num_patches)
            img4_desc_reshaped = img4_desc.permute(0,1,3,2).reshape(-1, img4_desc.shape[-1], num_patches, num_patches)
            img5_desc_reshaped = img5_desc.permute(0,1,3,2).reshape(-1, img5_desc.shape[-1], num_patches, num_patches)

            src_color_map, dense_output_2, dense_output_3, dense_output_4, dense_output_5 = find_nearest_patchs_replace_all(mask1, mask2, mask3, mask4, mask5, img1, img2, img3, img4, img5, img1_desc_reshaped, img2_desc_reshaped, img3_desc_reshaped, img4_desc_reshaped, img5_desc_reshaped, mask=mask, resolution=156)
            if not os.path.exists(f'{save_path}/{category[0]}'):
                os.makedirs(f'{save_path}/{category[0]}')
            fig_colormap, axes = plt.subplots(1, 5, figsize=(40, 8))
            for ax, img in zip(axes, [src_color_map, dense_output_2, dense_output_3, dense_output_4, dense_output_5]):
                ax.axis('off')
                ax.imshow(img)
            fig_colormap.savefig(f'{save_path}/{category[0]}/swap.png')
            plt.close(fig_colormap)
        if not DRAW_SWAP and not DRAW_DENSE:
            result.append([img1_desc.cpu(), img2_desc.cpu(), img3_desc.cpu(), img4_desc.cpu(), img5_desc.cpu()])
        else:
            result.append([img1_desc.cpu(), img2_desc.cpu(), img3_desc.cpu(), img4_desc.cpu(), img5_desc.cpu(), mask1.cpu(), mask2.cpu(), mask3.cpu(), mask4.cpu(), mask5.cpu()])

    pbar.update(1)
    return result

def find_nearest_patchs_all(mask_src, mask2, mask3, mask4, mask5, image_src, image2, image3, image4, image5, features_src, features2, features3, features4, features5, mask=False, resolution=None, edit_image=None):
    # mask 2 is source
    def polar_color_map(image_shape):
        h, w = image_shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Find the center of the mask
        mask=mask_src.cpu()
        mask_center = np.array(np.where(mask > 0))
        mask_center = np.round(np.mean(mask_center, axis=1)).astype(int)
        mask_center_y, mask_center_x = mask_center

        # Calculate distance and angle based on mask_center
        xx_shifted, yy_shifted = xx - x[mask_center_x], yy - y[mask_center_y]
        max_radius = np.sqrt(h**2 + w**2) / 2
        radius = np.sqrt(xx_shifted**2 + yy_shifted**2) * max_radius
        angle = np.arctan2(yy_shifted, xx_shifted) / (2 * np.pi) + 0.5

        angle = 0.2 + angle * 0.6  # Map angle to the range [0.25, 0.75]
        radius = np.where(radius <= max_radius, radius, max_radius)  # Limit radius values to the unit circle
        radius = 0.2 + radius * 0.6 / max_radius  # Map radius to the range [0.1, 1]

        return angle, radius
    
    if resolution is not None: # resize the feature map to the resolution
        features_src = F.interpolate(features_src, size=resolution, mode='bilinear')
        features2 = F.interpolate(features2, size=resolution, mode='bilinear')
        features3 = F.interpolate(features3, size=resolution, mode='bilinear')
        features4 = F.interpolate(features4, size=resolution, mode='bilinear')
        features5 = F.interpolate(features5, size=resolution, mode='bilinear')
    
    # resize the image to the shape of the feature map
    resized_image_src = resize(image_src, features_src.shape[2], resize=True, to_pil=False)
    resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)
    resized_image3 = resize(image3, features3.shape[2], resize=True, to_pil=False)
    resized_image4 = resize(image4, features4.shape[2], resize=True, to_pil=False)
    resized_image5 = resize(image5, features5.shape[2], resize=True, to_pil=False)

    if mask: # mask the features
        resized_mask_src = F.interpolate(mask_src.cuda().unsqueeze(0).unsqueeze(0).float(), size=features_src.shape[2:], mode='nearest')
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[2:], mode='nearest')
        resized_mask3 = F.interpolate(mask3.cuda().unsqueeze(0).unsqueeze(0).float(), size=features3.shape[2:], mode='nearest')
        resized_mask4 = F.interpolate(mask4.cuda().unsqueeze(0).unsqueeze(0).float(), size=features4.shape[2:], mode='nearest')
        resized_mask5 = F.interpolate(mask5.cuda().unsqueeze(0).unsqueeze(0).float(), size=features5.shape[2:], mode='nearest')
        features_src = features_src * resized_mask_src.repeat(1, features_src.shape[1], 1, 1)
        features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
        features3 = features3 * resized_mask3.repeat(1, features3.shape[1], 1, 1)
        features4 = features4 * resized_mask4.repeat(1, features4.shape[1], 1, 1)
        features5 = features5 * resized_mask5.repeat(1, features5.shape[1], 1, 1)
        # set where mask==0 a very large number
        features_src[(features_src.sum(1)==0).repeat(1, features_src.shape[1], 1, 1)] = 100000
        features2[(features2.sum(1)==0).repeat(1, features2.shape[1], 1, 1)] = 100000
        features3[(features3.sum(1)==0).repeat(1, features3.shape[1], 1, 1)] = 100000
        features4[(features4.sum(1)==0).repeat(1, features4.shape[1], 1, 1)] = 100000
        features5[(features5.sum(1)==0).repeat(1, features5.shape[1], 1, 1)] = 100000

    features_src_2d = features_src.reshape(features_src.shape[1], -1).permute(1, 0).cpu().detach().numpy()
    features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0).cpu().detach().numpy()
    features3_2d = features3.reshape(features3.shape[1], -1).permute(1, 0).cpu().detach().numpy()
    features4_2d = features4.reshape(features4.shape[1], -1).permute(1, 0).cpu().detach().numpy()
    features5_2d = features5.reshape(features5.shape[1], -1).permute(1, 0).cpu().detach().numpy()

    features_src_2d = torch.tensor(features_src_2d).to("cuda")
    features2_2d = torch.tensor(features2_2d).to("cuda")
    features3_2d = torch.tensor(features3_2d).to("cuda")
    features4_2d = torch.tensor(features4_2d).to("cuda")
    features5_2d = torch.tensor(features5_2d).to("cuda")
    resized_image_src = torch.tensor(resized_image_src).to("cuda").float()
    resized_image2 = torch.tensor(resized_image2).to("cuda").float()
    resized_image3 = torch.tensor(resized_image3).to("cuda").float()
    resized_image4 = torch.tensor(resized_image4).to("cuda").float()
    resized_image5 = torch.tensor(resized_image5).to("cuda").float()

    mask_src = F.interpolate(mask_src.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image_src.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask3 = F.interpolate(mask3.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image3.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask4 = F.interpolate(mask4.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image4.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask5 = F.interpolate(mask5.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image5.shape[:2], mode='nearest').squeeze(0).squeeze(0)

    # Mask the images
    resized_image_src = resized_image_src * mask_src.unsqueeze(-1).repeat(1, 1, 3)
    resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
    resized_image3 = resized_image3 * mask3.unsqueeze(-1).repeat(1, 1, 3)
    resized_image4 = resized_image4 * mask4.unsqueeze(-1).repeat(1, 1, 3)
    resized_image5 = resized_image5 * mask5.unsqueeze(-1).repeat(1, 1, 3)
    # Normalize the images to the range [0, 1]
    resized_image_src = (resized_image_src - resized_image_src.min()) / (resized_image_src.max() - resized_image_src.min())
    resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())
    resized_image3 = (resized_image3 - resized_image3.min()) / (resized_image3.max() - resized_image3.min())
    resized_image4 = (resized_image4 - resized_image4.min()) / (resized_image4.max() - resized_image4.min())
    resized_image5 = (resized_image5 - resized_image5.min()) / (resized_image5.max() - resized_image5.min())

    angle, radius = polar_color_map(resized_image_src.shape)

    angle_mask = angle * mask_src.cpu().numpy()
    radius_mask = radius * mask_src.cpu().numpy()

    hsv_mask = np.zeros(resized_image_src.shape, dtype=np.float32)
    hsv_mask[:, :, 0] = angle_mask
    hsv_mask[:, :, 1] = radius_mask
    hsv_mask[:, :, 2] = 1

    rainbow_mask_src = cv2.cvtColor((hsv_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR) / 255

    if edit_image is not None:
        rainbow_mask_src = cv2.imread(edit_image, cv2.IMREAD_COLOR)
        rainbow_mask_src = cv2.cvtColor(rainbow_mask_src, cv2.COLOR_BGR2RGB) / 255
        rainbow_mask_src = cv2.resize(rainbow_mask_src, (resized_image_src.shape[1], resized_image_src.shape[0]))

    # Apply the rainbow mask to image2
    rainbow_image_src = rainbow_mask_src * mask_src.cpu().numpy()[:, :, None]

    # Create a white background image
    background_color = np.array([1, 1, 1], dtype=np.float32)
    background_image = np.ones(resized_image_src.shape, dtype=np.float32) * background_color

    # Apply the rainbow mask to image2 only in the regions where mask2 is 1
    rainbow_image_src = np.where(mask_src.cpu().numpy()[:, :, None] == 1, rainbow_mask_src, background_image)
    
    nearest_patches_2 = []
    distances_2 = torch.cdist(features2_2d, features_src_2d)
    nearest_patch_indices_2 = torch.argmin(distances_2, dim=1)
    nearest_patches_2 = torch.index_select(torch.tensor(rainbow_mask_src).cuda().reshape(-1, 3), 0, nearest_patch_indices_2)
    nearest_patches_image_2 = nearest_patches_2.reshape(resized_image2.shape)
    nearest_patches_image_2 = (nearest_patches_image_2).cpu().numpy()

    nearest_patches_3 = []
    distances_3 = torch.cdist(features3_2d, features_src_2d)
    nearest_patch_indices_3 = torch.argmin(distances_3, dim=1)
    nearest_patches_3 = torch.index_select(torch.tensor(rainbow_mask_src).cuda().reshape(-1, 3), 0, nearest_patch_indices_3)
    nearest_patches_image_3 = nearest_patches_3.reshape(resized_image3.shape)
    nearest_patches_image_3 = (nearest_patches_image_3).cpu().numpy()

    nearest_patches_4 = []
    distances_4 = torch.cdist(features4_2d, features_src_2d)
    nearest_patch_indices_4 = torch.argmin(distances_4, dim=1)
    nearest_patches_4 = torch.index_select(torch.tensor(rainbow_mask_src).cuda().reshape(-1, 3), 0, nearest_patch_indices_4)
    nearest_patches_image_4 = nearest_patches_4.reshape(resized_image4.shape)
    nearest_patches_image_4 = (nearest_patches_image_4).cpu().numpy()

    nearest_patches_5 = []
    distances_5 = torch.cdist(features5_2d, features_src_2d)
    nearest_patch_indices_5 = torch.argmin(distances_5, dim=1)
    nearest_patches_5 = torch.index_select(torch.tensor(rainbow_mask_src).cuda().reshape(-1, 3), 0, nearest_patch_indices_5)
    nearest_patches_image_5 = nearest_patches_5.reshape(resized_image5.shape)
    nearest_patches_image_5 = (nearest_patches_image_5).cpu().numpy()


    # TODO: upsample the nearest_patches_image to the resolution of the original image
    # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
    # rainbow_image2 = F.interpolate(rainbow_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

    rainbow_image_src = torch.tensor(rainbow_image_src).to("cuda")
    resized_image_src = (rainbow_image_src).cpu().numpy()

    return resized_image_src, nearest_patches_image_2, nearest_patches_image_3, nearest_patches_image_4, nearest_patches_image_5

def find_nearest_patchs_replace_all(mask_src, mask2, mask3, mask4, mask5, image_src, image2, image3, image4, image5, features_src, features2, features3, features4, features5, mask=False, resolution=128, draw_gif=False, save_path=None, gif_reverse=False):
    
    if resolution is not None: # resize the feature map to the resolution
        features_src = F.interpolate(features_src, size=resolution, mode='bilinear')
        features2 = F.interpolate(features2, size=resolution, mode='bilinear')
        features3 = F.interpolate(features3, size=resolution, mode='bilinear')
        features4 = F.interpolate(features4, size=resolution, mode='bilinear')
        features5 = F.interpolate(features5, size=resolution, mode='bilinear')
    
    # resize the image to the shape of the feature map
    resized_image_src = resize(image_src, features_src.shape[2], resize=True, to_pil=False)
    resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)
    resized_image3 = resize(image3, features3.shape[2], resize=True, to_pil=False)
    resized_image4 = resize(image4, features4.shape[2], resize=True, to_pil=False)
    resized_image5 = resize(image5, features5.shape[2], resize=True, to_pil=False)

    if mask: # mask the features
        resized_mask_src = F.interpolate(mask_src.cuda().unsqueeze(0).unsqueeze(0).float(), size=features_src.shape[2:], mode='nearest')
        resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[2:], mode='nearest')
        resized_mask3 = F.interpolate(mask3.cuda().unsqueeze(0).unsqueeze(0).float(), size=features3.shape[2:], mode='nearest')
        resized_mask4 = F.interpolate(mask4.cuda().unsqueeze(0).unsqueeze(0).float(), size=features4.shape[2:], mode='nearest')
        resized_mask5 = F.interpolate(mask5.cuda().unsqueeze(0).unsqueeze(0).float(), size=features5.shape[2:], mode='nearest')

        features_src = features_src * resized_mask_src.repeat(1, features_src.shape[1], 1, 1)
        features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
        features3 = features3 * resized_mask3.repeat(1, features3.shape[1], 1, 1)
        features4 = features4 * resized_mask4.repeat(1, features4.shape[1], 1, 1)
        features5 = features5 * resized_mask5.repeat(1, features5.shape[1], 1, 1)

        # set where mask==0 a very large number
        features_src[(features_src.sum(1)==0).repeat(1, features_src.shape[1], 1, 1)] = 100000
        features2[(features2.sum(1)==0).repeat(1, features2.shape[1], 1, 1)] = 100000
        features3[(features3.sum(1)==0).repeat(1, features3.shape[1], 1, 1)] = 100000
        features4[(features4.sum(1)==0).repeat(1, features4.shape[1], 1, 1)] = 100000
        features5[(features5.sum(1)==0).repeat(1, features5.shape[1], 1, 1)] = 100000
    
    features_src_2d = features_src.reshape(features_src.shape[1], -1).permute(1, 0)
    features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0)
    features3_2d = features3.reshape(features3.shape[1], -1).permute(1, 0)
    features4_2d = features4.reshape(features4.shape[1], -1).permute(1, 0)
    features5_2d = features5.reshape(features5.shape[1], -1).permute(1, 0)

    resized_image_src = torch.tensor(resized_image_src).to("cuda").float()
    resized_image2 = torch.tensor(resized_image2).to("cuda").float()
    resized_image3 = torch.tensor(resized_image3).to("cuda").float()
    resized_image4 = torch.tensor(resized_image4).to("cuda").float()
    resized_image5 = torch.tensor(resized_image5).to("cuda").float()

    mask_src = F.interpolate(mask_src.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image_src.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask3 = F.interpolate(mask3.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image3.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask4 = F.interpolate(mask4.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image4.shape[:2], mode='nearest').squeeze(0).squeeze(0)
    mask5 = F.interpolate(mask5.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image5.shape[:2], mode='nearest').squeeze(0).squeeze(0)

    # Mask the images
    resized_image_src = resized_image_src * mask_src.unsqueeze(-1).repeat(1, 1, 3)
    resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
    resized_image3 = resized_image3 * mask3.unsqueeze(-1).repeat(1, 1, 3)
    resized_image4 = resized_image4 * mask4.unsqueeze(-1).repeat(1, 1, 3)
    resized_image5 = resized_image5 * mask5.unsqueeze(-1).repeat(1, 1, 3)

    # Normalize the images to the range [0, 1]
    resized_image_src = (resized_image_src - resized_image_src.min()) / (resized_image_src.max() - resized_image_src.min())
    resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())
    resized_image3 = (resized_image3 - resized_image3.min()) / (resized_image3.max() - resized_image3.min())
    resized_image4 = (resized_image4 - resized_image4.min()) / (resized_image4.max() - resized_image4.min())
    resized_image5 = (resized_image5 - resized_image5.min()) / (resized_image5.max() - resized_image5.min())

    distances_2 = torch.cdist(features2_2d, features_src_2d)
    nearest_patch_indices_2 = torch.argmin(distances_2, dim=1)
    nearest_patches_2 = torch.index_select(resized_image_src.cuda().clone().detach().reshape(-1, 3), 0, nearest_patch_indices_2)
    nearest_patches_image_2 = nearest_patches_2.reshape(resized_image2.shape)

    distances_3 = torch.cdist(features3_2d, features_src_2d)
    nearest_patch_indices_3 = torch.argmin(distances_3, dim=1)
    nearest_patches_3 = torch.index_select(resized_image_src.cuda().clone().detach().reshape(-1, 3), 0, nearest_patch_indices_3)
    nearest_patches_image_3 = nearest_patches_3.reshape(resized_image3.shape)

    distances_4 = torch.cdist(features4_2d, features_src_2d)
    nearest_patch_indices_4 = torch.argmin(distances_4, dim=1)
    nearest_patches_4 = torch.index_select(resized_image_src.cuda().clone().detach().reshape(-1, 3), 0, nearest_patch_indices_4)
    nearest_patches_image_4 = nearest_patches_4.reshape(resized_image4.shape)

    distances_5 = torch.cdist(features5_2d, features_src_2d)
    nearest_patch_indices_5 = torch.argmin(distances_5, dim=1)
    nearest_patches_5 = torch.index_select(resized_image_src.cuda().clone().detach().reshape(-1, 3), 0, nearest_patch_indices_5)
    nearest_patches_image_5 = nearest_patches_5.reshape(resized_image5.shape)

    # if draw_gif:
    #     assert save_path is not None, "save_path must be provided when draw_gif is True"
    #     img_1 = resize(image1, features1.shape[2], resize=True, to_pil=True)
    #     img_src = resize(image_src, features_src.shape[2], resize=True, to_pil=True)
    #     mapping = torch.zeros((img_1.size[1], img_1.size[0], 2))
    #     for i in range(len(nearest_patch_indices)):
    #         mapping[i // img_1.size[0], i % img_1.size[0]] = torch.tensor([nearest_patch_indices[i] // img_src.size[0], nearest_patch_indices[i] % img_src.size[0]])
    #     # animate_image_transfer(img_1, img_src, mapping, save_path) if gif_reverse else animate_image_transfer_reverse(img_1, img_src, mapping, save_path)

    # TODO: upsample the nearest_patches_image to the resolution of the original image
    # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
    # resized_image2 = F.interpolate(resized_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

    nearest_patches_image_2 = (nearest_patches_image_2).cpu().numpy()
    nearest_patches_image_3 = (nearest_patches_image_3).cpu().numpy()
    nearest_patches_image_4 = (nearest_patches_image_4).cpu().numpy()
    nearest_patches_image_5 = (nearest_patches_image_5).cpu().numpy()


    resized_image_src = (resized_image_src).cpu().numpy()

    return resized_image_src, nearest_patches_image_2, nearest_patches_image_3, nearest_patches_image_4, nearest_patches_image_5

def vis_pca_mask(result,save_path):
    # PCA visualization mask version
    for idx, (feature1,feature2,feature3,feature4,feature5,mask1,mask2,mask3,mask4,mask5) in enumerate(result):
        # feature1 shape (1,1,3600,768*2)
        # feature2 shape (1,1,3600,768*2)
        num_patches = int(math.sqrt(feature1.shape[-2]))
        # pca the concatenated feature to 3 dimensions
        feature1 = feature1.squeeze() # shape (3600,768*2)
        feature2 = feature2.squeeze() # shape (3600,768*2)
        feature3 = feature3.squeeze() # shape (3600,768*2)
        feature4 = feature4.squeeze() # shape (3600,768*2)
        feature5 = feature5.squeeze() # shape (3600,768*2)
        channel_dim = feature1.shape[-1]
        # resize back
        src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda() # 1500 x 60 x 60
        feature2_reshaped = feature2.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature3_reshaped = feature3.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature4_reshaped = feature4.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature5_reshaped = feature5.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()

        resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda() # 60 x 60
        resized_mask_2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_3 = F.interpolate(mask3.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_4 = F.interpolate(mask4.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_5 = F.interpolate(mask5.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()

        src_feature_upsampled = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0],1,1)
        feature_upsampled_2 = feature2_reshaped * resized_mask_2.repeat(src_feature_reshaped.shape[0],1,1)
        feature_upsampled_3 = feature3_reshaped * resized_mask_3.repeat(src_feature_reshaped.shape[0],1,1)
        feature_upsampled_4 = feature4_reshaped * resized_mask_4.repeat(src_feature_reshaped.shape[0],1,1)
        feature_upsampled_5 = feature5_reshaped * resized_mask_5.repeat(src_feature_reshaped.shape[0],1,1)

        feature1=src_feature_upsampled.reshape(channel_dim,-1).permute(1,0)
        feature2=feature_upsampled_2.reshape(channel_dim,-1).permute(1,0)
        feature3=feature_upsampled_3.reshape(channel_dim,-1).permute(1,0)
        feature4=feature_upsampled_4.reshape(channel_dim,-1).permute(1,0)
        feature5=feature_upsampled_5.reshape(channel_dim,-1).permute(1,0)

        n_components=4 # the first component is to seperate the object from the background
        pca = sklearnPCA(n_components=n_components)
        feature1_n_featuren = torch.cat((feature1,feature2,feature3,feature4,feature5),dim=0) # shape (7200,768*2)
        feature1_n_featuren = pca.fit_transform(feature1_n_featuren.cpu().numpy()) # shape (7200,3)
        feature1 = feature1_n_featuren[:feature1.shape[0],:] # shape (3600,3)
        feature2 = feature1_n_featuren[feature1.shape[0]:2*feature1.shape[0],:] # shape (3600,3)
        feature3 = feature1_n_featuren[2*feature1.shape[0]:3*feature1.shape[0],:] # shape (3600,3)
        feature4 = feature1_n_featuren[3*feature1.shape[0]:4*feature1.shape[0],:] # shape (3600,3)
        feature5 = feature1_n_featuren[4*feature1.shape[0]:5*feature1.shape[0],:] # shape (3600,3)
        
        
        fig, axes = plt.subplots(4, 5, figsize=(25, 5))
        for show_channel in range(n_components):
            if show_channel==0:
                continue
            # min max normalize the feature map
            feature1[:, show_channel] = (feature1[:, show_channel] - feature1[:, show_channel].min()) / (feature1[:, show_channel].max() - feature1[:, show_channel].min())
            feature2[:, show_channel] = (feature2[:, show_channel] - feature2[:, show_channel].min()) / (feature2[:, show_channel].max() - feature2[:, show_channel].min())
            feature3[:, show_channel] = (feature3[:, show_channel] - feature3[:, show_channel].min()) / (feature3[:, show_channel].max() - feature3[:, show_channel].min())
            feature4[:, show_channel] = (feature4[:, show_channel] - feature4[:, show_channel].min()) / (feature4[:, show_channel].max() - feature4[:, show_channel].min())
            feature5[:, show_channel] = (feature5[:, show_channel] - feature5[:, show_channel].min()) / (feature5[:, show_channel].max() - feature5[:, show_channel].min())

            feature1_first_channel = feature1[:, show_channel].reshape(num_patches,num_patches)
            feature2_first_channel = feature2[:, show_channel].reshape(num_patches,num_patches)
            feature3_first_channel = feature3[:, show_channel].reshape(num_patches,num_patches)
            feature4_first_channel = feature4[:, show_channel].reshape(num_patches,num_patches)
            feature5_first_channel = feature5[:, show_channel].reshape(num_patches,num_patches)

            axes[show_channel-1, 0].imshow(feature1_first_channel)
            axes[show_channel-1, 0].axis('off')
            axes[show_channel-1, 1].imshow(feature2_first_channel)
            axes[show_channel-1, 1].axis('off')
            axes[show_channel-1, 2].imshow(feature3_first_channel)
            axes[show_channel-1, 2].axis('off')
            axes[show_channel-1, 3].imshow(feature4_first_channel)
            axes[show_channel-1, 3].axis('off')
            axes[show_channel-1, 4].imshow(feature5_first_channel)
            axes[show_channel-1, 4].axis('off')

            axes[show_channel-1, 0].set_title('Feature 1 - Channel {}'.format(show_channel ), fontsize=14)
            axes[show_channel-1, 1].set_title('Feature 2 - Channel {}'.format(show_channel ), fontsize=14)
            axes[show_channel-1, 1].set_title('Feature 3 - Channel {}'.format(show_channel ), fontsize=14)
            axes[show_channel-1, 1].set_title('Feature 4 - Channel {}'.format(show_channel ), fontsize=14)
            axes[show_channel-1, 1].set_title('Feature 5 - Channel {}'.format(show_channel ), fontsize=14)


        feature1_resized = feature1[:, 1:4].reshape(num_patches,num_patches, 3)
        feature2_resized = feature2[:, 1:4].reshape(num_patches,num_patches, 3)
        feature3_resized = feature3[:, 1:4].reshape(num_patches,num_patches, 3)
        feature4_resized = feature4[:, 1:4].reshape(num_patches,num_patches, 3)
        feature5_resized = feature5[:, 1:4].reshape(num_patches,num_patches, 3)

        axes[3, 0].imshow(feature1_resized)
        axes[3, 0].axis('off')
        axes[3, 1].imshow(feature2_resized)
        axes[3, 1].axis('off')
        axes[3, 2].imshow(feature3_resized)
        axes[3, 2].axis('off')
        axes[3, 3].imshow(feature4_resized)
        axes[3, 3].axis('off')
        axes[3, 4].imshow(feature5_resized)
        axes[3, 4].axis('off')
        axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
        axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)
        axes[3, 2].set_title('Feature 3 - All Channels', fontsize=14)
        axes[3, 3].set_title('Feature 4 - All Channels', fontsize=14)
        axes[3, 4].set_title('Feature 5 - All Channels', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(save_path+f'/masked_pca_{idx}.png', dpi=300)
        plt.close()

def vis_pca(result,save_path,src_img_path,img_path_2,img_path_3,img_path_4,img_path_5):
    # PCA visualization
    for idx, (feature1,feature2,feature3,feature4,feature5,mask1,mask2,mask3,mask4,mask5) in enumerate(result):
        # feature1 shape (1,1,3600,768*2)
        # feature2 shape (1,1,3600,768*2)
        num_patches=int(math.sqrt(feature1.shape[2]))
        # pca the concatenated feature to 3 dimensions
        feature1 = feature1.squeeze() # shape (3600,768*2)
        feature2 = feature2.squeeze() # shape (3600,768*2)
        feature3 = feature3.squeeze() # shape (3600,768*2)
        feature4 = feature4.squeeze() # shape (3600,768*2)
        feature5 = feature5.squeeze() # shape (3600,768*2)
        chennel_dim = feature1.shape[-1]
        # resize back
        h1, w1 = Image.open(src_img_path).size
        scale_h1 = h1/num_patches
        scale_w1 = w1/num_patches
        
        if scale_h1 > scale_w1:
            scale = scale_h1
            scaled_w = int(w1/scale)
            feature1 = feature1.reshape(num_patches,num_patches,chennel_dim)
            feature1_uncropped=feature1[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w1
            scaled_h = int(h1/scale)
            feature1 = feature1.reshape(num_patches,num_patches,chennel_dim)
            feature1_uncropped=feature1[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]
        
        h2, w2 = Image.open(img_path_2).size
        scale_h2 = h2/num_patches
        scale_w2 = w2/num_patches
        if scale_h2 > scale_w2:
            scale = scale_h2
            scaled_w = int(w2/scale)
            feature2 = feature2.reshape(num_patches,num_patches,chennel_dim)
            feature2_uncropped=feature2[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w2
            scaled_h = int(h2/scale)
            feature2 = feature2.reshape(num_patches,num_patches,chennel_dim)
            feature2_uncropped=feature2[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]

        h3, w3 = Image.open(img_path_3).size
        scale_h3 = h3/num_patches
        scale_w3 = w3/num_patches
        if scale_h3 > scale_w3:
            scale = scale_h3
            scaled_w = int(w3/scale)
            feature3 = feature3.reshape(num_patches,num_patches,chennel_dim)
            feature3_uncropped=feature3[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w3
            scaled_h = int(h3/scale)
            feature3 = feature3.reshape(num_patches,num_patches,chennel_dim)
            feature3_uncropped=feature3[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]
            
        h4, w4 = Image.open(img_path_4).size
        scale_h4 = h4/num_patches
        scale_w4 = w4/num_patches
        if scale_h4 > scale_w4:
            scale = scale_h4
            scaled_w = int(w4/scale)
            feature4 = feature4.reshape(num_patches,num_patches,chennel_dim)
            feature4_uncropped=feature4[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w4
            scaled_h = int(h4/scale)
            feature4 = feature4.reshape(num_patches,num_patches,chennel_dim)
            feature4_uncropped=feature4[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]

        h5, w5 = Image.open(img_path_5).size
        scale_h5 = h5/num_patches
        scale_w5 = w5/num_patches
        if scale_h5 > scale_w5:
            scale = scale_h5
            scaled_w = int(w5/scale)
            feature5 = feature5.reshape(num_patches,num_patches,chennel_dim)
            feature5_uncropped=feature5[(num_patches-scaled_w)//2:num_patches-(num_patches-scaled_w)//2,:,:]
        else:
            scale = scale_w5
            scaled_h = int(h5/scale)
            feature5 = feature5.reshape(num_patches,num_patches,chennel_dim)
            feature5_uncropped=feature5[:,(num_patches-scaled_h)//2:num_patches-(num_patches-scaled_h)//2,:]


        f1_shape=feature1_uncropped.shape[:2]
        f2_shape=feature2_uncropped.shape[:2]
        f3_shape=feature3_uncropped.shape[:2]
        f4_shape=feature4_uncropped.shape[:2]
        f5_shape=feature5_uncropped.shape[:2]

        feature1 = feature1_uncropped.reshape(f1_shape[0]*f1_shape[1],chennel_dim)
        feature2 = feature2_uncropped.reshape(f2_shape[0]*f2_shape[1],chennel_dim)
        feature3 = feature3_uncropped.reshape(f3_shape[0]*f3_shape[1],chennel_dim)
        feature4 = feature4_uncropped.reshape(f4_shape[0]*f4_shape[1],chennel_dim)
        feature5 = feature5_uncropped.reshape(f5_shape[0]*f5_shape[1],chennel_dim)

        n_components=3
        pca = sklearnPCA(n_components=n_components)
        feature1_n_feature_n = torch.cat((feature1,feature2,feature3,feature4,feature5),dim=0) # shape (7200,768*2)
        feature1_n_feature_n = pca.fit_transform(feature1_n_feature_n.cpu().numpy()) # shape (7200,3)
        feature1 = feature1_n_feature_n[:feature1.shape[0],:] # shape (3600,3)
        feature2 = feature1_n_feature_n[feature1.shape[0]:2*feature1.shape[0],:] # shape (3600,3)
        feature3 = feature1_n_feature_n[2*feature1.shape[0]:3*feature1.shape[0],:] # shape (3600,3)
        feature4 = feature1_n_feature_n[3*feature1.shape[0]:4*feature1.shape[0],:] # shape (3600,3)
        feature5 = feature1_n_feature_n[4*feature1.shape[0]:5*feature1.shape[0],:] # shape (3600,3)
        
        fig, axes = plt.subplots(4, 5, figsize=(10, 35))
        for show_channel in range(n_components):
            # min max normalize the feature map
            feature1[:, show_channel] = (feature1[:, show_channel] - feature1[:, show_channel].min()) / (feature1[:, show_channel].max() - feature1[:, show_channel].min())
            feature2[:, show_channel] = (feature2[:, show_channel] - feature2[:, show_channel].min()) / (feature2[:, show_channel].max() - feature2[:, show_channel].min())
            feature3[:, show_channel] = (feature3[:, show_channel] - feature3[:, show_channel].min()) / (feature3[:, show_channel].max() - feature3[:, show_channel].min())
            feature4[:, show_channel] = (feature4[:, show_channel] - feature4[:, show_channel].min()) / (feature4[:, show_channel].max() - feature4[:, show_channel].min())
            feature5[:, show_channel] = (feature5[:, show_channel] - feature5[:, show_channel].min()) / (feature5[:, show_channel].max() - feature5[:, show_channel].min())

            feature1_first_channel = feature1[:, show_channel].reshape(f1_shape[0], f1_shape[1])
            feature2_first_channel = feature2[:, show_channel].reshape(f2_shape[0], f2_shape[1])
            feature3_first_channel = feature3[:, show_channel].reshape(f3_shape[0], f3_shape[1])
            feature4_first_channel = feature4[:, show_channel].reshape(f4_shape[0], f4_shape[1])
            feature5_first_channel = feature5[:, show_channel].reshape(f5_shape[0], f5_shape[1])

            axes[show_channel, 0].imshow(feature1_first_channel)
            axes[show_channel, 0].axis('off')
            axes[show_channel, 1].imshow(feature2_first_channel)
            axes[show_channel, 1].axis('off')
            axes[show_channel, 2].axis('off')
            axes[show_channel, 2].imshow(feature3_first_channel)
            axes[show_channel, 3].axis('off')
            axes[show_channel, 3].imshow(feature4_first_channel)
            axes[show_channel, 4].axis('off')
            axes[show_channel, 4].imshow(feature5_first_channel)

            axes[show_channel, 0].set_title('Feature 1 - Channel {}'.format(show_channel + 1), fontsize=14)
            axes[show_channel, 1].set_title('Feature 2 - Channel {}'.format(show_channel + 1), fontsize=14)
            axes[show_channel, 2].set_title('Feature 3 - Channel {}'.format(show_channel + 1), fontsize=14)
            axes[show_channel, 3].set_title('Feature 4 - Channel {}'.format(show_channel + 1), fontsize=14)
            axes[show_channel, 4].set_title('Feature 5 - Channel {}'.format(show_channel + 1), fontsize=14)


        feature1_resized = feature1[:, :3].reshape(f1_shape[0], f1_shape[1], 3)
        feature2_resized = feature2[:, :3].reshape(f2_shape[0], f2_shape[1], 3)
        feature3_resized = feature3[:, :3].reshape(f3_shape[0], f3_shape[1], 3)
        feature4_resized = feature4[:, :3].reshape(f4_shape[0], f4_shape[1], 3)
        feature5_resized = feature5[:, :3].reshape(f5_shape[0], f5_shape[1], 3)

        axes[3, 0].imshow(feature1_resized)
        axes[3, 0].axis('off')
        axes[3, 1].imshow(feature2_resized)
        axes[3, 1].axis('off')
        axes[3, 2].imshow(feature3_resized)
        axes[3, 2].axis('off')
        axes[3, 3].imshow(feature4_resized)
        axes[3, 3].axis('off')
        axes[3, 4].imshow(feature5_resized)
        axes[3, 4].axis('off')
        axes[3, 0].set_title('Feature 1 - All Channels', fontsize=14)
        axes[3, 1].set_title('Feature 2 - All Channels', fontsize=14)
        axes[3, 2].set_title('Feature 3 - All Channels', fontsize=14)
        axes[3, 3].set_title('Feature 4 - All Channels', fontsize=14)
        axes[3, 4].set_title('Feature 5 - All Channels', fontsize=14)

        plt.tight_layout()
        plt.show()
        fig.savefig(save_path+f'/pca_{idx}.png', dpi=300)

def co_pca_all(features1, features2, features3, features4, features5, dim=[128,128,128]):
    
    processed_features1 = {}
    processed_features2 = {}
    processed_features3 = {}
    processed_features4 = {}
    processed_features5 = {}
    s5_size = features1['s5'].shape[-1]
    s4_size = features1['s4'].shape[-1]
    s3_size = features1['s3'].shape[-1]
    # Get the feature tensors
    s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
    s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
    s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)

    s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
    s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
    s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)

    s5_3 = features3['s5'].reshape(features3['s5'].shape[0], features3['s5'].shape[1], -1)
    s4_3 = features3['s4'].reshape(features3['s4'].shape[0], features3['s4'].shape[1], -1)
    s3_3 = features3['s3'].reshape(features3['s3'].shape[0], features3['s3'].shape[1], -1)

    s5_4 = features4['s5'].reshape(features4['s5'].shape[0], features4['s5'].shape[1], -1)
    s4_4 = features4['s4'].reshape(features4['s4'].shape[0], features4['s4'].shape[1], -1)
    s3_4 = features4['s3'].reshape(features4['s3'].shape[0], features4['s3'].shape[1], -1)

    s5_5 = features5['s5'].reshape(features5['s5'].shape[0], features5['s5'].shape[1], -1)
    s4_5 = features5['s4'].reshape(features5['s4'].shape[0], features5['s4'].shape[1], -1)
    s3_5 = features5['s3'].reshape(features5['s3'].shape[0], features5['s3'].shape[1], -1)

    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [[s5_1, s5_2, s5_3, s5_4, s5_5], [s4_1, s4_2, s4_3, s4_4, s4_5], [s3_1, s3_2, s3_3, s3_4, s3_5]]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        split_features_shape = features.shape[-1] // 5

        # Split the features
        processed_features1[name] = features[:, :, :split_features_shape] # Bx(d)x(t_x)
        processed_features2[name] = features[:, :, split_features_shape:2*split_features_shape] # Bx(d)x(t_y)
        processed_features3[name] = features[:, :, 2*split_features_shape:3*split_features_shape] # Bx(d)x(t_y)
        processed_features4[name] = features[:, :, 3*split_features_shape:4*split_features_shape] # Bx(d)x(t_y)
        processed_features5[name] = features[:, :, 4*split_features_shape:] # Bx(d)x(t_y)

    # reshape the features
    processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size, s5_size)
    processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size, s4_size)
    processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size, s3_size)

    processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size, s5_size)
    processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size, s4_size)
    processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size, s3_size)

    processed_features3['s5']=processed_features3['s5'].reshape(processed_features3['s5'].shape[0], -1, s5_size, s5_size)
    processed_features3['s4']=processed_features3['s4'].reshape(processed_features3['s4'].shape[0], -1, s4_size, s4_size)
    processed_features3['s3']=processed_features3['s3'].reshape(processed_features3['s3'].shape[0], -1, s3_size, s3_size)

    processed_features4['s5']=processed_features4['s5'].reshape(processed_features4['s5'].shape[0], -1, s5_size, s5_size)
    processed_features4['s4']=processed_features4['s4'].reshape(processed_features4['s4'].shape[0], -1, s4_size, s4_size)
    processed_features4['s3']=processed_features4['s3'].reshape(processed_features4['s3'].shape[0], -1, s3_size, s3_size)

    processed_features5['s5']=processed_features5['s5'].reshape(processed_features5['s5'].shape[0], -1, s5_size, s5_size)
    processed_features5['s4']=processed_features5['s4'].reshape(processed_features5['s4'].shape[0], -1, s4_size, s4_size)
    processed_features5['s3']=processed_features5['s3'].reshape(processed_features5['s3'].shape[0], -1, s3_size, s3_size)

    # Upsample s5 spatially by a factor of 2
    processed_features1['s5'] = F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features2['s5'] = F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features3['s5'] = F.interpolate(processed_features3['s5'], size=(processed_features3['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features4['s5'] = F.interpolate(processed_features4['s5'], size=(processed_features4['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features5['s5'] = F.interpolate(processed_features5['s5'], size=(processed_features5['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1['s5'] = torch.cat([processed_features1['s4'], processed_features1['s5']], dim=1)
    processed_features2['s5'] = torch.cat([processed_features2['s4'], processed_features2['s5']], dim=1)
    processed_features3['s5'] = torch.cat([processed_features3['s4'], processed_features3['s5']], dim=1)
    processed_features4['s5'] = torch.cat([processed_features4['s4'], processed_features4['s5']], dim=1)
    processed_features5['s5'] = torch.cat([processed_features5['s4'], processed_features5['s5']], dim=1)

    # Set s3 as the new s4
    processed_features1['s4'] = processed_features1['s3']
    processed_features2['s4'] = processed_features2['s3']
    processed_features3['s4'] = processed_features3['s3']
    processed_features4['s4'] = processed_features4['s3']
    processed_features5['s4'] = processed_features5['s3']

    # Remove s3 from the features dictionary
    processed_features1.pop('s3')
    processed_features2.pop('s3')
    processed_features3.pop('s3')
    processed_features4.pop('s3')
    processed_features5.pop('s3')

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat([processed_features1['s4'], F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features2_gether_s4_s5 = torch.cat([processed_features2['s4'], F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features3_gether_s4_s5 = torch.cat([processed_features3['s4'], F.interpolate(processed_features3['s5'], size=(processed_features3['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features4_gether_s4_s5 = torch.cat([processed_features4['s4'], F.interpolate(processed_features4['s5'], size=(processed_features4['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features5_gether_s4_s5 = torch.cat([processed_features5['s4'], F.interpolate(processed_features5['s5'], size=(processed_features5['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features1_gether_s4_s5, features2_gether_s4_s5, features3_gether_s4_s5, features4_gether_s4_s5, features5_gether_s4_s5



def perform_clustering(features, n_clusters=10):
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    # Convert the features to float32
    features = features.cpu().detach().numpy().astype('float32')
    # Initialize a k-means clustering index with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # Train the k-means index with the features
    kmeans.fit(features)
    # Assign the features to their nearest cluster
    labels = kmeans.predict(features)

    return labels

def cluster_and_match(result, save_path, n_clusters=6):
    for idx, (feature1,feature2,feature3,feature4,feature5,mask1,mask2,mask3,mask4,mask5) in enumerate(result):
        # feature1 shape (1,1,3600,768*2)
        num_patches = int(math.sqrt(feature1.shape[-2]))
        # pca the concatenated feature to 3 dimensions
        feature1 = feature1.squeeze() # shape (3600,768*2)
        feature2 = feature2.squeeze() # shape (3600,768*2)
        feature3 = feature3.squeeze() # shape (3600,768*2)
        feature4 = feature4.squeeze() # shape (3600,768*2)
        feature5 = feature5.squeeze() # shape (3600,768*2)
        # resize back

        src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature2_reshaped = feature2.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature3_reshaped = feature3.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature4_reshaped = feature4.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()
        feature5_reshaped = feature5.squeeze().permute(1,0).reshape(-1,num_patches,num_patches).cuda()

        resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_2 = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_3 = F.interpolate(mask3.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_4 = F.interpolate(mask4.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()
        resized_mask_5 = F.interpolate(mask5.unsqueeze(0).unsqueeze(0), size=(num_patches, num_patches), mode='nearest').squeeze().cuda()

        src_feature_upsampled = src_feature_reshaped * resized_src_mask.repeat(src_feature_reshaped.shape[0],1,1)
        feature2_upsampled = feature2_reshaped * resized_mask_2.repeat(src_feature_reshaped.shape[0],1,1)
        feature3_upsampled = feature3_reshaped * resized_mask_3.repeat(src_feature_reshaped.shape[0],1,1)
        feature4_upsampled = feature4_reshaped * resized_mask_4.repeat(src_feature_reshaped.shape[0],1,1)
        feature5_upsampled = feature5_reshaped * resized_mask_5.repeat(src_feature_reshaped.shape[0],1,1)

        feature1=src_feature_upsampled.unsqueeze(0)
        feature2=feature2_upsampled.unsqueeze(0)
        feature3=feature3_upsampled.unsqueeze(0)
        feature4=feature4_upsampled.unsqueeze(0)
        feature5=feature5_upsampled.unsqueeze(0)
        
        w1, h1 = feature1.shape[2], feature1.shape[3]
        w2, h2 = feature2.shape[2], feature2.shape[3]
        w3, h3 = feature3.shape[2], feature3.shape[3]
        w4, h4 = feature4.shape[2], feature4.shape[3]
        w5, h5 = feature5.shape[2], feature5.shape[3]

        features1_2d = feature1.reshape(feature1.shape[1], -1).permute(1, 0)
        features2_2d = feature2.reshape(feature2.shape[1], -1).permute(1, 0)
        features3_2d = feature3.reshape(feature3.shape[1], -1).permute(1, 0)
        features4_2d = feature4.reshape(feature4.shape[1], -1).permute(1, 0)
        features5_2d = feature5.reshape(feature5.shape[1], -1).permute(1, 0)

        labels_img1 = perform_clustering(features1_2d, n_clusters)
        labels_img2 = perform_clustering(features2_2d, n_clusters)
        labels_img3 = perform_clustering(features3_2d, n_clusters)
        labels_img4 = perform_clustering(features4_2d, n_clusters)
        labels_img5 = perform_clustering(features5_2d, n_clusters)

        cluster_means_img1 = [features1_2d.cpu().detach().numpy()[labels_img1 == i].mean(axis=0) for i in range(n_clusters)]
        cluster_means_img2 = [features2_2d.cpu().detach().numpy()[labels_img2 == i].mean(axis=0) for i in range(n_clusters)]
        cluster_means_img3 = [features3_2d.cpu().detach().numpy()[labels_img3 == i].mean(axis=0) for i in range(n_clusters)]
        cluster_means_img4 = [features4_2d.cpu().detach().numpy()[labels_img4 == i].mean(axis=0) for i in range(n_clusters)]
        cluster_means_img5 = [features5_2d.cpu().detach().numpy()[labels_img5 == i].mean(axis=0) for i in range(n_clusters)]

        distances_2 = np.linalg.norm(np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img2, axis=0), axis=-1)
        row_ind_2, col_ind_2 = linear_sum_assignment(distances_2) # Use Hungarian algorithm to find the optimal bijective mapping
        distances_3 = np.linalg.norm(np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img3, axis=0), axis=-1)
        row_ind_3, col_ind_3 = linear_sum_assignment(distances_3) # Use Hungarian algorithm to find the optimal bijective mapping
        distances_4 = np.linalg.norm(np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img4, axis=0), axis=-1)
        row_ind_4, col_ind_4 = linear_sum_assignment(distances_4) # Use Hungarian algorithm to find the optimal bijective mapping
        distances_5 = np.linalg.norm(np.expand_dims(cluster_means_img1, axis=1) - np.expand_dims(cluster_means_img5, axis=0), axis=-1)
        row_ind_5, col_ind_5 = linear_sum_assignment(distances_5) # Use Hungarian algorithm to find the optimal bijective mapping

        relabeled_img2 = np.zeros_like(labels_img2)
        relabeled_img3 = np.zeros_like(labels_img3)
        relabeled_img4 = np.zeros_like(labels_img4)
        relabeled_img5 = np.zeros_like(labels_img5)

        for i, match in zip(row_ind_2, col_ind_2):
            relabeled_img2[labels_img2 == match] = i

        for i, match in zip(row_ind_3, col_ind_3):
            relabeled_img3[labels_img3 == match] = i
        
        for i, match in zip(row_ind_4, col_ind_4):
            relabeled_img4[labels_img4 == match] = i
        
        for i, match in zip(row_ind_5, col_ind_5):
            relabeled_img5[labels_img5 == match] = i

        labels_img1 = labels_img1.reshape(w1, h1)
        relabeled_img2 = relabeled_img2.reshape(w2, h2)
        relabeled_img3 = relabeled_img3.reshape(w3, h3)
        relabeled_img4 = relabeled_img4.reshape(w4, h4)
        relabeled_img5 = relabeled_img5.reshape(w5, h5)

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        # Plot the results
        ax_img1 = axs[0]
        axs[0].axis('off')
        ax_img1.imshow(labels_img1, cmap='tab20')
        
        ax_img2 = axs[1]
        axs[1].axis('off')
        ax_img2.imshow(relabeled_img2, cmap='tab20')

        ax_img3 = axs[2]
        axs[2].axis('off')
        ax_img3.imshow(relabeled_img3, cmap='tab20')

        ax_img4 = axs[3]
        axs[3].axis('off')
        ax_img4.imshow(relabeled_img4, cmap='tab20')

        ax_img5 = axs[4]
        axs[4].axis('off')
        ax_img5.imshow(relabeled_img5, cmap='tab20')

        plt.tight_layout()
        plt.show()
        fig.savefig(save_path+f'/clustering_{idx}.png', dpi=300)

def process_images(src_img_path,trg_img_path):
    category = src_img_path.split('/')[2]
    categories = [[category], [category]]
    files = []
    files.append(src_img_path)
    for target_img in trg_img_path[:4]:
        files.append(target_img)
    save_path = f'./results_vis_all_views_bad/{category}'
    result = compute_pair_feature(model, aug, save_path, files, mask=MASK, category=categories, dist=DIST)
    if MASK:
        vis_pca_mask(result, save_path)
        cluster_and_match(result, save_path)
    if 'Anno' not in src_img_path:
        vis_pca(result, save_path,*files)

    return result

trg_img_path = []
base_path = "sd_dino/data/megapose/motorbike"
for image_name in sorted(os.listdir(base_path)):
    image_path = os.path.join(base_path, image_name)
    if "rgb" in image_name:
        src_img_path = image_path
    else:
        trg_img_path.append(image_path)
print(src_img_path,trg_img_path)
result = process_images(src_img_path, trg_img_path)