from PIL import Image
from utils.utils_correspondence import co_pca, resize, find_nearest_patchs, find_nearest_patchs_replace
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
real_size=960
img_size = 840

img1 = Image.open('/cluster/home/simschla/master_thesis/sd-dino/sd_dino/data/megapose/boat/2.png').convert('RGB')
img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

print("done")