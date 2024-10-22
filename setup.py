import sys
sys.path.append('/cluster/home/simschla/master_thesis/sd-dino-env/lib/python3.11/site-packages')
sys.path.append('.')
import os
from setuptools import setup, find_packages
for i in range(10):
    print("python path:", sys.path)
import torch

# '/cluster/home/simschla/master_thesis/sd-dino-env/lib/python3.11/site-packages/torch/__init__.py'


torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

setup(
    name="sd_dino",
    author="Junyi Zhang",
    description="Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence",
    python_requires=">=3.8",
    packages=find_packages(),
    py_modules=[],
    install_requires=[
        "loguru>=0.5.3",
        "faiss-cpu>=1.7.1",
        "matplotlib>=3.4.2",
        "tqdm>=4.61.2",
        "numpy>=1.21.0",
        "gdown>=3.13.0",
        # f"mask2former @ file://localhost/{os.getcwd()}/third_party/Mask2Former/",
        # f"odise @ file://localhost/{os.getcwd()}/third_party/ODISE/"
    ],
    include_package_data=True,
)