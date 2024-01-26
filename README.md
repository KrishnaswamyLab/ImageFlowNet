# Medical Image Progression
### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/MedicalImageProgression.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/MedicalImageProgression/)

## Goal
We want to predict the progression of diseases by interpolating or extrapolating the medical images at different time points.

## Repository Hierarchy
```
```

## Under `external_src/SAM/`
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

## Data Provided

## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name mip pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c nvidia -c anaconda -c conda-forge
conda activate mip
conda install scikit-learn scikit-image pillow matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge
conda install read-roi -c conda-forge
python -m pip install -U albumentations
python -m pip install timm
python -m pip install opencv-python
python -m pip install git+https://github.com/facebookresearch/segment-anything.git
python -m pip install monai
python -m pip install torchdiffeq
python -m pip install torch-ema
<!-- conda install -c conda-forge gcc=12.1.0  # If you see version `GLIBCXX_3.4.30' not found. -->
```

## Usage
1. Experiments on synthetic data.
First, we have a strong intuition that:
    - ODE-AE will work if disease progression is the only variable in the image, while in non-disease regions we have pixel-perfect spatial alignment.
    - ODE-AE will work if, even though we don't have pixel-perfect alignment in non-disease regions, the spatial transformation is well-defined and only depend on time.
    - ODE-AE will NOT work if, the spatial transformation is random and chaotic.

To verify this, we train ODE-AE on 3 synthetic datasets.
```
1. `./data/synthesized/base/`: non-disease regions are not moving.
2. `./data/synthesized/rotation/`: non-disease regions are rotated, and the rotation only depends on time.
3. `./data/synthesized/mixing/`: non-disease regions are transformed, and the transformation is chaotic.
```


<!-- ```
conda create --name mip python==3.8
conda activate mip
conda install -c conda-forge torchdiffeq
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda
python -m pip install opencv-python
python -m pip install -U roifile[all]
python -m pip install click
python -m pip install psutil
python -m pip install tensorboard
python -m pip install pytorch-ssim

# StyleGAN2
conda install pytorch-gpu==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python3 -m pip install setuptools==59.5.0
python3 -m pip install ninja


# 1. Use requirements
conda env create --file requirements.yaml
# 2. Additional installation
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
python -m pip install opencv-python
```

## DEBUG:
1. OSError: xxxx/libcublas.so.11: undefined symbol: cublasLtBIIMatmulAlgoGetHeuristic, version libcublasLt.so.11
```
# In this case, you may need to add the correct location of `libcublasLt.so.11` into the environment variable `$LD_LIBRARY_PATH`.
# For me, this means:
export LD_LIBRARY_PATH=/PATH_TO_MY_CONDA_ENV/.conda_envs/mip-i2sb/lib:$LD_LIBRARY_PATH
``` -->

<!-- pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -->

<!-- ## Usage
```
python train.py --cond-x1 --log-writer wandb --wandb-api-key a8e550aa2ec6835c2890425e63b466a8b46c01ab --wandb-user cl2482
```


## Acknowledgements
Codebase heavily adapted from [I^2SB: Image-to-Image Schrodinger Bridge](https://github.com/NVlabs/I2SB) -->