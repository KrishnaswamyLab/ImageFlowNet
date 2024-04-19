# Medical Image Progression
### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/MedicalImageProgression.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/MedicalImageProgression/)

## Goal
We want to predict the progression of diseases by interpolating or extrapolating the medical images at different time points.

## Repository Hierarchy
```
```

## Data Provided
Retinal dataset from UCSF.

## Setup
1. Create the environment following instructions in **Dependencies**.
2. Download a pre-trained segmentation model (for data preprocessing purposes).
```
cd `external_src/SAM/`
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

3. Data preprocessing
```
cd src/preprocessing
python 01_preprocess_retina_UCSF.py
python 02_register_retina_UCSF.py
python 03_crop_retina_UCSF.py
```

## Reproduce the results

### Image registration
```
cd src/preprocessing
python test_registration.py
```

### Training a segmentation network (only for quantitative evaluation purposes)
```
cd src/
python train_segmentor.py --mode train --config ../config/segment_retinaUCSF_seed1.yaml
```

### Training the main network (matching between 2 timepoints).
```
cd src/
# Time-conditional UNet (baseline)
python train_2pt_all.py --mode train --config ../config/retinaUCSF_T-UNet_seed1.yaml

# Schrodinger Bridge
python train_2pt_all.py --mode train --config ../config/retinaUCSF_I2SB_seed1.yaml

# ODE-UNet
python train_2pt_all.py --mode train --config ../config/retinaUCSF_ODEUNet_seed1.yaml
```

### Training the main network (matching between multiple timepoints).
```
cd src/
# Schrodinger Bridge
?

# CDE-UNet
python train_npt_cde.py --mode train --config ../config/retinaUCSF_CDEUNet_seed1.yaml
```


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
python -m pip install torchcde
```


## Acknowledgements
We adapted some of the code from
1. [I^2SB: Image-to-Image Schrodinger Bridge](https://github.com/NVlabs/I2SB)