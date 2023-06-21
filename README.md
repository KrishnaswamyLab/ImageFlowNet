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

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.8 + PyTorch 1.12.1.
How we created the conda environment:
```
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
```
<!-- pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 -->
Installation usually takes between 20 minutes and 1 hour on a normal desktop computer.

## Usage


## Acknowledgements
ODE modules adapated from https://github.com/EmilienDupont/augmented-neural-odes and https://github.com/DIAGNijmegen/neural-odes-segmentation.
