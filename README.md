# Medical Image Progression
### Krishnaswamy Lab, Yale University
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![Github Stars](https://img.shields.io/github/stars/ChenLiu-1996/MedicalImageProgression.svg?style=social&label=Stars)](https://github.com/ChenLiu-1996/MedicalImageProgression/)



## Repository Hierarchy
```
```

## Data Provided

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name mip pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda activate mip
conda install scikit-image pillow matplotlib seaborn tqdm -c anaconda
python -m pip install -U phate
python -m pip install torchdiffeq
python -m pip install -U roifile[all]
```
Installation usually takes between 20 minutes and 1 hour on a normal desktop computer.

## Usage


## Acknowledgements
ODE modules adapated from https://github.com/EmilienDupont/augmented-neural-odes and https://github.com/DIAGNijmegen/neural-odes-segmentation.
