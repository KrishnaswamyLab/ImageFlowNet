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

### Training the main network.
```
cd src/
# Time-conditional UNet (baseline)
python train_2pt_all.py --model T_UNet --random-seed 1 --mode train
python train_2pt_all.py --model T_UNet --random-seed 1 --mode test --run-count 1

# Schrodinger Bridge
python train_2pt_all.py --model I2SBUNet --random-seed 1
python train_2pt_all.py --model I2SBUNet --random-seed 1 --mode test --run-count 1

# ODE-UNet
python train_2pt_all.py --model StaticODEUNet --random-seed 1
python train_2pt_all.py --model StaticODEUNet --random-seed 1 --mode test --run-count 1
```

### Additional configurations to try.
1. Gradient field formulation.
```
python train_2pt_all.py --model ODEUNet
python train_2pt_all.py --model StaticODEUNet
```

2. Which latent representations for ODE?
```
python train_2pt_all.py --model StaticODEUNet --ode-location 'bottleneck'
python train_2pt_all.py --model StaticODEUNet --ode-location 'all_resolutions'
python train_2pt_all.py --model StaticODEUNet --ode-location 'all_connections' # default
```

3. Latent feature regularization.
```
python train_2pt_all.py --model StaticODEUNet --coeff-latent 0.1
```

4. Contrastive learning regularization.
```
python train_2pt_all.py --model StaticODEUNet --coeff-contrastive 0.1
```

5. Trajectory smoothness regularization.
```
python train_2pt_all.py --model StaticODEUNet --coeff-smoothness 0.1
```

### Comparisons
Image interpolation/extrapolation methods.
```
cd comparison/interpolation
python run_baseline_interp.py --method linear
python run_baseline_interp.py --method cubic_spline
```

Style-based Manifold Extrapolation (Nat. Mach. Int. 2022).
```
conda deactivate
conda activate stylegan

cd src/preprocessing
python 04_unpack_retina_UCSF.py

cd ../../comparison/style_manifold_extrapolation/stylegan2-ada-pytorch
python train.py --outdir=../training-runs --data='../../../data/retina_ucsf/UCSF_images_final_unpacked_256x256/' --gpus=1
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
python -m pip install torchsde
python -m pip install phate
python -m pip install psutil
python -m pip install ninja
```

# Environment for stylegan2-ada
```
conda create --name stylegan python=3.8 -c anaconda
conda activate stylegan
conda install scikit-learn scikit-image pillow matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge
python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install opencv-python
python -m pip install psutil
python -m pip install ninja
python -m pip install requests
conda install -c conda-forge gcc=12.1.0
```


## Acknowledgements
We adapted some of the code from
1. [I^2SB: Image-to-Image Schrodinger Bridge](https://github.com/NVlabs/I2SB)