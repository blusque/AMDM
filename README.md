# <p align="center"> Interactive Character Control with Auto-Regressive Motion Diffusion Models </p>
### <p align="center"> [Yi Shi](https://github.com/Yi-Shi94/), [Jingbo Wang](https://wangjingbo1219.github.io/), [Xuekun Jiang](), [Bingkun Lin](), [Bo Dai](https://daibo.info/), [Xue Bin Peng](https://xbpeng.github.io/) </p>
<p align="center">
  <img width="100%" src="assets/images/AMDM_teaser.png"/>
</p>

## Implementation of Auto-regressive Motion Diffusion Model (A-MDM)
We implemented a PyTorch framework for kinematic-based auto-regressive models. Our framework supports training and inference for A-MDM. Additionally, it offers real-time inpainting and reinforcement learning-based interactive control tasks. 

## Dataset Preparation
### LaFAN1:
[Download](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) and extract under ```./data/``` directory.
We didn't include files with a prefix of 'obstacle'

### 100STYLE:
[Download](https://www.ianxmason.com/100style/) and extract under ```./data/``` directory.

### Any other BVH dataset:
Download and extract under ```./data/``` directory. Create a yaml config file in ```./config/model/```, 

### AMASS:
Follow the procedure described in the repo of [HuMoR](https://github.com/davrempe/humor)

### HumanML3D:
Follow the procedure described in the repo of [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) 


### Sanity Check:
Specify your model config file and run
```
python run_sanity_data.py
```

## Base Model
### Training

```
python run_base.py --arg_file args/amdm_DATASET_train.txt
```

### Inference
```
python run_env.py --arg_file args/RP_amdm_DATASET.txt
```

#### Inpainting
```
python run_env.py --arg_file args/PI_amdm_DATASET.txt
```

### High-Level Controller


#### Training
```
python run_env.py --arg_file args/ENV_train_amdm_DATASET.txt
```
#### Inference
```
python run_env.py --arg_file args/ENV_test_amdm_DATASET.txt
```

### Installation
```
conda create -n amdm python=3.7
pip install -r requirement.txt
```

## Acknowledgement
The RL related modules are built using existing code base of [MotionVAE](https://github.com/electronicarts/character-motion-vaes)

## BibTex
```
@article{
        shi2024amdm,
        author = {Shi, Yi and Wang, Jingbo and Jiang, Xuekun and Lin, Bingkun and Dai, Bo and Peng, Xue Bin},
        title = {Interactive Character Control with Auto-Regressive Motion Diffusion Models},
        year = {2024},
        issue_date = {August 2024},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {43},
        journal = {ACM Trans. Graph.},
        month = {jul},
        keywords = {motion synthesis, diffusion model, reinforcement learning}
      }
```
