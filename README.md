# Robust AI-ECG for Pediatric

This repository contains the implementation for the paper *"[Robust AI-ECG for Predicting Left Ventricular Systolic Dysfunction in Pediatric Congenital Heart Disease](https://arxiv.org/abs/2509.19564)"*, focusing on building robust AI-ECG models to predict LVSD. 

# Installation
- Clone the repository
- (Option A) Install with pip using requirements.txt
    ```
    pip install -r requirements.txt
    ```
- (Option B) Create conda environment using environment.yml
    ```
    conda env create -f environment.yml
    conda activate your_env_name
    ```

# Contents
- `models/`
    - `ViT_MAE_24ch.py`: ViT-MAE model for pre-training 
    - `resnet.py`: ResNet model for classification
- `preprocessing/postprocessing`: code for preprocessing/postprocessing
- `data.py`: dataset class
- `train_model.py`: main training code for single models
- `utils.py`: random utility functions



# Training

Use `train_model.py` to train a model.
```
python train_model.py finetune **args
```
Key arguments include:
- `ecg_path`: path of ECG
- `label_train_file`: path of labels
- `labels`: list of outcomes, e.g., ['less50','less45','less40','less35','less30']
- `covariate_path`: path of dysfunction characteristics
- `subsample`: training data ratio
- `adv_train`: whether use adversarial training
- `train_group`: select groups you want keep for training ("None" for keeping all groups).
- `perturb_level`: 
    - "input": add perturbations directly on input ECG
    - "embedding": add perturbations on embedding space
- `perturb_type`: 
    - "adversarial": generate adversarial perturbations
    - "gaussian": add Gaussian noise

# Citation
If you use this code or find our work helpful, please consider citing our paper:
```
@misc{yang2025robustaiecgpredictingleft,
      title={Robust AI-ECG for Predicting Left Ventricular Systolic Dysfunction in Pediatric Congenital Heart Disease}, 
      author={Yuting Yang and Lorenzo Peracchio and Joshua Mayourian and John K. Triedman and Timothy Miller and William G. La Cava},
      year={2025},
      eprint={2509.19564},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2509.19564}, 
}
```