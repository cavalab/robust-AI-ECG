import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
# from lightning.pytorch import seed_everything


import os
import importlib
import json

################################################################################
# save printouts to log file
import logging as logging

# from sklearn.metrics import mean_squared_error as mse, r2_score
import time
import uuid
from datetime import datetime

# import torch, torch.nn as nn, torch.utils.data as data
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

from utils import initializer
from data import H5Dataset, H5LabelledDataset
from functools import partial


"""
You are using a CUDA device ('NVIDIA L40') that has Tensor Cores. 
To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. 
For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
"""
torch.set_float32_matmul_precision("medium")


class Trainer:
    @initializer  # this decorator automatically sets arguments to class attributes.
    def __init__(
        self,
        ecg_path="data/All_ECGs/",  # all_ECGs.h5',
        ecg_file=None,  # defaults to all_ECGs_float32.h5
        save_dir="results_test/",
        exp_name='',
        log_dir='',
        seed=42,
        nn=None,  # options: "ViT_MAE_24ch" (auto-encoder)
        initial_model=None,
        gpu=0,
        # Data
        split=0.8,
        # L.Trainer arguments
        max_epochs=150,
        max_time="02:00:00:00",  # days:hrs:mins:secs
        batch_size=128,
        # Early stopping
        patience=20,
        # Model arguments
        loss="mse",
        lr=1e-3, 
        min_lr=1e-7,
        # weighted=False,
        scale=True,
        signal_prep=None,  # "spectrogram",  # 'fft', 'spectogram'
        data_preprocess=None,  # "spectrogram_preprocessing",
        subsample= None, # random sample part of training set
    ):
        self.time = time.time()
        self.datetime = str(datetime.now()).replace(" ", "_")

        # str_sample_rate = (f'{sample_rate:.2f}').replace('.','')
        if ecg_file is None:
            if signal_prep == "fft":
                self.data_file = f"{ecg_path}/all_ECGs_fft.h5"
            elif signal_prep == "spectrogram":
                self.data_file = f"{ecg_path}/all_ECGs_spectrogram_clip.h5"
            else:
                self.data_file = f"{ecg_path}/all_ECGs_float32_T.h5"
        else:
            self.data_file = f"{ecg_path}/{ecg_file}"

        assert os.path.exists(self.data_file), f"cannot find {self.data_file}"

        L.seed_everything(self.seed, workers=True)

    def save(self):
        """Save parameters of run to a json file."""
        save_name = f"{self.save_dir}/{self.exp_name}/train_model_log.json"
        with open(save_name, "w") as of:
            payload = {
                k: v
                for k, v in vars(self).items()
                if any(isinstance(v, t) for t in [bool, int, float, str, dict, list])
            }
            print("payload:", json.dumps(payload, indent=2))
            json.dump(payload, of, indent=4)

    @initializer
    def finetune(
        self,
        label_train_file=None,
        label_test_file=None,
        labels=None,
        checkpoint=None, # encoder checkpoint
        clf="resnet", # classification model
        adv_train=False, # whether use adversarial training
        train_group=None, # which groups are kept for training, if "None", use all groups
        perturb_level=None, # add perturbations on input/embedding
        perturb_type=None, # adversarial perturbations / Gaussian noise
        **model_kwargs,
    ):
        """Train a classification model (resnet)"""

        self.model_kwargs = model_kwargs
        if not label_test_file:
            label_test_file = label_train_file.replace("training", "test").replace(
                "train", "test"
            )
        # logging
        os.makedirs(self.save_dir, exist_ok=True)

        # get algorithm
        print(f"import from methods.{self.nn}")
        algorithm = importlib.__import__("models." + clf, globals(), locals(), ["*"])

        ########################################################################
        # define data
        ########################################################################
        print("loading data")
        train_dataset = H5LabelledDataset(
            ecg_path=self.data_file, label_path=label_train_file, labels=labels, train_group=train_group, covariate_path=covariate_path
        )
        n_labels = train_dataset.n_labels
        if labels is None:
            labels = train_dataset.labels
        test = H5LabelledDataset(
            ecg_path=self.data_file, label_path=label_test_file,labels=labels
        )


        self.input_shape = train_dataset[0][0].shape
        print("input shape:", self.input_shape)

        if self.subsample:
            # random sample part of training set
            indices = torch.randperm(len(train_dataset))[:int(self.subsample*len(train_dataset))].tolist()
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        val_size = 1 - self.split 
        train_size = 1 - val_size
        print("train dataset size:", len(train_dataset))
        print("test dataset size:", len(test))
        print(f"train/val:{train_size:.2f}/{val_size:.2f}")

        train, val = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]  
        )
        data_loader_kwargs = dict(
            batch_size=self.batch_size, num_workers=2, pin_memory=True
        )
        ########################################################################
        # make model
        ########################################################################
        print("construct model")
        model = algorithm.Model(
            input_shape=self.input_shape,
            n_labels=n_labels,
            lr=self.lr,
            min_lr=self.min_lr,
            labels=labels,
            adv_train=adv_train,
            perturb_level=perturb_level,
            perturb_type=perturb_type,
            **self.model_kwargs,
        )
        
        if adv_train:
            model_path = 'path/of/original/model/checkpoint'
            ckpt_name = [f for f in os.listdir(model_path) if 'best' in f][0]
            model.load_state_dict(torch.load(model_path+ckpt_name)['state_dict'], strict=False) 

        summary(
            model,
            input_size=(self.batch_size, self.input_shape[0], self.input_shape[1]),
            depth=3,
        )
        ########################################################################
        # train model
        ########################################################################
        print("train")
        log_args = dict(
            save_dir=self.log_dir,
            name=self.exp_name,
            version="" 
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.save_dir,self.exp_name),
            monitor="val/auroc/macro",   
            mode="max",                  
            save_top_k=1,   
            save_last=True,              
            save_weights_only=False,     
            filename="best-{epoch:02d}",  
            verbose=True
        )
        early_stop_callback = EarlyStopping(
            monitor="val/auroc/macro",
            mode="max",
            patience=self.patience,
            min_delta=0,
        )

        csv_logger = CSVLogger(**log_args)
        tb_logger = TensorBoardLogger(**log_args)
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            max_time=self.max_time,
            logger=[csv_logger, tb_logger],
            deterministic=False,  # WGL: this makes cumsum faster
            log_every_n_steps=1, # 10
            callbacks=[checkpoint_callback,early_stop_callback],
        )
        trainer.fit(
            model,
            torch.utils.data.DataLoader(train, **data_loader_kwargs),
            torch.utils.data.DataLoader(val, **data_loader_kwargs),
        )

        ########################################################################
        # Evaluate model on test holdout
        # test_metrics = trainer.test(
        #     model,
        #     dataloaders=torch.utils.data.DataLoader(test, **data_loader_kwargs),
        #     ckpt_path="best",
        # )
        # print("test_metrics:", json.dumps(test_metrics, indent=2))
        ###############################################################################
        # save results
        self.save()


import fire

if __name__ == "__main__":
    fire.Fire(Trainer)