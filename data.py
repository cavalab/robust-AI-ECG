import torch
import h5py
import pandas as pd
import numpy as np
from operator import itemgetter

class H5Dataset(torch.utils.data.Dataset):
    """
    see https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
    """

    def __init__(self, path, preprocessing=None):
        self.file_path = path
        self.dataset = None
        self.preprocessing = preprocessing
        with h5py.File(self.file_path, "r") as file:
            self.dataset_len = len(file["tracings"])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")["tracings"]
        sample = self.dataset[index]
        return sample if self.preprocessing is None else self.preprocessing(sample)

    def __len__(self):
        return self.dataset_len


class H5DatasetMap(torch.utils.data.Dataset):
    """
    Read ECG data corresponding to a file specifying the records to sample.

    ecg_path: str
        A path to an `h5` file. The keys of the file should be ECG IDs, with corresponding ECG tracing values.
    sample_path: str
        A path to a text file containing one ECG ID per line.
        The ECGs in this file are the only ones that will be used in training.
    """

    def __init__(self, ecg_path, sample_path, preprocessing=None):
        self.ecg_path = ecg_path
        self.sample_path = sample_path
        self.dataset = None
        self.preprocessing = preprocessing
        with open(sample_path, "r") as file:
            self.sample_ids = {i: r.strip() for i, r in enumerate(file.readlines())}
            self.dataset_len = len(self.sample_ids)

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")
        sample = self.dataset[self.sample_ids[index]]
        return sample if self.preprocessing is None else self.preprocessing(sample)

    def __len__(self):
        return self.dataset_len


diagnosis_columns = ['tof','cardiomyopathy','asd','cavc','coa','dorv','dtga','ebstein','hlhs','ltga','pa','tapvr','triatresia', 'truncus','vsd','dextrocardia','pacemaker']

class H5LabelledDataset(torch.utils.data.Dataset):
    """
    Load an ECG dataset with labels. Meant for supervised tasks.

    ecg_path: str
        A path to an `h5` file. The keys of the file should be ECG IDs, with corresponding ECG tracing values.
    label_path: str
        A path to a csv file containing ECG IDs and their corresponding labels.
        The label/ECG_ID pairs in this file are the only ones that will be used in training.
    labels: list[str] or None
        Optionally specify which columns of `label_path` to use for training. By default all of
        the ECG-related labels will be used.

    """

    def __init__(self, ecg_path, label_path, labels=None, train_group=None,covariate_path=covariate_path_train):
        self.df_labels = pd.read_csv(label_path).set_index("ECG_ID")
        if labels is None:
            self.labels = [c for c in self.df_labels.columns if c not in ["ECG_ID"]]
        else:
            self.labels = labels
        print("training labels:", self.labels)
        self.ecg_path = ecg_path
        self.ecg_dataset = None
        self.n_labels = len(self.labels)

        for t in task:
            if t in label_path:
                self.covs = pd.read_csv(covariate_path, index_col=ecg_id[t])
                self.covs.index = self.covs.index.astype(str)
                self.covs = self.covs.rename(columns={age_name[t]:'age'})
                self.covs.index.name = 'ECG_ID'
        '''filter by lesion '''
        if train_group:
            self.covs_unique = self.covs[diagnosis_columns].reset_index().drop_duplicates(subset='ECG_ID').set_index('ECG_ID')
            merged_df = self.df_labels.merge(self.covs_unique[diagnosis_columns], left_index=True, right_index=True,how='left')
            self.df_labels = merged_df[merged_df[train_group]==1]
        self.dataset_len = len(self.df_labels)
        
    def __getitem__(self, index):
        labels = self.df_labels.iloc[index][self.labels].astype(np.float32).values  
        if self.ecg_dataset is None:
            self.ecg_dataset = h5py.File(self.ecg_path, "r")
        sample = self.ecg_dataset[self.df_labels.iloc[index].name][:]
        return sample, labels

    def __len__(self):
        return self.dataset_len