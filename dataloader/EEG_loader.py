import os
import sys
#set the path to the EEG folder
sys.path.append('/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/JEPA/dataloader')
import subprocess
import time
import dataset as ds
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from logging import getLogger
import dataset
import torch
import torchvision
from dataloader.config import Config
_GLOBAL_SEED = 0
logger = getLogger()
from torch.utils.data import DataLoader


#we weill need to return 

# 1. A dataset object
# 2. A dataloader object
# 3. A distributed sampler object

def load_eeg_data(participant_ids,
                  batch_size=32,
                  collator=None,
                  num_workers=4,
                  pin_mem=True):
    """
    Load EEG data into a DataLoader for training.

    Args:
        participant_ids (list): List of participant IDs to include.
        batch_size (int): Number of samples per batch.
        collator (callable, optional): Function to merge a list of samples into a mini-batch.
        num_workers (int): How many subprocesses to use for data loading.
        pin_mem (bool): If True, the data loader will copy Tensors into CUDA pinned memory.

    Returns:
        tuple: Contains the dataset and the DataLoader.
    """
    path_indices_to_use = [1, 2, 3, 6, 8, 11, 12, 13]
    paths_EEG = []

    for i in range(1, 51):
        if i - 1 not in path_indices_to_use:
            continue
        #root = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
        root = "/home/hutchins/projects/def-yalda/hutchins/data/sub-"
        mid = "/eeg/sub-"
        end = "_task-rsvp_eeg.vhdr"
        
        if i < 10:
            path = root + "0" + str(i) + mid + "0" + str(i) + end
        else:
            path = root + str(i) + mid + str(i) + end

        paths_EEG.append(path)
    
    datasets = []
    for path in paths_EEG:
        dataset = ds.create_dataset(raw_file_path=path, 
                                         event_description='Event/E  1', 
                                         batch_size=batch_size)
        datasets.append(dataset)

    # Combine all individual datasets into a single ConcatDataset
    combined_dataset = ConcatDataset(datasets)
    
    config = Config()

    
    
    # Use a simple DataLoader without a sampler for non-distributed training
    dataloader_eeg = DataLoader(combined_dataset,
                                collate_fn=collator,
                                batch_size=batch_size,
                                shuffle=True,  # Assuming you want to shuffle for training
                                num_workers=num_workers,
                                pin_memory=pin_mem)
    
    return combined_dataset, dataloader_eeg