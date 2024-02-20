import h5py
import torch
from torch.utils.data import Dataset

# class UniAttackDataset(Dataset):
#     def __init__(self, hdf5_filename, labels_filename, dataset_name):
#         self.hdf5_filename = hdf5_filename
#         self.dataset_name = dataset_name
        
#         # Read the labels from the text file
#         with open(labels_filename, 'r') as file:
#             self.labels = [int(line.strip().split()[-1]) for line in file.readlines()]  # Split line and convert label part
#             print(len(self.labels))

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Open the HDF5 file and read the specified dataset
#         with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
#             features_dataset = hdf5_file[self.dataset_name]
#             feature = torch.tensor(features_dataset[idx], dtype=torch.float32)
        
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return feature, label

import h5py
import torch
from torch.utils.data import Dataset

class UniAttackDataset(Dataset):
    def __init__(self, hdf5_filename, labels_filename=None, dataset_name='train_features'):
        self.hdf5_filename = hdf5_filename
        self.dataset_name = dataset_name
        self.labels_available = labels_filename is not None

        if self.labels_available:
            # Read the labels from the text file, assuming each line is "image_path label"
            with open(labels_filename, 'r') as file:
                self.labels = [int(line.strip().split()[-1]) for line in file.readlines()]
        else:
            self.labels = []

    def __len__(self):
        # If labels are not available, determine length from the HDF5 dataset
        if not self.labels_available:
            with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
                return hdf5_file[self.dataset_name].shape[0]
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
            features_dataset = hdf5_file[self.dataset_name]
            feature = torch.tensor(features_dataset[idx], dtype=torch.float32)
        
        if self.labels_available:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return feature, label
        else:
            return feature

class AuDataset(Dataset):
    def __init__(self, hdf5_filename, labels_filename=None, dataset_name='train_features'):
        self.hdf5_filename = hdf5_filename
        self.dataset_name = dataset_name
        self.labels_available = labels_filename is not None

        if self.labels_available:
            # Modify here to read multi-label data
            with open(labels_filename, 'r') as file:
                # Assuming each line is "image_path label1,label2,...,label12"
                self.labels = [list(map(int, line.strip().split()[-1].split(','))) for line in file.readlines()]
        else:
            self.labels = []

    def __len__(self):
        if not self.labels_available:
            with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
                return hdf5_file[self.dataset_name].shape[0]
        return len(self.labels)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_filename, 'r') as hdf5_file:
            features_dataset = hdf5_file[self.dataset_name]
            feature = torch.tensor(features_dataset[idx], dtype=torch.float32)
        
        if self.labels_available:
            # Labels are now multi-dimensional
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return feature, label
        else:
            return feature

if __name__ == '__main__':
    # Example usage for the training dataset
    train_dataset = AuDataset(
    hdf5_filename='au_train.h5',
    labels_filename='au_train.txt',
    dataset_name='train_features'
)
    val_dataset = AuDataset(
    hdf5_filename='au_val.h5',
    labels_filename='au_val.txt',
    dataset_name='val_features'
)

#     # Calculate the number of labels equal to 1 and 0
    # num_labels_1 = sum(label == 1 for label in train_dataset.labels)
    # num_labels_0 = sum(label == 0 for label in train_dataset.labels)

    # print(f"Number of labels = 1: {num_labels_1}")
    # print(f"Number of labels = 0: {num_labels_0}")
    print(f"Length of train dataset: {len(train_dataset)}")
    pass