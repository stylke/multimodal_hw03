"""
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * ImageCaptionDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
"""

import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer


class ImageCaptionDataset(Dataset):
    # TODO: 需要实现一个 ImageCaptionDataset
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (str): Path to the pickle file containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

        print(f"dataset len: {len(self.data['image_embeddings'])}")

    def __len__(self):
        return len(self.data['image_embeddings'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data['image_names'][idx]
        img_emb = self.data['image_embeddings'][idx]
        img_cap = self.data['image_captions'][idx]

        sample = {'image_name': img_name, 'image_embedding': img_emb, 'image_caption': img_cap}

        if self.transform:
            sample = self.transform(sample)

        return sample



def cl_fn(batch, tokenizer):
    # TODO: 需要实现一个 collate function
    """
    Collate function to process a batch of data and return tensors.

    Args:
        batch (list): List of samples from the dataset.
        tokenizer (GPT2Tokenizer): Tokenizer to process the captions.

    Returns:
        Tuple of tensors: (img_emb, input_ids, attention_mask)
    """
    img_embs = torch.tensor(np.array([item['image_embedding'] for item in batch]))
    img_caps = [item['image_caption'] for item in batch]

    # Tokenize the captions
    encoding = tokenizer(img_caps, return_tensors='pt', padding=True, truncation=True)

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return img_embs, input_ids, attention_mask


def get_loader(dataset, bs_exp=5, shuffle=True, num_workers=0, pin_memory=False):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    return DataLoader(
        dataset,
        batch_size=2**bs_exp,
        collate_fn=lambda b: cl_fn(b, tokenizer),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
