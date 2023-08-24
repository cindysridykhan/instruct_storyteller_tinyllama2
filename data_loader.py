import math
import os
import glob
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs


class PretokDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            max_seq_len, 
            split,
            data_cache_dir,
            seed=42            
            ):
        self.max_seq_len = max_seq_len
        self.split = split
        self.seed = seed
        self.data_cache_dir = data_cache_dir

    def __iter__(self):
        rng = random.Random(self.seed)
        shard_filenames = sorted(glob.glob(os.path.join(self.data_cache_dir, 'data*.bin')))
        shard_filenames = shard_filenames[1:] if self.split == 'train' else shard_filenames[:1]

        labels_shard_filenames = sorted(glob.glob(os.path.join(self.data_cache_dir, 'labels*.bin')))
        labels_shard_filenames = labels_shard_filenames[1:] if self.split == 'train' else labels_shard_filenames[:1]

        
        ixs_shards = list(range(len(shard_filenames)))
        while True:
            rng.shuffle(ixs_shards)
            for i in ixs_shards:
                m = np.memmap(shard_filenames[i], dtype=np.uint16, mode='r')
                labels_m = np.memmap(labels_shard_filenames[i], dtype=np.uint16, mode='r')
                num_batches = len(m) // self.max_seq_len - 1
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1 
                    chunk_x = torch.from_numpy(m[start:end].astype(np.int64))
                    
                    chunk_y = torch.from_numpy(labels_m[start:end].astype(np.int64))
                    
                    x, y = chunk_x[:-1], chunk_y[1:]
                    x[x==65436] = 2 # tokenizer.eos_id()
                    y[y==65436] = -1
                    # padding will be ignored by the loss, as well as prompt in y
                    yield x, y

def iter_batch_func(device, batch_size, **dataset_kwargs):
    ds = PretokDataset(**dataset_kwargs)
    dl = torch.utils.data.DataLoader(
        dataset=ds, 
        batch_size=batch_size,
        pin_memory=True, num_workers=0
    ) 
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yield x, y