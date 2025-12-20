# split_dataset.py

import random

def split_indices(n, train_frac=0.7, val_frac=0.15, seed=0):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

