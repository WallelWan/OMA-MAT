"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        output_batch = dict()

        for key in batch[0]:
            if 'offset' in key:
                output_batch[key] = torch.cumsum(
                        collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                        dim=0)
            elif 'inverse' in key:
                category = key.split("_")[0]

                if category == 'path':
                    friend_key = 'coord'
                elif category == 'grid':
                    friend_key = 'grid_coord'
                else:
                    raise ValueError(f"{category} is not supported")
                
                total_size = 0
                inverse_list = [d[key] for d in batch]
                friend_size_list = [d[friend_key].shape[0] for d in batch]
                output = None
                for inverse, friend_size in zip(inverse_list, friend_size_list):
                    if output is None:
                        output = inverse
                    else:
                        output = torch.cat([output, inverse + total_size], dim=0)
                    
                    total_size += friend_size
                output_batch[key] = output

            else:
                output_batch[key] = collate_fn([d[key] for d in batch])

        return output_batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        if "offset" in batch.keys():
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
