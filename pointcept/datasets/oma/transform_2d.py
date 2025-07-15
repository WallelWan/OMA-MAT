import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

from ..transform import *

@TRANSFORMS.register_module()
class CenterShift2D(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):

        total_x_min, total_y_min, total_x_max, total_y_max = None, None, None, None
        for key in data_dict.keys():
            if "point" in key:
                x_min, y_min = data_dict[key].min(axis=0)
                x_max, y_max = data_dict[key].max(axis=0)

                if total_x_min is None:
                    total_x_min = x_min
                    total_y_min = y_min
                    total_x_max = x_max
                    total_y_max = y_max
                else:
                    total_x_min = min(total_x_min, x_min)
                    total_y_min = min(total_y_min, y_min)
                    total_x_max = max(total_x_max, x_max)
                    total_y_max = max(total_y_max, y_max)

        shift = [(total_x_min + total_x_max) / 2, (total_y_min + total_y_max) / 2]
        for key in data_dict.keys():
            if "point" in key:
                data_dict[key] -= shift
        return data_dict
    
@TRANSFORMS.register_module()
class RandomShift2D(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2))):
        self.shift = shift

    def __call__(self, data_dict):
        shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
        shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])

        for key in data_dict.keys():
            if "point" in key:
                data_dict[key] += [shift_x, shift_y]
        return data_dict
    
# @TRANSFORMS.register_module()
# class RandomDropout2D(object):
#     def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
#         """
#         upright_axis: axis index among x,y,z, i.e. 2 for z
#         """
#         self.dropout_ratio = dropout_ratio
#         self.dropout_application_ratio = dropout_application_ratio

#     def __call__(self, data_dict):
#         for key in data_dict.keys():
#             if "coord" in key:
#                 if random.random() < self.dropout_application_ratio:
#                     n = len(data_dict[key])
#                     idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
#                     # TODO: sample the id of key

#                     data_dict = index_operator(data_dict, idx)
#         return data_dict
    
@TRANSFORMS.register_module()
class RandomRotate2D(object):
    def __init__(self, angle=None, center=None, always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_t = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]])

        if self.center is None:
            total_x_min, total_y_min, total_x_max, total_y_max = None, None, None, None
            for key in data_dict.keys():
                if "point" in key:
                    x_min, y_min = data_dict[key].min(axis=0)
                    x_max, y_max = data_dict[key].max(axis=0)

                    if total_x_min is None:
                        total_x_min = x_min
                        total_y_min = y_min
                        total_x_max = x_max
                        total_y_max = y_max
                    else:
                        total_x_min = min(total_x_min, x_min)
                        total_y_min = min(total_y_min, y_min)
                        total_x_max = max(total_x_max, x_max)
                        total_y_max = max(total_y_max, y_max)
            center = [(total_x_min + total_x_max) / 2, (total_y_min + total_y_max) / 2]

        else:
            center = self.center

        for key in data_dict.keys():
            if "point" in key:
                data_dict[key] -= center
                data_dict[key] = np.dot(data_dict[key], np.transpose(rot_t))
                data_dict[key] += center

        return data_dict
    
@TRANSFORMS.register_module()
class RandomRotateTargetAngle2D(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, always_apply=False, p=0.75
    ):
        self.angle = angle
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_t = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]])

        if self.center is None:
            total_x_min, total_y_min, total_x_max, total_y_max = None, None, None, None
            for key in data_dict.keys():
                if "point" in key:
                    x_min, y_min = data_dict[key].min(axis=0)
                    x_max, y_max = data_dict[key].max(axis=0)

                    if total_x_min is None:
                        total_x_min = x_min
                        total_y_min = y_min
                        total_x_max = x_max
                        total_y_max = y_max
                    else:
                        total_x_min = min(total_x_min, x_min)
                        total_y_min = min(total_y_min, y_min)
                        total_x_max = max(total_x_max, x_max)
                        total_y_max = max(total_y_max, y_max)
            center = [(total_x_min + total_x_max) / 2, (total_y_min + total_y_max) / 2]

        else:
            center = self.center

        for key in data_dict.keys():
            if "point" in key:
                data_dict[key] -= center
                data_dict[key] = np.dot(data_dict[key], np.transpose(rot_t))
                data_dict[key] += center
        return data_dict

@TRANSFORMS.register_module()
class RandomFlip2D(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            for key in data_dict.keys():
                if "point" in key:
                    data_dict[key][:, 0] = -data_dict[key][:, 0]
        if np.random.rand() < self.p:
            for key in data_dict.keys():
                if "point" in key:
                    data_dict[key][:, 1] = -data_dict[key][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter2D(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        for key in data_dict.keys():
            if "point" in key:
                jitter = np.clip(
                    self.sigma * np.random.randn(data_dict[key].shape[0], 2),
                    -self.clip,
                    self.clip,
                )
                data_dict[key] += jitter
        return data_dict

@TRANSFORMS.register_module()
class RandomScale2D(object):
    def __init__(self, scale=None):
        self.scale = scale if scale is not None else [0.95, 1.05]

    def __call__(self, data_dict):
        scale = np.random.uniform(
                self.scale[0], self.scale[1], 2
            )
        for key in data_dict.keys():
            if "point" in key:
                data_dict[key] *= scale
        return data_dict

@TRANSFORMS.register_module()
class ClipGaussianJitter2D(object):
    def __init__(self, scalar=0.02):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96

    def __call__(self, data_dict):
        for key in data_dict.keys():
            if "point" in key:
                jitter = np.random.multivariate_normal(
                    self.mean, self.cov, data_dict[key].shape[0]
                )
                jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
                data_dict[key] += jitter
        return data_dict

@TRANSFORMS.register_module()
class PointToVector(object):
    def __init__(self):
        pass

    def safe_arctan2(self, vector, epsilon=1e-12):
        x = vector[..., 0]
        y = vector[..., 1]
        
        # 判断是否接近零向量（避免浮点误差）
        is_near_zero = (np.abs(x) < epsilon) & (np.abs(y) < epsilon)
        
        # 核心优化：仅对非零向量计算角度
        angle = np.where(
            is_near_zero,
            0.0,  # 对接近零的向量直接赋0
            np.arctan2(y, x)  # 仅对非零向量计算
        )
        return angle[:, None]

    def __call__(self, data_dict):
        update_dict = dict()
        for key in data_dict.keys():
            if "point" in key:
                category = key.split("_")[0]
                N = data_dict[category + "_id"].shape[0]
                update_dict[category + "_coord"] = data_dict[key].reshape(N, 2, -1)[:, 0] + data_dict[key].reshape(N, 2, -1)[:, 1]
                update_dict[category + "_vector"] = data_dict[key].reshape(N, 2, -1)[:, 1] - data_dict[key].reshape(N, 2, -1)[:, 0]
                update_dict[category + "_angle"] = self.safe_arctan2(update_dict[category + "_vector"]) # [-pi, pi]
                
        data_dict.update(update_dict)

        return data_dict

@TRANSFORMS.register_module()
class GridSample2D(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=True,
        return_grid_coord=True,
        return_min_coord=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):

        add_dict = dict()
        for key in data_dict.keys():
            if "point" in key:
                category = key.split("_")[0]

                assert category+"_coord" in data_dict.keys()
                assert category+"_vector" in data_dict.keys()
                assert category+"_angle" in data_dict.keys()

                coord = np.concatenate([data_dict[category+"_coord"], data_dict[category+"_angle"]], axis=-1)

                scaled_coord = coord / np.array(self.grid_size)
                grid_coord = np.floor(scaled_coord).astype(int)
                min_coord = grid_coord.min(0)
                grid_coord -= min_coord
                scaled_coord -= min_coord
                min_coord = min_coord * np.array(self.grid_size)
                key = self.hash(grid_coord)
                idx_sort = np.argsort(key)
    
                # 构建双射映射
                unique_grid, inverse, counts = np.unique(
                    grid_coord, axis=0, return_inverse=True, return_counts=True
                )
                
                idx_select = (
                    np.cumsum(np.insert(counts, 0, 0)[0:-1])
                    + np.random.randint(0, counts.max(), counts.size) % counts
                )
                idx_unique = idx_sort[idx_select]

                if self.return_inverse:
                    add_dict[category + "_inverse"] = inverse
                if self.return_grid_coord:
                    add_dict[category + "_grid_coord"] = grid_coord[idx_unique]
                if self.return_min_coord:
                    add_dict[category + "_min_coord"] = min_coord.reshape([1, 3])
            
        return data_dict | add_dict

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

@TRANSFORMS.register_module()
class MergeMapInput(object):
    def __init__(self, keys_dict):
        self.keys_dict = keys_dict

    def __call__(self, input_dict):
        for key in self.keys_dict.keys():
            output = None
            offset = []

            if 'inverse'in key:
                category = key.split("_")[0]

                if category == 'path':
                    friend_category = 'coord'
                elif category == 'grid':
                    friend_category = 'grid_coord'
                else:
                    raise ValueError(f"{category} is not supported")
                
                total_size = 0
                for k in self.keys_dict[key]:
                    if k in input_dict.keys():
                        instance_name = k.split("_")[0]
                        friend_key = instance_name + "_" + friend_category
                        friend_shape = input_dict[friend_key].shape[0]

                        if output is None:
                            output = input_dict[k]
                            total_size += friend_shape
                        else:
                            output = np.concatenate([output, input_dict[k] + total_size], axis=0)
                            total_size += friend_shape
                
                    input_dict[key] = output

            else:
                for k in self.keys_dict[key]:
                    if k in input_dict.keys():
                        if output is None:
                            output = input_dict[k]
                            offset.append(output.shape[0])
                        else:
                            output = np.concatenate([output, input_dict[k]], axis=0)
                            offset.append(output.shape[0])
                input_dict[key] = output
                input_dict[key + "_offset"] = np.array(offset)

        return input_dict