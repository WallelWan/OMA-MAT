"""
Mapping dataset for Nuscenes dataset

Author: Wan Jiaxu (wanjiaxu@buaa.edu.cn)
"""

import os
import shutil
import glob
import json
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from ..builder import DATASETS
from ..defaults import DefaultDataset
from ..transform import Compose, TRANSFORMS


@DATASETS.register_module()
class MappingDataset(DefaultDataset):
    def __init__(
        self,
        data_dir_path,
        dim=2,
        split="train",
        using_boundary=True,
        **kwargs,
    ):
        self.data_dir_path = data_dir_path
        self.dim = dim
        self.data_list = glob.glob(os.path.join(self.data_dir_path, "*.json"))
        self.using_boundary = using_boundary

        super().__init__(**kwargs)

    def get_data_list(self):
        return self.data_list

    def get_data(self, idx):
        json_path = self.data_list[idx % len(self.data_list)]

        origin_data = json.load(open(json_path, "r"))

        origin_data["scene_id"] = os.path.basename(json_path)[:-5]

        origin_data_str = json.dumps(origin_data)

        road, lane, boundary, path = origin_data["link"], origin_data["lane"], origin_data["vector"], origin_data['path']

        # make vectorial point list
        
        token_category = []

        road_point = [] # [N, 2, 2]
        road_id = []
        road_path_id = []
        road_idx = []
        road_path_inverse = []
        for j, data in enumerate(road):
            road_path_id.append(data["link_id"])
            for i in range(len(data["coords_norm"]) - 1):
                road_point.append([data["coords_norm"][i], data["coords_norm"][i + 1]])
                road_id.append(data["link_id"]) 
                road_idx.append([j, i, 0])
                road_path_inverse.append(len(road_path_inverse))
                token_category.append(0)

        lane_point = []
        lane_id = []
        lane_gt = []
        for data in lane:
            lane_point.append(data["coords_norm"])
            lane_id.append(data["instance_id"])
            token_category.append(1)
            if data["link_ids"][0] == "-1":
                lane_gt.append(-1)
            else:
                lane_gt.append(road_path_id.index(data["link_ids"][0]))

        boundary_point = []
        boundary_id = []
        boundary_idx = []
        boundary_path_inverse = []
        if self.using_boundary:
            for j, data in enumerate(boundary):
                for i in range(len(data["coords_norm"]) - 1):
                    boundary_point.append([data["coords_norm"][i], data["coords_norm"][i + 1]])
                    boundary_id.append(data["instance_id"]) 
                    boundary_idx.append([j, i, 2])
                    boundary_path_inverse.append(len(boundary_path_inverse))
                    token_category.append(2)
        else:
            boundary_point.append([[0.0, 0.0], [0.0, 0.0]])
            boundary_id.append("-1")
            boundary_idx.append([0, 0, 2])
            boundary_path_inverse.append(len(boundary_path_inverse))
            token_category.append(2)
        
        lane_idx = []
        lane_path_inverse = []
        for j, p in enumerate(path):
            for i, id in enumerate(p):
                lane_path_inverse.append(lane_id.index(id))
                lane_idx.append([j, i, 1])
        
        
        assert len(road_point) != 0 and len(lane_point) != 0, os.remove(json_path)
        if len(boundary_point) == 0:
            data_dict = {
                "scene_id": origin_data["scene_id"],
                "origin_data": origin_data_str,
                "road_point": (np.array(road_point)[:, :, :self.dim]).reshape(-1, 2),
                "road_id": np.array(road_id),
                "lane_point": (np.array(lane_point)[:, :, :self.dim]).reshape(-1, 2),
                "lane_id": np.array(lane_id),
                "road_path_inverse": np.array(road_path_inverse),
                "lane_path_inverse": np.array(lane_path_inverse),
                "lane_to_path": np.array(lane_path_inverse),
                "road_idx": np.array(road_idx),
                "lane_idx": np.array(lane_idx),
                "gt": np.array(lane_gt),
                "category": np.array(token_category)[:, None]
            }

        else:
            data_dict = {
                "scene_id": origin_data["scene_id"],
                "origin_data": origin_data_str,
                "road_point": (np.array(road_point)[:, :, :self.dim]).reshape(-1, 2),
                "road_id": np.array(road_id),
                "lane_point": (np.array(lane_point)[:, :, :self.dim]).reshape(-1, 2),
                "lane_id": np.array(lane_id),
                "boundary_point": (np.array(boundary_point)[:, :, :self.dim]).reshape(-1, 2),
                "boundary_id": np.array(boundary_id),
                "road_path_inverse": np.array(road_path_inverse),
                "lane_path_inverse": np.array(lane_path_inverse),
                "boundary_path_inverse": np.array(boundary_path_inverse),
                "lane_to_path": np.array(lane_path_inverse),
                "road_idx": np.array(road_idx),
                "lane_idx": np.array(lane_idx),
                "boundary_idx": np.array(boundary_idx),
                "gt": np.array(lane_gt),
                "category": np.array(token_category)[:, None]
            }
        return data_dict

    def prepare_test_data(self, idx):
        # load data

        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        if self.test_voxelize is not None:
            data_dict = self.test_voxelize(data_dict)

        if self.test_crop is not None:
            data_dict = self.test_crop(data_dict)

        if self.post_transform is not None:
            data_dict = self.post_transform(data_dict)

        return data_dict
