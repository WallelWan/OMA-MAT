import torch
import torch.nn as nn
import torch_scatter
import numpy as np


from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from pointcept.models.builder import MODELS, build_model


@MODELS.register_module()
class DefaultMapping(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        road_reduce='mean',
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

        self.road_reduce = road_reduce

    def forward(self, input_dict):
        feat = self.backbone(input_dict)

        # calculate similarity
        batch = offset2batch(input_dict['origin_batch_offset'])
        gt_batch = offset2batch(input_dict['gt_offset'])
        path_batch = offset2batch(input_dict['path_inverse_offset'])
        road_idx_batch = offset2batch(input_dict['road_idx_offset'])
        category = input_dict['category']
        path_inverse = input_dict['lane_to_path']
        road_idx = input_dict['road_idx']
        gt = input_dict['gt']

        input_dict['feat'] = feat

        batch_sim_list = []
        batch_sim_softmax_list = []
        batch_gt_list = []

        batch_path_sim_list = []
        batch_path_gt_list = []

        for i in range(batch.max().item() + 1):

            batch_gt = gt[gt_batch == i]
            batch_lane_index = (batch == i) * (category[:, 0] == 1)
            batch_lane_feat = feat[batch_lane_index]

            batch_road_index = (batch == i) * (category[:, 0] == 0)
            batch_road_feat = feat[batch_road_index]

            batch_road_idx = road_idx[road_idx_batch == i][:, 0]

            batch_path_index = (path_batch == i)
            batch_path_inverse = path_inverse[batch_path_index]

            if self.road_reduce == 'mean':
                batch_road_feat = torch_scatter.scatter_mean(
                    batch_road_feat, batch_road_idx, dim=0)
                batch_sim = (batch_lane_feat @ batch_road_feat.T) / np.sqrt(batch_lane_feat.shape[1])
                batch_sim_softmax = torch.softmax(batch_sim, dim=-1)

            elif self.road_reduce == 'max':
                batch_road_feat = torch_scatter.scatter_max(
                    batch_road_feat, batch_road_idx, dim=0)[0]
                batch_sim = (batch_lane_feat @ batch_road_feat.T) / np.sqrt(batch_lane_feat.shape[1])
                batch_sim_softmax = torch.softmax(batch_sim, dim=-1)

            elif self.road_reduce == 'softmax':
                batch_sim = (batch_lane_feat @ batch_road_feat.T) / np.sqrt(batch_lane_feat.shape[1])
                batch_softmax_sim = torch_scatter.scatter_softmax(
                    batch_sim, batch_road_idx, dim=1)
                batch_sim = torch_scatter.scatter_mean(
                    batch_softmax_sim * batch_sim, batch_road_idx, dim=1)
                batch_sim_softmax = torch.softmax(batch_sim, dim=-1)
            else:
                raise NotImplementedError

            batch_sim_list.append(batch_sim)
            batch_gt_list.append(batch_gt)
            batch_sim_softmax_list.append(batch_sim_softmax)

            batch_path_sim_list.append(batch_sim[batch_path_inverse])
            batch_path_gt_list.append(batch_gt[batch_path_inverse])

        input_dict['batch_gt_list'] = batch_gt_list
        input_dict['batch_sim_list'] = batch_sim_list

        input_dict['batch_path_sim_list'] = batch_path_sim_list
        input_dict['batch_path_gt_list'] = batch_path_gt_list

        if self.training:
            loss = self.criteria(input_dict, None)
            return dict(loss=loss)
        elif 'gt' in input_dict:
            loss = self.criteria(input_dict, None)
            return dict(loss=loss, pred=batch_sim_softmax_list, target=batch_gt_list, meta=input_dict)
        else:
            return dict(pred=batch_sim_softmax_list, meta=input_dict)