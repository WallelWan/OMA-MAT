from ..losses.builder import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F

@LOSSES.register_module()
class MappingCELoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(MappingCELoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, meta_info, tmp=None):
        pred = meta_info["batch_sim_list"]
        target = meta_info["batch_gt_list"]

        all_loss = torch.stack([torch.mean(p) * 0.0 for p in pred]).mean()

        for p, t in zip(pred, target):
            if (t == -1).all():
                continue
            batch_loss = self.loss(p, t)
            if batch_loss.isnan():
                continue
            all_loss += batch_loss * self.loss_weight
        
        all_loss /= len(pred)

        return all_loss
    
@LOSSES.register_module()
class MappingCTCLoss(nn.Module):
    def __init__(
        self,
        reduction="mean",
        loss_weight=1.0,
    ):
        super(MappingCTCLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = nn.CTCLoss(
            reduction=reduction,
        )
    def forward(self, meta_info, tmp=None):
        pred = meta_info["batch_path_sim_list"]
        target = meta_info["batch_path_gt_list"]

        all_loss = torch.stack([torch.mean(p) * 0.0 for p in pred]).mean()

        for p, t in zip(pred, target):
            p = torch.cat([torch.zeros((p.shape[0], 1), device=p.device), p], dim=-1)
            t = t + 1

            if (t == 0).any():
                continue

            p = p.view(-1, 1, p.shape[-1])
            t = t.view(1, -1)

            batch_loss = self.loss(p, t, (1,), (1,))
            if batch_loss.isnan():
                continue
            all_loss += batch_loss * self.loss_weight
        
        all_loss /= len(pred)
        return all_loss