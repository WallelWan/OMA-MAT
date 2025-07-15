from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.serialization import encode
from pointcept.models.utils import (
    offset2batch,
    batch2offset,
    offset2bincount,
    bincount2offset,
)

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_flash_attention_utils import apply_rotary_emb
import numpy as np

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()

    assert cos.shape[1] == q.shape[-1] // 2
    assert cos.shape[0] == q.shape[1]

    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed

class ROPE(nn.Module):
    def __init__(self, channel: int, dim = 3, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, channel // dim, 2, dtype=torch.float) / channel))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    
def traditional_flash_attention(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p,
    softmax_scale,
    return_attn_probs
):
    """
    传统注意力实现，接口与 FlashAttention 的 flash_attn_varlen_qkvpacked_func 一致。
    """
    total_tokens = qkv.shape[0]
    num_heads = qkv.shape[2]
    head_dim = qkv.shape[3]

    # Step 1: 拆分 QKV
    q, k, v = torch.split(qkv, 1, dim=1)
    q = q.squeeze(1)  # (total_tokens, num_heads, head_dim)
    k = k.squeeze(1)
    v = v.squeeze(1)

    # Step 2: 生成序列 ID
    batch_size = len(cu_seqlens) - 1
    seq_ids = torch.zeros(total_tokens, dtype=torch.long, device=q.device)
    for i in range(batch_size):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        seq_ids[start:end] = i

    # Step 3: 构造注意力 mask
    mask = (seq_ids[:, None] == seq_ids[None, :]).unsqueeze(1)  # (total_tokens, 1, total_tokens)

    # Step 4: 调整 q 和 k 的维度顺序
    q = q.permute(1, 0, 2)  # (num_heads, total_tokens, head_dim)
    k = k.permute(1, 0, 2)  # (num_heads, total_tokens, head_dim)

    # Step 5: 计算注意力分数
    k_transposed = k.transpose(-1, -2)  # (num_heads, head_dim, total_tokens)
    scores = torch.matmul(q, k_transposed) * softmax_scale  # (num_heads, total_tokens, total_tokens)
    scores = scores.permute(1, 0, 2)  # (total_tokens, num_heads, total_tokens)

    # Step 6: 应用 mask
    scores = scores.masked_fill(~mask, float('-inf'))
    scores = scores.permute(1, 0, 2)  # (num_heads, total_tokens, total_tokens)

    # Step 7: 应用 softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Step 8: 应用 dropout
    if dropout_p > 0:
        attn_weights_dropout = F.dropout(attn_weights, p=dropout_p, training=True)
    else:
        attn_weights_dropout = attn_weights

    # Step 9: 计算输出
    v = v.permute(1, 0, 2)  # (num_heads, total_tokens, head_dim)
    feat = torch.matmul(attn_weights_dropout, v)  # (total_tokens, num_heads, head_dim)
    feat = feat.permute(1, 0, 2)  # (num_heads, total_tokens, head_dim) -> (total_tokens, num_heads, head_dim)

    # Step 10: 返回结果
    if return_attn_probs:
        return feat, torch.mean(attn_weights, dim=0), None
    else:
        return feat


class PathAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        using_rope=True,
        stage_index=0,
        block_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.stage_index = stage_index
        self.block_index = block_index
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index

        self.patch_size = patch_size
        self.attn_drop = attn_drop

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        self.using_rope = using_rope

    def forward(self, feat, path_coord, path_inverse, path_offset, position_embeddings):

        H = self.num_heads
        C = self.channels
        K = self.patch_size

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(path_offset)

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(feat)[path_inverse][pad]
        cos, sin = position_embeddings
        cos = cos[pad]
        sin = sin[pad]

        # qkv rope
        if self.using_rope:
            q, k, v = qkv.reshape(-1, 3, H, C // H).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
            q = q.squeeze(0)
            k = k.squeeze(0)
            qkv = torch.stack([q, k, v], dim=0).permute(1, 0, 2, 3)

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half().reshape(-1, 3, H, C // H),
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        )

        feat = feat.reshape(-1, C)
        feat = feat.to(qkv.dtype)[unpad]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        feat = torch_scatter.scatter(feat, path_inverse, 0, reduce='mean')

        return feat

    @torch.no_grad()
    def get_padding_and_inverse(self, offset):
        bincount = offset2bincount(offset)
        bincount_pad = (
            torch.div(
                bincount + self.patch_size - 1,
                self.patch_size,
                rounding_mode="trunc",
            )
            * self.patch_size
        )
        # only pad point when num of points larger than patch_size
        mask_pad = bincount > self.patch_size
        bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(
            torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(_offset_pad[-1], device=offset.device)
        unpad = torch.arange(_offset[-1], device=offset.device)
        cu_seqlens = []
        for i in range(len(offset)):
            unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1]
                    - self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1]
                    - 2 * self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    - self.patch_size
                ]
            pad[_offset_pad[i]: _offset_pad[i + 1]
                ] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i],
                    _offset_pad[i + 1],
                    step=self.patch_size,
                    dtype=torch.int32,
                    device=offset.device,
                )
            )

        # fix bug for last cu_seqlens
        cu_seqlens.append(_offset_pad[-1:].type(torch.int32))
        cu_seqlens = torch.cat(cu_seqlens, dim=0)

        return pad, unpad, cu_seqlens


class SpatialAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        shuffle_orders=True,
        using_rope=True,
        stage_index=0,
        block_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.stage_index = stage_index
        self.block_index = block_index
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.shuffle_orders = shuffle_orders

        self.patch_size = patch_size
        self.attn_drop = attn_drop

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        self.using_rope = using_rope

    def forward(self, feat, grid_coord, grid_inverse, grid_offset, serialized_order, serialized_inverse,
                position_embeddings):
        H = self.num_heads
        C = self.channels

        feat = torch_scatter.scatter(feat, grid_inverse, 0, reduce='mean')

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(grid_offset)

        # padding and reshape feat and batch for serialized point patch
        order = serialized_order[self.order_index][pad]
        inverse = unpad[serialized_inverse[self.order_index]]

        qkv = self.qkv(feat)[order]
        cos, sin = position_embeddings
        cos = cos[order]
        sin = sin[order]

        # qkv rope
        if self.using_rope:
            q, k, v = qkv.reshape(-1, 3, H, C // H).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
            q = q.squeeze(0)
            k = k.squeeze(0)
            qkv = torch.stack([q, k, v], dim=0).permute(1, 0, 2, 3)

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half().reshape(-1, 3, H, C // H),
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        )

        feat = feat.reshape(-1, C)
        feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        feat = feat[grid_inverse]

        return feat

    @torch.no_grad()
    def get_padding_and_inverse(self, offset):
        bincount = offset2bincount(offset)
        bincount_pad = (
            torch.div(
                bincount + self.patch_size - 1,
                self.patch_size,
                rounding_mode="trunc",
            )
            * self.patch_size
        )
        # only pad point when num of points larger than patch_size
        mask_pad = bincount > self.patch_size
        bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(
            torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(_offset_pad[-1], device=offset.device)
        unpad = torch.arange(_offset[-1], device=offset.device)
        cu_seqlens = []
        for i in range(len(offset)):
            unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1]
                    - self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1]
                    - 2 * self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    - self.patch_size
                ]
            pad[_offset_pad[i]: _offset_pad[i + 1]
                ] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i],
                    _offset_pad[i + 1],
                    step=self.patch_size,
                    dtype=torch.int32,
                    device=offset.device,
                )
            )

        # fix bug for last cu_seqlens
        cu_seqlens.append(_offset_pad[-1:].type(torch.int32))
        cu_seqlens = torch.cat(cu_seqlens, dim=0)

        return pad, unpad, cu_seqlens


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        attention,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        using_rope=True,
        stage_index=0,
        block_index=0,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.attention = attention
        if attention == 'path':
            attention_module = PathAttention
        elif attention == 'spatial':
            attention_module = SpatialAttention
        else:
            raise NotImplementedError

        self.norm1 = norm_layer(channels)
        self.attn = attention_module(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            using_rope=using_rope,
            stage_index=stage_index,
            block_index=block_index,
        )
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, map_dict):

        shortcut = map_dict["feat"]

        if self.pre_norm:
            feat = self.norm1(map_dict["feat"])

        if self.attention == 'path':
            feat = self.attn(feat, map_dict["path_coord"], map_dict["path_inverse"], map_dict["path_offset"],
                             map_dict["path_position_embeddings"])
        elif self.attention == 'spatial':
            feat = self.attn(feat, map_dict["grid_coord"], map_dict["grid_inverse"],
                             map_dict["grid_offset"], map_dict["spatial_order"], map_dict["spatial_inverse"],
                             map_dict["grid_position_embeddings"])
        else:
            raise NotImplementedError

        feat = self.drop_path(feat)
        feat = shortcut + feat
        if not self.pre_norm:
            feat = self.norm1(feat)

        shortcut = feat
        if self.pre_norm:
            feat = self.norm2(feat)
        feat = self.drop_path(self.mlp(feat))
        feat = shortcut + feat
        if not self.pre_norm:
            feat = self.norm2(feat)

        map_dict["feat"] = feat

        return map_dict

class MapSample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        norm_layer=None,
        act_layer=None,
        shuffle_orders=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle_orders = shuffle_orders
        self.num_heads = num_heads

        self.rotary_emb = ROPE(out_channels // num_heads)

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        if act_layer is not None:
            self.act = act_layer()
    def forward(self, map_dict):
        
        feat = map_dict["feat"]
        feat = self.act(self.norm(self.proj(feat)))
        map_dict["feat"] = feat

        # ROPE
        max_grid_coord = map_dict["max_grid_coord"]
        grid_coord = map_dict["grid_coord"]

        gird_rotary_pos_emb_full = self.rotary_emb(max_grid_coord + 1)
        grid_rotary_pos_emb = gird_rotary_pos_emb_full[grid_coord].flatten(1)
        grid_rotary_pos_emb = torch.cat((grid_rotary_pos_emb, grid_rotary_pos_emb), dim=-1)
        grid_position_embeddings = (grid_rotary_pos_emb.cos(), grid_rotary_pos_emb.sin())
        map_dict["grid_position_embeddings"] = grid_position_embeddings

        max_path_coord = map_dict["max_path_coord"]
        path_coord = map_dict["path_coord"]
        path_rotary_pos_emb_full = self.rotary_emb(max_path_coord + 1)
        path_rotary_pos_emb = path_rotary_pos_emb_full[path_coord].flatten(1)
        path_rotary_pos_emb = torch.cat((path_rotary_pos_emb, path_rotary_pos_emb), dim=-1)
        path_position_embeddings = (path_rotary_pos_emb.cos(), path_rotary_pos_emb.sin())
        map_dict["path_position_embeddings"] = path_position_embeddings

        if self.shuffle_orders:
            order = map_dict["spatial_order"]
            inverse = map_dict["spatial_inverse"]

            perm = torch.randperm(order.shape[0])
            order = order[perm]
            inverse = inverse[perm]

            map_dict["spatial_order"] = order
            map_dict["spatial_inverse"] = inverse
        
        return map_dict


@MODELS.register_module("MT-v1m1")
class MapTransformer(nn.Module):
    def __init__(self,
                 in_channels=6,
                 out_channels=512,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 depths=(4, 4, 4, 12, 4),
                 channels=(48, 96, 192, 384, 512),
                 num_head=(3, 6, 12, 24, 32),
                 patch_size=(1024, 1024, 1024, 1024, 1024),
                 attention_list=('spatial', 'path'),
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.3,
                 pre_norm=True,
                 shuffle_orders=True,
                 using_rope=True):

        super().__init__()
        self.num_stages = len(depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(depths)
        assert self.num_stages == len(channels)
        assert self.num_stages == len(num_head)
        assert self.num_stages == len(patch_size)

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = nn.Sequential(
            nn.Linear(in_channels, channels[0]),
            bn_layer(channels[0]),
            act_layer(),
        )

        # encoder
        drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(depths))
        ]
        self.model = nn.Sequential()
        for s in range(self.num_stages):
            drop_path_ = drop_path[
                sum(depths[:s]): sum(depths[: s + 1])
            ]
            stage = nn.Sequential()
            if s > 0:
                stage.append(
                    MapSample(
                        in_channels=channels[s - 1],
                        out_channels=channels[s],
                        num_heads=num_head[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                )
            for i in range(depths[s]):
                stage.append(
                    Block(
                        channels=channels[s],
                        num_heads=num_head[s],
                        patch_size=patch_size[s],
                        attention=attention_list[i % len(attention_list)],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        using_rope=using_rope,
                        stage_index=s,
                        block_index=i,
                    )
                )
            if len(stage) != 0:
                self.model.append(module=stage)

            self.proj = nn.Sequential(
                ln_layer(channels[s]),
                nn.Linear(channels[s], mlp_ratio * out_channels),
                act_layer(),
                nn.Linear(mlp_ratio * out_channels, out_channels))
            
            self.rotary_emb = ROPE(channels[0] / num_head[0])

    def forward(self, input_dict):
        feat = input_dict["feat"]

        # path attention parameter
        path_inverse = input_dict["path_inverse"]
        path_offset = input_dict["id_offset"]
        path_coord = input_dict["id"]

        # grid attention parameter
        grid_inverse = input_dict["grid_inverse"]
        grid_offset = input_dict["grid_batch_offset"]
        grid_coord = input_dict["grid_coord"]

        _, spatial_order, spatial_inverse = self.serialization(
            grid_coord, grid_offset, self.order)
        
        # calculate max grid
        max_path_coord = path_coord.max()
        max_grid_coord = grid_coord.max()

        # create position embeddings to be shared across the decoder layers
        gird_rotary_pos_emb_full = self.rotary_emb(max_grid_coord + 1)
        grid_rotary_pos_emb = gird_rotary_pos_emb_full[grid_coord].flatten(1)
        grid_rotary_pos_emb = torch.cat((grid_rotary_pos_emb, grid_rotary_pos_emb), dim=-1)
        grid_position_embeddings = (grid_rotary_pos_emb.cos(), grid_rotary_pos_emb.sin())

        path_rotary_pos_emb_full = self.rotary_emb(max_path_coord + 1)
        path_rotary_pos_emb = path_rotary_pos_emb_full[path_coord].flatten(1)
        path_rotary_pos_emb = torch.cat((path_rotary_pos_emb, path_rotary_pos_emb), dim=-1)
        path_position_embeddings = (path_rotary_pos_emb.cos(), path_rotary_pos_emb.sin())

        feat = self.embedding(feat)

        map_dict = {
            "feat": feat,
            "path_coord": path_coord,
            "path_inverse": path_inverse,
            "path_offset": path_offset,
            "grid_coord": grid_coord,
            "grid_inverse": grid_inverse,
            "grid_offset": grid_offset,
            "spatial_order": spatial_order,
            "spatial_inverse": spatial_inverse,
            "max_path_coord": max_path_coord,
            "path_position_embeddings": path_position_embeddings,
            "max_grid_coord": max_grid_coord,
            "grid_position_embeddings": grid_position_embeddings,
        }

        map_dict = self.model(map_dict)
        feat = self.proj(map_dict["feat"])
        return feat

    @torch.no_grad()
    def serialization(self, grid_coord, offset, order=["z"], depth=None):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        batch = offset2batch(offset)

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(grid_coord.max() + 1).bit_length()
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(grid_coord, batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        # if shuffle_orders:
        #     perm = torch.randperm(code.shape[0])
        #     code = code[perm]
        #     order = order[perm]
        #     inverse = inverse[perm]

        return code, order, inverse
