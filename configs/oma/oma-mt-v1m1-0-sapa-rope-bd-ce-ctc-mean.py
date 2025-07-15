_base_ = ["../_base_/default_runtime.py"]

# Tester
test = dict(type="MatchTester", verbose=True)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="MatchEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

# misc custom setting
batch_size = 128  # bs: total bs in all gpus
num_worker = 16
mix_prob = 0

enable_amp = True

empty_cache = False
empty_cache_per_epoch = True

grid_size = [0.1, 0.1, 3.14 / 16]

# ==================================
# Ablation study
attention_list = ('spatial', 'path')
using_rope = True
using_boundary = True
loss_funciton = [dict(type="MappingCELoss", loss_weight=1.0, ignore_index=-1,
                   label_smoothing=0.1),
             dict(type="MappingCTCLoss", loss_weight=0.01)]
road_reduce = 'mean'
# ==================================

# model settings
model = dict(
    type="DefaultMapping",
    backbone=dict(
        type="MT-v1m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        depths=(4, 4, 4, 12, 4),
        channels=(96, 192, 384, 768, 1536),
        num_head=(4, 4, 8, 8, 8),
        patch_size=(1024, 1024, 1024, 1024, 1024),
        attention_list=attention_list,
        using_rope=using_rope,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
    ),
    criteria=loss_funciton,
    road_reduce=road_reduce,
)

# scheduler settings
epoch = 50
eval_epoch = 10
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "MappingDataset"

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_dir_path="dataset/nuscenes/pointcept/train",
        using_boundary=using_boundary,
        transform=[
            dict(type="CenterShift2D"),
            dict(type="RandomRotate2D", angle=[-1, 1], p=0.5),
            dict(type="RandomScale2D", scale=[0.9, 1.1]),
            dict(type="RandomFlip2D", p=0.5),
            dict(type="RandomJitter2D", sigma=0.005, clip=0.02),
            dict(type="PointToVector"),
            dict(
                type="GridSample2D",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(
                type="MergeMapInput",
                keys_dict={
                    "coord": ["road_coord", "lane_coord", "boundary_coord"],
                    "vector": ["road_vector", "lane_vector", "boundary_vector"],
                    "angle": ["road_angle", "lane_angle", "boundary_angle"],
                    "id": ["road_idx", "lane_idx", "boundary_idx"],
                    "path_inverse": ["road_path_inverse", "lane_path_inverse", "boundary_path_inverse"],
                    "grid_inverse": ["road_inverse", "lane_inverse", "boundary_inverse"],
                    "grid_coord": ["road_grid_coord", "lane_grid_coord", "boundary_grid_coord"],
                }
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "vector", "angle", "id", "category",
                      "path_inverse", "grid_inverse", "grid_coord", 
                      "id_offset", "coord_offset",
                      "gt", "road_idx", "lane_to_path"),
                feat_keys=("coord", "vector", "angle", "category"),
                offset_keys_dict={
                    "origin_batch_offset": "coord",
                    "grid_batch_offset": "grid_coord",
                    "path_inverse_offset": "lane_to_path",
                    "gt_offset": "gt",
                    "road_idx_offset": "road_idx",
                }
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_dir_path="dataset/nuscenes/pointcept/val",
        using_boundary=using_boundary,
        transform=[
            dict(type="CenterShift2D"),
            dict(type="PointToVector"),
            dict(
                type="GridSample2D",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(
                type="MergeMapInput",
                keys_dict={
                    "coord": ["road_coord", "lane_coord", "boundary_coord"],
                    "vector": ["road_vector", "lane_vector", "boundary_vector"],
                    "angle": ["road_angle", "lane_angle", "boundary_angle"],
                    "id": ["road_idx", "lane_idx", "boundary_idx"],
                    "path_inverse": ["road_path_inverse", "lane_path_inverse", "boundary_path_inverse"],
                    "grid_inverse": ["road_inverse", "lane_inverse", "boundary_inverse"],
                    "grid_coord": ["road_grid_coord", "lane_grid_coord", "boundary_grid_coord"],
                }
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "vector", "angle", "id", "category",
                      "path_inverse", "grid_inverse", "grid_coord", 
                      "id_offset", "coord_offset",
                      "gt", "road_idx", "lane_to_path"),
                feat_keys=("coord", "vector", "angle", "category"),
                offset_keys_dict={
                    "origin_batch_offset": "coord",
                    "grid_batch_offset": "grid_coord",
                    "path_inverse_offset": "lane_to_path",
                    "gt_offset": "gt",
                    "road_idx_offset": "road_idx",
                }
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_dir_path="dataset/nuscenes/pointcept/test",
        using_boundary=using_boundary,
        transform=[
            dict(type="CenterShift2D"),
            dict(type="PointToVector"),
            dict(
                type="GridSample2D",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(
                type="MergeMapInput",
                keys_dict={
                    "coord": ["road_coord", "lane_coord", "boundary_coord"],
                    "vector": ["road_vector", "lane_vector", "boundary_vector"],
                    "angle": ["road_angle", "lane_angle", "boundary_angle"],
                    "id": ["road_idx", "lane_idx", "boundary_idx"],
                    "path_inverse": ["road_path_inverse", "lane_path_inverse", "boundary_path_inverse"],
                    "grid_inverse": ["road_inverse", "lane_inverse", "boundary_inverse"],
                    "grid_coord": ["road_grid_coord", "lane_grid_coord", "boundary_grid_coord"],
                }
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "vector", "angle", "id", "category",
                      "path_inverse", "grid_inverse", "grid_coord", 
                      "id_offset", "coord_offset",
                      "gt", "road_idx", "lane_to_path", "origin_data"),
                feat_keys=("coord", "vector", "angle", "category"),
                offset_keys_dict={
                    "origin_batch_offset": "coord",
                    "grid_batch_offset": "grid_coord",
                    "path_inverse_offset": "lane_to_path",
                    "gt_offset": "gt",
                    "road_idx_offset": "road_idx",
                }
            ),
        ],
        test_mode=False,
    ),
)
