'''
Filename: /home/wanjiaxu.wjx/workspace/mapping/code/MappingNet/inference/metrics/metrics.py
Path: /home/wanjiaxu.wjx/workspace/mapping/code/MappingNet/inference/metrics
Created Date: Wednesday, April 16th 2025, 8:38:37 pm
Author: wanjiaxu

Copyright (c) 2025 Alibaba.com
'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import queue

from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from glob import glob

error_case_dict = dict()

import numpy as np

def is_sequence_consistent(pred_segments, gt_segments):
    def remove_consec_duplicates(segments, is_gt):
        ids = []
        for seg in segments:
            # 获取ID，处理gt的-1和pred的id
            current_id = seg["link_ids"][0] if is_gt else seg["pred_link_ids"][0]
            if is_gt and current_id == -1:  # 仅忽略gt中的-1
                continue
            ids.append(current_id)
        
        # 去除连续重复
        if not ids:
            return []
        simplified = [ids[0]]
        for id in ids[1:]:
            if id != simplified[-1]:
                simplified.append(id)
        return simplified

    # 生成简化序列
    pred_simplified = remove_consec_duplicates(pred_segments, is_gt=False)
    gt_simplified = remove_consec_duplicates(gt_segments, is_gt=True)

    # 当gt简化为空时认为一致
    return pred_simplified == gt_simplified if gt_simplified else True

def calculate_length_accuracy(pred_segments, gt_segments):
    def get_segment_info(segments, is_gt):
        """返回 (starts, ids, total_length) """
        if not segments:
            return [], [], 0
        
        # 计算每个线段的长度
        lengths = []
        for seg in segments:
            coords = seg["coords_norm"]
            length = np.linalg.norm(coords[1] - coords[0])
            lengths.append(length)
        
        total_length = sum(lengths)
        if total_length == 0:
            return [], [], 0
        
        # 计算百分比区间起点
        starts = np.cumsum([0] + lengths) / total_length * 100
        starts[-1] = 100.0  # 修正精度误差
        
        # 获取ID序列
        ids = [
            seg["link_ids"][0] if is_gt else seg["pred_link_ids"][0]
            for seg in segments
        ]
        
        return starts.tolist(), ids, total_length

    # 获取预测和真实路径的区间信息
    pred_starts, pred_ids, _ = get_segment_info(pred_segments, is_gt=False)
    gt_starts, gt_ids, gt_total = get_segment_info(gt_segments, is_gt=True)

    # 处理空路径
    if not gt_starts or gt_total == 0:
        return 1.0  # gt无效时认为完全正确

    # 合并所有事件点并排序
    event_points = np.unique(np.concatenate([pred_starts, gt_starts]))
    event_points = np.clip(event_points, 0, 100)
    event_points.sort()

    # 确保覆盖0-100
    if event_points[0] != 0:
        event_points = np.insert(event_points, 0, 0)
    if event_points[-1] != 100:
        event_points = np.append(event_points, 100)

    # 双指针遍历
    i_pred = 0
    i_gt = 0
    correct = 0.0

    for i in range(len(event_points)-1):
        start = event_points[i]
        end = event_points[i+1]
        interval_length = end - start

        # 移动pred指针
        while i_pred < len(pred_starts)-1 and pred_starts[i_pred+1] <= start:
            i_pred += 1
        
        # 移动gt指针
        while i_gt < len(gt_starts)-1 and gt_starts[i_gt+1] <= start:
            i_gt += 1

        # 获取当前区间ID
        pred_id = pred_ids[i_pred] if i_pred < len(pred_ids) else -1
        gt_id = gt_ids[i_gt] if i_gt < len(gt_ids) else -1

        # 当gt_id为-1时全部算正确
        if gt_id == -1 or pred_id == gt_id:
            correct += interval_length

    return correct / 100.0

def check_path_in_topo(path, gt_path, link_topo):
    global error_case_dict
    for i in range(len(path)-1):
        if path[i]['pred_link_ids'][0] == path[i]['pred_link_ids'][1]:
            link_id = path[i]['pred_link_ids'][0]
        elif path[i]['pred_link_id_probs'][0] > path[i]['pred_link_id_probs'][1]:
            link_id = path[i]['pred_link_ids'][0]
        else:
            link_id = path[i]['pred_link_ids'][1]

        if path[i+1]['pred_link_ids'][0] == path[i+1]['pred_link_ids'][1]:
            next_link_id = path[i]['pred_link_ids'][0]
        elif path[i+1]['pred_link_id_probs'][0] > path[i+1]['pred_link_id_probs'][1]:
            next_link_id = path[i+1]['pred_link_ids'][0]
        else:
            next_link_id = path[i+1]['pred_link_ids'][1]

        # link_id = path[i]['link_ids'][0]
        # next_link_id = path[i+1]['link_ids'][0]

        if link_id == next_link_id:
            continue
        if path[i]['link_ids'][0] == '-1':
            continue
        if link_id not in link_topo['topo_to'].keys() or next_link_id not in link_topo['topo_to'][link_id]:
            key = path[i]['scene_id'] + '@' + \
                str(link_id) + '@' + str(next_link_id)
            error_case_dict[key] = (path[i], path[i+1])
            return False
    return True

def check_path_in_topo_with_easy_acc(path, gt_path, link_topo):
    global error_case_dict

    gt_label_list = []
    for i in range(len(gt_path)):
        if len(gt_label_list) == 0:
            gt_label_list.append(gt_path[i]['link_ids'][0])
        else:
            if gt_path[i]['link_ids'][0] != gt_label_list[-1]:
                gt_label_list.append(gt_path[i]['link_ids'][0])
    
    gt_label_now = 0
    for i in range(len(path)):
        if path[i]['link_ids'][0] == '-1':
            continue
        if path[i]['pred_link_ids'][0] == gt_label_list[gt_label_now]:
            continue
        else:
            gt_label_now += 1
            if gt_label_now >= len(gt_label_list):
                return False
            if path[i]['pred_link_ids'][0] != gt_label_list[gt_label_now]:
                return False
            
    if gt_label_now != len(gt_label_list) - 1:
        return False
    return True

def check_path_in_topo_with_hard_acc(path, gt_path, link_topo):
    global error_case_dict
    for i in range(len(path)):
        if path[i]['link_ids'][0] == '-1':
            continue
        if path[i]['pred_link_ids'][0] != path[i]['link_ids'][0]:
            return False
    return True

def check_all_path_in_topo(path_list, link_topo):
    right = 0
    total = 0
    for long_path in path_list:
        for i in range(len(long_path)):
            for j in range(i+1, len(long_path)):
                path = long_path[i:j+1]

                is_path_topo = check_path_in_topo(path, link_topo)

                if is_path_topo:
                    acc += 1
                total += 1
    return right / total, right, total

import numpy as np

def is_sequence_consistent(pred_segments, gt_segments):
    def remove_consec_duplicates(segments, is_gt):
        ids = []
        for seg in segments:
            # 获取ID，处理gt的-1和pred的id
            current_id = seg["link_ids"][0] if is_gt else seg["pred_link_ids"][0]
            if is_gt and current_id == -1:  # 仅忽略gt中的-1
                continue
            ids.append(current_id)
        
        # 去除连续重复
        if not ids:
            return []
        simplified = [ids[0]]
        for id in ids[1:]:
            if id != simplified[-1]:
                simplified.append(id)
        return simplified

    # 生成简化序列
    pred_simplified = remove_consec_duplicates(pred_segments, is_gt=False)
    gt_simplified = remove_consec_duplicates(gt_segments, is_gt=True)

    # 当gt简化为空时认为一致
    return pred_simplified == gt_simplified if gt_simplified else True

def calculate_length_accuracy(pred_segments, gt_segments):
    def get_segment_info(segments, is_gt):
        """返回 (starts, ids, total_length) """
        if not segments:
            return [], [], 0
        
        # 计算每个线段的长度
        lengths = []
        for seg in segments:
            coords = np.array(seg["coords_norm"])
            length = np.linalg.norm(coords[1] - coords[0])
            lengths.append(length)
        
        total_length = sum(lengths)
        if total_length == 0:
            return [], [], 0
        
        # 计算百分比区间起点
        starts = np.cumsum([0] + lengths) / total_length * 100
        starts[-1] = 100.0  # 修正精度误差
        
        # 获取ID序列
        ids = [
            seg["link_ids"][0] if is_gt else seg["pred_link_ids"][0]
            for seg in segments
        ]
        
        return starts.tolist(), ids, total_length

    # 获取预测和真实路径的区间信息
    pred_starts, pred_ids, _ = get_segment_info(pred_segments, is_gt=False)
    gt_starts, gt_ids, gt_total = get_segment_info(gt_segments, is_gt=True)

    # 处理空路径
    if not gt_starts or gt_total == 0:
        return 1.0  # gt无效时认为完全正确

    # 合并所有事件点并排序
    event_points = np.unique(np.concatenate([pred_starts, gt_starts]))
    event_points = np.clip(event_points, 0, 100)
    event_points.sort()

    # 确保覆盖0-100
    if event_points[0] != 0:
        event_points = np.insert(event_points, 0, 0)
    if event_points[-1] != 100:
        event_points = np.append(event_points, 100)

    # 双指针遍历
    i_pred = 0
    i_gt = 0
    correct = 0.0

    for i in range(len(event_points)-1):
        start = event_points[i]
        end = event_points[i+1]
        interval_length = end - start

        # 移动pred指针
        while i_pred < len(pred_starts)-1 and pred_starts[i_pred+1] <= start:
            i_pred += 1
        
        # 移动gt指针
        while i_gt < len(gt_starts)-1 and gt_starts[i_gt+1] <= start:
            i_gt += 1

        # 获取当前区间ID
        pred_id = pred_ids[i_pred] if i_pred < len(pred_ids) else -1
        gt_id = gt_ids[i_gt] if i_gt < len(gt_ids) else -1

        # 当gt_id为-1时全部算正确
        if gt_id == -1 or pred_id == gt_id:
            correct += interval_length

    return correct / 100.0


def point_match(point1, point2, threshold):
    """
    支持一对多多匹配的改进版二分图匹配
    
    参数:
    point1 (np.ndarray): 原始点集 (N, 2/3)
    point2 (np.ndarray): 目标点集 (M, 2/3)
    threshold (float): 匹配最大欧氏距离
    
    返回:
    tuple: (matched_p1, matched_p2, unmatched_p1, unmatched_p2)
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # 计算距离矩阵 [N, M]
    dist_matrix = np.linalg.norm(point1[:, None] - point2, axis=2)
    
    # 找出每个point1最近的point2索引 [N]
    min_indices = np.argmin(dist_matrix, axis=1)
    min_distances = dist_matrix[np.arange(len(point1)), min_indices]
    
    # 构建匹配关系
    matched_p1 = []
    matched_p2 = []
    
    # 遍历每个point1
    for i in range(len(point1)):
        if min_distances[i] <= threshold:
            matched_p1.append(i)
            matched_p2.append(min_indices[i])
    
    # 转换为numpy数组
    matched_p1 = np.array(matched_p1, dtype=int)
    matched_p2 = np.array(matched_p2, dtype=int)
    
    # 计算未匹配点
    unmatched_p1 = np.setdiff1d(np.arange(len(point1)), matched_p1)
    all_p2 = np.arange(len(point2))
    matched_p2_unique = np.unique(matched_p2)
    unmatched_p2 = np.setdiff1d(all_p2, matched_p2_unique)
    
    return matched_p1, matched_p2, unmatched_p1, unmatched_p2

def find_path(lane_graph, point_start, point_end):
    point_start = int(point_start)
    point_end = int(point_end)

    topo_to = lane_graph['topo_to']

    path_list = []
    instance_path_list = []
    bfs_queue = queue.Queue()
    bfs_queue.put((point_start, [point_start], [], [], f"{point_start}"))

    path_hash_set = set()

    while not bfs_queue.empty():
        cur_node, cur_path, cur_instance_path, cur_instance, cur_str = bfs_queue.get()

        if cur_node == point_end:
            path_list.append(cur_path)
            instance_path_list.append(cur_instance)
        else:
            for child_node, lane_info in topo_to[cur_node]:
                next_str = f"{cur_str},{child_node}"
                if next_str not in path_hash_set and child_node not in cur_path and lane_info['instance_id'] not in cur_instance:
                    bfs_queue.put(
                        (child_node, cur_path + [child_node], cur_instance_path + [lane_info['instance_id']], cur_instance + [lane_info], next_str))
                    path_hash_set.add(next_str)
        

    return path_list, instance_path_list


def fit_adaptive_curve(points, k_samples=50):
    """自适应曲线拟合，支持2-3个点的路径"""
    n_points = len(points)

    # 两点情况：线性插值
    if n_points == 2:
        t = np.linspace(0, 1, k_samples)
        return points[0] + t[:, None] * (points[1] - points[0])

    # 三点情况：二次B样条
    elif n_points == 3:
        chord_lengths = np.array([0, np.linalg.norm(points[1]-points[0]),
                                  np.linalg.norm(points[2]-points[1])])
        t = chord_lengths.cumsum() / chord_lengths.sum()
        spl = make_interp_spline(t, points, k=2)  # 二次样条
        return spl(np.linspace(t[0], t[-1], k_samples))

    # 四点及以上：三次B样条
    else:
        chord_lengths = np.zeros(n_points)
        for i in range(1, n_points):
            chord_lengths[i] = chord_lengths[i-1] + \
                np.linalg.norm(points[i]-points[i-1])
        t = chord_lengths / chord_lengths[-1]
        spl = make_interp_spline(t, points, k=3)
        return spl(np.linspace(0, 1, k_samples))


def compute_trajectory_distance(traj1, traj2, k_samples=50):
    """ 计算两个轨迹之间的平均点距离 """
    sampled1 = fit_adaptive_curve(traj1, k_samples)
    sampled2 = fit_adaptive_curve(traj2, k_samples)
    return np.mean(np.linalg.norm(sampled1 - sampled2, axis=1))

def compute_chamfer_distance(traj1, traj2):
    """
    计算两个轨迹之间的倒角距离（Chamfer Distance），直接使用原始轨迹点

    参数:
        traj1 (List[List[float]]): 第一个轨迹点列表，格式为 [[x1, y1], [x2, y2], ...]
        traj2 (List[List[float]]): 第二个轨迹点列表，格式为 [[x1, y1], [x2, y2], ...]

    返回:
        float: 倒角距离
    """
    # 将轨迹点转换为 NumPy 数组
    points1 = np.array(traj1)
    points2 = np.array(traj2)

    # 计算倒角距离
    def chamfer_distance(points1, points2):
        # 计算 points1 中每个点到 points2 的最近距离
        distances1 = np.min(np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2), axis=1)
        # 计算 points2 中每个点到 points1 的最近距离
        distances2 = np.min(np.linalg.norm(points2[:, None, :] - points1[None, :, :], axis=2), axis=1)
        # 双向倒角距离（平均）
        return (np.mean(distances1) + np.mean(distances2)) / 2

    return chamfer_distance(points1, points2)

def path_match(pred, gt, max_distance):
    """
    带距离约束的轨迹最近邻匹配

    参数:
    pred/gt: 轨迹列表，每个元素为形状(N,2)或(N,3)的数组
    max_distance: 允许匹配的最大轨迹距离
    k_samples: 每条轨迹的采样点数（未使用，可忽略）

    返回:
    matches: 匹配对的索引列表，形如[(i,j),...]
    unmatched1: 未匹配的pred轨迹索引
    unmatched2: 未匹配的gt轨迹索引
    """
    n, m = len(pred), len(gt)
    cost = np.full((n, m), np.inf)

    for i in range(n):
        for j in range(m):
            try:
                dist = compute_chamfer_distance(pred[i], gt[j])
                if dist <= max_distance:
                    cost[i, j] = dist
            except Exception as e:
                print(f"轨迹{i}与{j}计算失败: {str(e)}")

    # 进行最近邻匹配
    matches = []
    for i in range(n):
        j_min = np.argmin(cost[i])
        if cost[i, j_min] <= max_distance:
            matches.append((i, j_min))

    # 计算未匹配项
    all_idx1 = set(range(n))
    matched_idx1 = set(i for i, _ in matches)
    unmatched1 = list(all_idx1 - matched_idx1)

    all_idx2 = set(range(m))
    matched_idx2 = set(j for _, j in matches)
    unmatched2 = list(all_idx2 - matched_idx2)

    return matches, unmatched1, unmatched2


def check_path_length(path):
    total_length = 0
    for i in range(len(path)):
        length = (path[i]['coords_norm'][0][0] - path[i]['coords_norm'][1][0]) ** 2 + \
            (path[i]['coords_norm'][0][1] - path[i]['coords_norm'][1][1]) ** 2
        length = np.sqrt(length)
        total_length += length
    total_length = int(np.floor(total_length / 2.5))
    return total_length


def eval_metric(pred_lane, gt_lane, link_topo, point_theshold, lane_theshold,
                acc_list= [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
                using_topo=True):
    # Step 1: Match sample point between pred and gt

    pred_sample_point = np.array(pred_lane['sample_point'])
    if 'sample_point_vaild' in pred_lane:
        pred_sample_point_vaild = np.array(pred_lane['sample_point_vaild'])
        valid_indices = np.where(pred_sample_point_vaild)[0]
        pred_sample_point_fitter = pred_sample_point[pred_sample_point_vaild]
    else:
        pred_sample_point_fitter = pred_sample_point

    gt_sample_point = np.array(gt_lane['sample_point'])

    # 进行匹配
    matched_pred_point, matched_gt_point, unmatched_pred_point, unmatched_gt_point = point_match(
        pred_sample_point_fitter, gt_sample_point, point_theshold)
    

    matched_pred_point = valid_indices[matched_pred_point]
    unmatched_pred_point = valid_indices[unmatched_pred_point]
    # Step 2: Match the path between pred and gt

    # Step 2.1: Search the path in same sample point between pred and gt
    # Tips: The point process in Step 1 is divided in 3 parts:
    #       1. The point both in the pred and gt
    #       2. The point just in the pred
    #       3. The point just in the gt
    # Then, the TP path only in the same sample point between pred and gt
    # And the TN path is in the path in Point 2-Point 2 and Point 1-Point 2.
    # The FP path is in the path in Point 2-Point 3 and Point 3-Point 3.

    K = 30

    def find_location_in_metrics(length):
        length = length
        if length > K - 1:
            return K - 1
        else:
            return length

    # TP = [0 for i in range(K)]
    # FP = [0 for i in range(K)]
    # FN = [0 for i in range(K)]
        
    TP = dict()
    FP = dict()
    FN = dict()

    for acc in acc_list:
        TP[acc] = [0 for i in range(K)]
        FP[acc] = [0 for i in range(K)]
        FN[acc] = [0 for i in range(K)]

    TP[-1] = [0 for i in range(K)]
    FP[-1] = [0 for i in range(K)]
    FN[-1] = [0 for i in range(K)]

    for i in tqdm(range(len(matched_pred_point)), leave=False):
        for j in  tqdm(range(i+1, len(matched_pred_point)), leave=False):

            gt_x = matched_gt_point[i]
            gt_y = matched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            pred_x = matched_pred_point[i]
            pred_y = matched_pred_point[j]
            pred_node_path, pred_instance_path = find_path(
                pred_lane, pred_x, pred_y)

            pred_coorm_path = [
                pred_sample_point[np.array(x)] for x in pred_node_path]
            gt_coorm_path = [
                gt_sample_point[np.array(x)] for x in gt_node_path]

            if len(pred_coorm_path) > 0 and len(gt_coorm_path) > 0:
                matches_path, unmatched_pred_path, unmatched_gt_path = path_match(
                    pred_coorm_path, gt_coorm_path, lane_theshold)
                for m_path in matches_path:
                    pre_check_pred_instance_path = pred_instance_path[m_path[0]]
                    pre_check_gt_instance_path = gt_instance_path[m_path[1]]

                    TP[-1][find_location_in_metrics(check_path_length(
                            pred_instance_path[m_path[0]]))] += 1
                    
                    
                    

                    # sequence_consistent
                    is_seq_cons = is_sequence_consistent(pre_check_pred_instance_path, pre_check_gt_instance_path)
                    if is_seq_cons:
                        accuracy = calculate_length_accuracy(pre_check_pred_instance_path, pre_check_gt_instance_path)

                        for acc in acc_list:
                            if accuracy >= acc:
                                TP[acc][find_location_in_metrics(check_path_length(
                                    pred_instance_path[m_path[0]]))] += 1
                            else:
                                FP[acc][find_location_in_metrics(check_path_length(
                                    pred_instance_path[m_path[0]]))] += 1

                    else:
                        for acc in acc_list:
                            FP[acc][find_location_in_metrics(check_path_length(
                                pred_instance_path[m_path[0]]))] += 1
                            
                for pred_i in unmatched_pred_path:
                    for acc in acc_list + [-1]:
                        FP[acc][find_location_in_metrics(
                            check_path_length(pred_instance_path[pred_i]))] += 1
                
                for gt_i in unmatched_gt_path:
                    for acc in acc_list + [-1]:
                        FN[acc][find_location_in_metrics(
                            check_path_length(gt_instance_path[gt_i]))] += 1

    # for i in range(len(matched_pred_point)):
    #     for j in range(len(unmatched_pred_point)):
    #         pred_x = matched_pred_point[i]
    #         pred_y = unmatched_pred_point[j]
    #         pred_node_path, pred_instance_path = find_path(
    #             pred_lane, pred_x, pred_y)

    #         for path in pred_instance_path:
    #             for acc in acc_list + [-1]:
    #                 FP[acc][find_location_in_metrics(check_path_length(path))] += 1

    # for i in range(len(unmatched_pred_point)):
    #     for j in range(len(matched_pred_point)):
    #         pred_x = unmatched_pred_point[i]
    #         pred_y = matched_pred_point[j]
    #         pred_node_path, pred_instance_path = find_path(
    #             pred_lane, pred_x, pred_y)

    #         for path in pred_instance_path:
    #             for acc in acc_list + [-1]:
    #                 FP[acc][find_location_in_metrics(check_path_length(path))] += 1

    # for i in range(len(unmatched_pred_point)):
    #     for j in range(len(unmatched_pred_point)):
    #         pred_x = unmatched_pred_point[i]
    #         pred_y = unmatched_pred_point[j]
    #         pred_node_path, pred_instance_path = find_path(
    #             pred_lane, pred_x, pred_y)

    #         for path in pred_instance_path:
    #             for acc in acc_list + [-1]:
    #                 FP[acc][find_location_in_metrics(check_path_length(path))] += 1

    for i in range(len(matched_gt_point)):
        for j in range(len(unmatched_gt_point)):
            gt_x = matched_gt_point[i]
            gt_y = unmatched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            for path in gt_instance_path:
                for acc in acc_list + [-1]:
                    FN[acc][find_location_in_metrics(check_path_length(path))] += 1

    for i in range(len(unmatched_gt_point)):
        for j in range(len(matched_gt_point)):
            gt_x = unmatched_gt_point[i]
            gt_y = matched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            for path in gt_instance_path:
                for acc in acc_list + [-1]:
                    FN[acc][find_location_in_metrics(check_path_length(path))] += 1

    for i in range(len(unmatched_gt_point)):
        for j in range(len(unmatched_gt_point)):
            gt_x = unmatched_gt_point[i]
            gt_y = unmatched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            for path in gt_instance_path:
                for acc in acc_list + [-1]:
                    FN[acc][find_location_in_metrics(check_path_length(path))] += 1

    # Step 3. Calculate the the P and R

    # P = Path in Point 1-Point 1 / (Path in Point 2-Point 2 + Path in Point 1-Point 1 + Path in Point 1-Point 2)
    # R = Path in Point 1-Point 1 / (Path in Point 3-Point 3 + Path in Point 1-Point 1 + Path in Point 1-Point 3)

    return TP, FP, FN


if __name__ == '__main__':
    from make_graph import make_graph_by_file
    import json
    import os
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Evaluate mapping metrics')
    parser.add_argument('--file_dir', type=str, required=True,
                        help='Directory containing prediction JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--gt_dir', type=str, 
                        default='/home/wanjiaxu.wjx/workspace/code/mapping/code/Pointcept/dataset/nuscenes/pointcept/val',
                        help='Directory containing ground truth JSON files')
    parser.add_argument('--distance_threshold', type=float, default=1.0,
                        help='Distance threshold for matching')
    
    args = parser.parse_args()
    
    error_file_list = []
    file_dir = args.file_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir
    distance_threshold = args.distance_threshold

    os.makedirs(output_dir, exist_ok=True)


    acc_list = [0.00, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    p_list = []
    r_list = []

    match_p_list = []
    match_r_list = []
    
    file_list = glob(os.path.join(file_dir, '*.json'))
    for json_path in tqdm(sorted(file_list)):
        if json_path.endswith('.json'):
            data = json.load(open(json_path, 'r'))

            # # special for mw data
            # data['sample_point'] = np.array(data['sample_point'])
            # data['sample_point'][:, 0] -= 48
            # data['sample_point'][:, 1] -= 32
            # data['sample_point'] = data['sample_point'][:, ::-1]  # 交换列
            # data['sample_point'][:, 0] *= -1
            # data['sample_point'] += 75
            # data['sample_point'] = data['sample_point'].tolist()

            if data['sample_point'] == [] or data['lane'] == []:
                error_file_list.append(json_path)
                continue
            
            if not os.path.exists(os.path.join(gt_dir, os.path.basename(json_path))):
                error_file_list.append(json_path)
                continue

            gt_data = json.load(open(os.path.join(gt_dir, os.path.basename(json_path)), 'r'))
            if gt_data['sample_point'] == [] or gt_data['lane'] == []:
                error_file_list.append(json_path)
                continue

            pred_lane_graph, _ = make_graph_by_file(data)

            gt_lane_graph, link_graph = make_graph_by_file(gt_data)

            sample_TP, sample_FP, sample_FN = eval_metric(
                pred_lane_graph, gt_lane_graph, link_graph, distance_threshold, distance_threshold,
                 acc_list=acc_list, using_topo=True)
            
            mean_p = []
            mean_r = []

            sample_P = dict()
            sample_R = dict()

            for acc in acc_list:
                tp = np.array(sample_TP[acc])
                fp = np.array(sample_FP[acc])
                fn = np.array(sample_FN[acc])

                tp[tp == np.nan] = 0
                fp[fp == np.nan] = 0
                fn[fn == np.nan] = 0
                total = ((tp > 0).astype(float) + (fn > 0).astype(float)+ (fp > 0).astype(float)) * 1e-8

                p = tp / (tp + fp + total)
                r = tp / (tp + fn + total)

                mean_p.append(p)
                mean_r.append(r)

                sample_P[acc] = p.tolist()
                sample_R[acc] = r.tolist()

                p_list.append(p.tolist())
                r_list.append(r.tolist())
            
            mean_p = np.nanmean(np.array(mean_p)[1:], axis=0)
            mean_r = np.nanmean(np.array(mean_r)[1:], axis=0)

            print("P: ", mean_p)
            print("R: ", mean_r)

            match_tp = np.array(sample_TP[-1])
            match_fp = np.array(sample_FP[-1])
            match_fn = np.array(sample_FN[-1])

            match_tp[match_tp == np.nan] = 0
            match_fn[match_fn == np.nan] = 0
            match_fp[match_fp == np.nan] = 0
            match_total = ((match_tp > 0).astype(float) + (match_fn > 0).astype(float)+ (match_fp > 0).astype(float)) * 1e-8

            match_p = match_tp / (match_tp + match_fp + match_total)
            match_r = match_tp / (match_tp + match_fn + match_total)

            match_p_list.append(match_p)
            match_r_list.append(match_r)

            print("Match P: ", match_p)
            print("Match R: ", match_r)

            output_dict = {"P": sample_P, "R": sample_R,
                           "TP": sample_TP, "FP": sample_FP, "FN": sample_FN,
                           "Match_P": match_p.tolist(), "Match_R": match_r.tolist()}
            
            json.dump(output_dict, open(
                os.path.join(output_dir, os.path.basename(json_path)), 'w'))
            
            p_list.append(mean_p)
            r_list.append(mean_r)

    print("=======================")
    print("Final Result:")
    print(
        f"P: {np.nanmean(np.array(p_list), axis=0)}, R: {np.nanmean(np.array(r_list), axis=0)}")
    print(f"Avg P: {np.nanmean(np.array(p_list))}, Avg R: {np.nanmean(np.array(r_list))}")
    print(f"Match P: {np.nanmean(np.array(match_p_list), axis=0)}, Match R: {np.nanmean(np.array(match_r_list), axis=0)}")
    print(f"Avg Match P: {np.nanmean(np.array(match_p_list))}, Avg Match R: {np.nanmean(np.array(match_r_list))}")
    print("=======================")
