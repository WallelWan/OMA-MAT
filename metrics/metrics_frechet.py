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
from similaritymeasures import frechet_dist
from frechetdist import frdist

# 全局字典，用于存储错误案例
error_case_dict = dict()


def is_sequence_consistent(pred_segments, gt_segments):
    """
    检查预测路径段和真实路径段的序列是否一致
    
    参数:
        pred_segments: 预测的路径段列表
        gt_segments: 真实的路径段列表
    
    返回:
        bool: 序列是否一致
    """
    def remove_consec_duplicates(segments, is_gt):
        """
        移除连续重复的ID，并忽略gt中的-1
        
        参数:
            segments: 路径段列表
            is_gt: 是否为真实标签
        
        返回:
            list: 简化后的ID序列
        """
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
    """
    计算预测路径段与真实路径段的长度准确率
    使用双指针算法在百分比区间上计算匹配度
    
    参数:
        pred_segments: 预测的路径段列表
        gt_segments: 真实的路径段列表
    
    返回:
        float: 长度准确率 (0-1之间)
    """
    def get_segment_info(segments, is_gt):
        """
        提取路径段的区间起点、ID序列和总长度
        
        参数:
            segments: 路径段列表
            is_gt: 是否为真实标签
        
        返回:
            tuple: (starts, ids, total_length)
                - starts: 百分比区间起点列表
                - ids: ID序列
                - total_length: 总长度
        """
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
    支持一对多匹配的点集匹配算法
    为每个point1找到最近的point2，距离小于阈值则匹配
    
    参数:
        point1 (np.ndarray): 原始点集 (N, 2/3)
        point2 (np.ndarray): 目标点集 (M, 2/3)
        threshold (float): 匹配最大欧氏距离
    
    返回:
        tuple: (matched_p1, matched_p2, unmatched_p1, unmatched_p2)
            - matched_p1: 匹配的point1索引
            - matched_p2: 匹配的point2索引
            - unmatched_p1: 未匹配的point1索引
            - unmatched_p2: 未匹配的point2索引
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
    """
    使用BFS在车道图中查找从起点到终点的所有路径
    
    参数:
        lane_graph: 车道图，包含拓扑关系
        point_start: 起点ID
        point_end: 终点ID
    
    返回:
        tuple: (path_list, instance_path_list)
            - path_list: 节点路径列表
            - instance_path_list: 实例路径列表
    """
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
            # 到达终点，保存路径
            path_list.append(cur_path)
            instance_path_list.append(cur_instance)
        else:
            # 扩展当前节点
            for child_node, lane_info in topo_to[cur_node]:
                next_str = f"{cur_str},{child_node}"
                # 避免环路和重复路径
                if next_str not in path_hash_set and child_node not in cur_path and lane_info['instance_id'] not in cur_instance:
                    bfs_queue.put(
                        (child_node, cur_path + [child_node], cur_instance_path + [lane_info['instance_id']], cur_instance + [lane_info], next_str))
                    path_hash_set.add(next_str)

    return path_list, instance_path_list


def fit_adaptive_curve(points, k_samples=50):
    """
    自适应曲线拟合，根据点数选择合适的插值方法
    
    参数:
        points: 点列表
        k_samples: 采样点数
    
    返回:
        np.ndarray: 拟合后的采样点
    """
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


def compute_chamfer_distance(traj1, traj2):
    """
    计算两个轨迹之间的倒角距离（Chamfer Distance）
    使用原始轨迹点，不进行采样
    
    参数:
        traj1 (List[List[float]]): 第一个轨迹点列表，格式为 [[x1, y1], [x2, y2], ...]
        traj2 (List[List[float]]): 第二个轨迹点列表，格式为 [[x1, y1], [x2, y2], ...]

    返回:
        float: 倒角距离（双向平均）
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
    为每个预测轨迹找到最近的真实轨迹
    
    参数:
        pred: 预测轨迹列表，每个元素为形状(N,2)或(N,3)的数组
        gt: 真实轨迹列表，每个元素为形状(N,2)或(N,3)的数组
        max_distance: 允许匹配的最大轨迹距离

    返回:
        tuple: (matches, unmatched1, unmatched2)
            - matches: 匹配对的索引列表，形如[(i,j),...]
            - unmatched1: 未匹配的pred轨迹索引
            - unmatched2: 未匹配的gt轨迹索引
    """
    n, m = len(pred), len(gt)
    cost = np.full((n, m), np.inf)

    # 计算所有轨迹对之间的距离
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


def compute_frechet_distance(path_points, start_point, end_point):
    """
    计算路径与起点到终点直线之间的 Fréchet 距离
    使用 similaritymeasures 库的 frechet_dist 函数
    
    参数:
        path_points: 路径点列表 (N, 2) 或 (N, 3)
        start_point: 起点坐标
        end_point: 终点坐标
    
    返回:
        float: Fréchet 距离
    """
    path_points = np.array(path_points)
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # 计算路径中每段的累计长度
    segment_lengths = []
    cumulative_lengths = [0.0]
    
    for i in range(len(path_points) - 1):
        length = np.linalg.norm(path_points[i + 1] - path_points[i])
        segment_lengths.append(length)
        cumulative_lengths.append(cumulative_lengths[-1] + length)
    
    total_length = cumulative_lengths[-1]
    
    # 如果总长度为0，返回0
    if total_length == 0:
        return 0.0
    
    # 归一化累计长度到 [0, 1]
    normalized_cumulative = np.array(cumulative_lengths) / total_length
    
    # 根据路径的累计长度在直线上采样对应的点
    line_points = []
    for t in normalized_cumulative:
        point = start_point + t * (end_point - start_point)
        line_points.append(point)
    
    line_points = np.array(line_points)
    
    # 使用 similaritymeasures 库计算 Fréchet 距离
    frechet_distance = frechet_dist(path_points, line_points)
    
    return frechet_distance


def calculate_frechet_bin(path, max_distance=5.6, num_bins=10):
    """
    计算路径的 Fréchet 距离并返回对应的 bin 索引
    将路径与起点到终点的直线比较，计算 Fréchet 距离
    然后映射到指定数量的 bin，每个 bin 的宽度为 max_distance / num_bins
    
    参数:
        path: 路径段列表，每个元素包含 'coords_norm' 字段
        max_distance: Fréchet 距离的最大值，默认 5.6
        num_bins: bin 的数量，默认 10
    
    返回:
        int: bin 索引 (0 到 num_bins-1)，如果距离 >= max_distance 则返回 num_bins-1
    """
    if not path or len(path) == 0:
        return 0
    
    # 提取路径上的所有点
    path_points = []
    for segment in path:
        coords = segment['coords_norm']
        # 添加起点
        if len(path_points) == 0:
            path_points.append(coords[0])
        # 添加终点
        path_points.append(coords[1])
    
    path_points = np.array(path_points)
    
    # 如果路径只有一个点或两个重合的点，返回 bin 0
    if len(path_points) < 2:
        return 0
    
    start_point = path_points[0]
    end_point = path_points[-1]
    
    # 如果起点和终点重合，返回 bin 0
    if np.allclose(start_point, end_point):
        return 0
    
    # 计算 Fréchet 距离
    frechet_dist = compute_frechet_distance(path_points, start_point, end_point)
    
    # 映射到 bin (0 到 num_bins-1)
    # 每个 bin 的宽度为 max_distance / num_bins
    # 例如 max_distance=5.6, num_bins=10: [0, 0.56) -> 0, [0.56, 1.12) -> 1, ..., [5.04, 5.6] -> 9
    bin_width = max_distance / num_bins
    bin_index = int(np.floor(frechet_dist / bin_width))
    
    # 确保 bin_index 在 [0, num_bins-1] 范围内
    bin_index = min(bin_index, num_bins - 1)
    bin_index = max(bin_index, 0)
    
    return bin_index


def eval_metric(pred_lane, gt_lane, link_topo, point_theshold, lane_theshold,
                acc_list=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00],
                using_topo=True, max_frechet_distance=5.6, num_bins=10):
    """
    评估车道预测的主函数
    计算不同准确率阈值下的TP、FP、FN
    
    参数:
        pred_lane: 预测的车道图
        gt_lane: 真实的车道图
        link_topo: 链接拓扑关系
        point_theshold: 点匹配阈值
        lane_theshold: 车道匹配阈值
        acc_list: 准确率阈值列表
        using_topo: 是否使用拓扑关系
        max_frechet_distance: Fréchet 距离的最大值，默认 5.6
        num_bins: bin 的数量，默认 10
    
    返回:
        tuple: (TP, FP, FN) 字典，键为准确率阈值，值为长度分布列表
    """
    # Step 1: 匹配预测和真实的采样点
    pred_sample_point = np.array(pred_lane['sample_point'])
    pred_sample_point_vaild = np.array(pred_lane['sample_point_vaild'])
    valid_indices = np.where(pred_sample_point_vaild)[0]
    pred_sample_point_fitter = pred_sample_point[pred_sample_point_vaild]

    gt_sample_point = np.array(gt_lane['sample_point'])

    # 进行点匹配
    matched_pred_point, matched_gt_point, unmatched_pred_point, unmatched_gt_point = point_match(
        pred_sample_point_fitter, gt_sample_point, point_theshold)
    
    matched_pred_point = valid_indices[matched_pred_point]
    unmatched_pred_point = valid_indices[unmatched_pred_point]

    # Step 2: 匹配预测和真实的路径
    # 路径分为三类：
    # 1. 两端点都匹配的路径（TP候选）
    # 2. 只有预测端点的路径（FP）
    # 3. 只有真实端点的路径（FN）

    K = num_bins  # Fréchet 距离 bin 数量

    # 初始化统计字典
    TP = dict()
    FP = dict()
    FN = dict()

    for acc in acc_list:
        TP[acc] = [0 for i in range(K)]
        FP[acc] = [0 for i in range(K)]
        FN[acc] = [0 for i in range(K)]

    # -1表示仅匹配，不考虑准确率
    TP[-1] = [0 for i in range(K)]
    FP[-1] = [0 for i in range(K)]
    FN[-1] = [0 for i in range(K)]

    def find_location_in_metrics(frechet_bin):
        """将 Fréchet 距离 bin 映射到统计索引"""
        # frechet_bin 已经在 [0, 9] 范围内
        return frechet_bin
    
    # 创建路径缓存，避免重复计算 Fréchet bin
    # 使用路径的 instance_id 序列和路径类型（pred/gt）作为唯一标识
    path_bin_cache = {}

    def get_path_bin_cached(path, is_gt=False):
        """
        获取路径的 Fréchet bin，使用缓存避免重复计算
        
        参数:
            path: 路径段列表
            is_gt: 是否为真实路径（用于区分 pred 和 gt）
        
        返回:
            int: bin 索引
        """
        # 使用路径的 instance_id 序列和类型作为缓存键
        # 添加 'gt' 或 'pred' 前缀以区分不同来源的路径
        path_type = 'gt' if is_gt else 'pred'
        path_key = (path_type, tuple(seg['instance_id'] for seg in path))
        
        if path_key not in path_bin_cache:
            path_bin_cache[path_key] = calculate_frechet_bin(path, max_frechet_distance, num_bins)
        
        return path_bin_cache[path_key]
    
    # 遍历所有匹配点对，查找并匹配路径
    for i in tqdm(range(len(matched_pred_point)), leave=False):
        for j in tqdm(range(i+1, len(matched_pred_point)), leave=False):

            # 查找真实路径
            gt_x = matched_gt_point[i]
            gt_y = matched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            # 查找预测路径
            pred_x = matched_pred_point[i]
            pred_y = matched_pred_point[j]
            pred_node_path, pred_instance_path = find_path(
                pred_lane, pred_x, pred_y)

            # 转换为坐标路径
            pred_coorm_path = [
                pred_sample_point[np.array(x)] for x in pred_node_path]
            gt_coorm_path = [
                gt_sample_point[np.array(x)] for x in gt_node_path]

            if len(pred_coorm_path) > 0 and len(gt_coorm_path) > 0:
                # 匹配路径
                matches_path, unmatched_pred_path, unmatched_gt_path = path_match(
                    pred_coorm_path, gt_coorm_path, lane_theshold)
                
                # 处理匹配的路径
                for m_path in matches_path:
                    pre_check_pred_instance_path = pred_instance_path[m_path[0]]
                    pre_check_gt_instance_path = gt_instance_path[m_path[1]]

                    # 计算一次 Fréchet bin 并缓存（标记为 pred 路径）
                    pred_bin = get_path_bin_cached(pre_check_pred_instance_path, is_gt=False)

                    # 无条件计数（仅匹配）
                    TP[-1][find_location_in_metrics(pred_bin)] += 1

                    # 检查序列一致性
                    is_seq_cons = is_sequence_consistent(pre_check_pred_instance_path, pre_check_gt_instance_path)
                    if is_seq_cons:
                        # 计算长度准确率
                        accuracy = calculate_length_accuracy(pre_check_pred_instance_path, pre_check_gt_instance_path)

                        # 根据不同准确率阈值统计
                        for acc in acc_list:
                            if accuracy >= acc:
                                TP[acc][find_location_in_metrics(pred_bin)] += 1
                            else:
                                FP[acc][find_location_in_metrics(pred_bin)] += 1
                    else:
                        # 序列不一致，计为FP
                        for acc in acc_list:
                            FP[acc][find_location_in_metrics(pred_bin)] += 1
                
                # 处理未匹配的预测路径（FP）
                for pred_i in unmatched_pred_path:
                    pred_bin = get_path_bin_cached(pred_instance_path[pred_i], is_gt=False)
                    for acc in acc_list + [-1]:
                        FP[acc][find_location_in_metrics(pred_bin)] += 1
                
                # 处理未匹配的真实路径（FN）
                for gt_i in unmatched_gt_path:
                    gt_bin = get_path_bin_cached(gt_instance_path[gt_i], is_gt=True)
                    for acc in acc_list + [-1]:
                        FN[acc][find_location_in_metrics(gt_bin)] += 1

    # 处理包含未匹配点的路径（FN）
    # 情况1: 匹配点 -> 未匹配点
    for i in range(len(matched_gt_point)):
        for j in range(len(unmatched_gt_point)):
            gt_x = matched_gt_point[i]
            gt_y = unmatched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            for path in gt_instance_path:
                gt_bin = get_path_bin_cached(path, is_gt=True)
                for acc in acc_list + [-1]:
                    FN[acc][find_location_in_metrics(gt_bin)] += 1

    # 情况2: 未匹配点 -> 匹配点
    for i in range(len(unmatched_gt_point)):
        for j in range(len(matched_gt_point)):
            gt_x = unmatched_gt_point[i]
            gt_y = matched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            for path in gt_instance_path:
                gt_bin = get_path_bin_cached(path, is_gt=True)
                for acc in acc_list + [-1]:
                    FN[acc][find_location_in_metrics(gt_bin)] += 1

    # 情况3: 未匹配点 -> 未匹配点
    for i in range(len(unmatched_gt_point)):
        for j in range(len(unmatched_gt_point)):
            gt_x = unmatched_gt_point[i]
            gt_y = unmatched_gt_point[j]
            gt_node_path, gt_instance_path = find_path(gt_lane, gt_x, gt_y)

            for path in gt_instance_path:
                gt_bin = get_path_bin_cached(path, is_gt=True)
                for acc in acc_list + [-1]:
                    FN[acc][find_location_in_metrics(gt_bin)] += 1

    return TP, FP, FN


if __name__ == '__main__':
    from make_graph import make_graph_by_file
    import json
    import os

    # 配置路径
    error_file_list = []
    file_dir = '/home/wanjiaxu.wjx/workspace/code/mapping/code/MappingNet/outputs/pointcept/mapping-ablation/match-mt-v1m1-0-sapa-rope-bd-ce-ctc-mean/standard_result'
    gt_dir = '/home/wanjiaxu.wjx/workspace/code/mapping/code/Pointcept/dataset/nuscenes/pointcept/val'
    output_dir = '/home/wanjiaxu.wjx/workspace/code/mapping/code/MappingNet/outputs/pointcept/mapping-ablation/match-mt-v1m1-0-sapa-rope-bd-ce-ctc-mean/standard_result_frechet_10bin'

    os.makedirs(output_dir, exist_ok=True)

    # 准确率阈值列表
    acc_list = [0.00, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    # 初始化统计列表
    p_list = []
    r_list = []
    match_p_list = []
    match_r_list = []
    
    distance_threshold = 1.0
    
    # 遍历所有预测文件
    file_list = glob(os.path.join(file_dir, '*.json'))
    for json_path in tqdm(sorted(file_list)):
        if json_path.endswith('.json'):
            # 加载预测数据
            data = json.load(open(json_path, 'r'))
            if data['sample_point'] == [] or data['lane'] == []:
                error_file_list.append(json_path)
                continue
            
            # 检查真实数据是否存在
            if not os.path.exists(os.path.join(gt_dir, os.path.basename(json_path))):
                error_file_list.append(json_path)
                continue

            # 加载真实数据
            gt_data = json.load(open(os.path.join(gt_dir, os.path.basename(json_path)), 'r'))
            if gt_data['sample_point'] == [] or gt_data['lane'] == []:
                error_file_list.append(json_path)
                continue

            # 构建车道图
            pred_lane_graph, _ = make_graph_by_file(data)
            gt_lane_graph, link_graph = make_graph_by_file(gt_data)

            # 评估指标
            sample_TP, sample_FP, sample_FN = eval_metric(
                pred_lane_graph, gt_lane_graph, link_graph, distance_threshold, distance_threshold,
                acc_list=acc_list, using_topo=True, max_frechet_distance=5.6, num_bins=10)
            
            mean_p = []
            mean_r = []

            sample_P = dict()
            sample_R = dict()

            # 计算不同准确率阈值下的精确率和召回率
            for acc in acc_list:
                tp = np.array(sample_TP[acc])
                fp = np.array(sample_FP[acc])
                fn = np.array(sample_FN[acc])

                # 处理NaN值
                tp[tp == np.nan] = 0
                fp[fp == np.nan] = 0
                fn[fn == np.nan] = 0
                total = ((tp > 0).astype(float) + (fn > 0).astype(float)+ (fp > 0).astype(float)) * 1e-8

                # 计算精确率和召回率
                p = tp / (tp + fp + total)
                r = tp / (tp + fn + total)

                mean_p.append(p)
                mean_r.append(r)

                sample_P[acc] = p.tolist()
                sample_R[acc] = r.tolist()

                p_list.append(p.tolist())
                r_list.append(r.tolist())
            
            # 计算平均精确率和召回率（排除acc=0.00）
            mean_p = np.nanmean(np.array(mean_p)[1:], axis=0)
            mean_r = np.nanmean(np.array(mean_r)[1:], axis=0)

            print("P: ", mean_p)
            print("R: ", mean_r)

            # 计算匹配精确率和召回率（不考虑准确率阈值）
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

            # 保存结果
            output_dict = {"P": sample_P, "R": sample_R,
                           "TP": sample_TP, "FP": sample_FP, "FN": sample_FN,
                           "Match_P": match_p.tolist(), "Match_R": match_r.tolist()}
            
            json.dump(output_dict, open(
                os.path.join(output_dir, os.path.basename(json_path)), 'w'))
            
            p_list.append(mean_p)
            r_list.append(mean_r)

    # 输出最终统计结果
    print("=======================")
    print("Final Result:")
    print(
        f"P: {np.nanmean(np.array(p_list), axis=0)}, R: {np.nanmean(np.array(r_list), axis=0)}")
    print(f"Avg P: {np.nanmean(np.array(p_list))}, Avg R: {np.nanmean(np.array(r_list))}")
    print(f"Match P: {np.nanmean(np.array(match_p_list), axis=0)}, Match R: {np.nanmean(np.array(match_r_list), axis=0)}")
    print(f"Avg Match P: {np.nanmean(np.array(match_p_list))}, Avg Match R: {np.nanmean(np.array(match_r_list))}")
    print("=======================")
