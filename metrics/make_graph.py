'''
Filename: /home/wanjiaxu.wjx/workspace/mapping/code/MappingNet/inference/metrics/read_file.py
Path: /home/wanjiaxu.wjx/workspace/mapping/code/MappingNet/inference/metrics
Created Date: Tuesday, April 15th 2025, 7:41:53 pm
Author: wanjiaxu

Copyright (c) 2025 Alibaba.com
'''


import os
import json
import numpy as np
import csv
import queue

def make_graph_by_file(data):
    # Step 1: make lane topo

    sample_point = data['sample_point']
    sample_point_vaild = data['sample_point_valid'] if 'sample_point_valid' in data else [True for _ in range(len(sample_point))]

    lane_data = data['lane']

    lane_topo_to = [[] for i in range(len(sample_point))]
    lane_topo_form = [[] for i in range(len(sample_point))]
    for i in range(len(lane_data)):
        lane_topo_to[lane_data[i]['coords_idx'][0]].append((lane_data[i]['coords_idx'][1], lane_data[i]))
        lane_topo_form[lane_data[i]['coords_idx'][1]].append((lane_data[i]['coords_idx'][0], lane_data[i]))

    root_node_list = []
    leaf_node_list = []

    for i in range(len(sample_point)):
        if len(lane_topo_to[i]) == 0 and len(lane_topo_form[i]) != 0:
            leaf_node_list.append(i)
        if len(lane_topo_to[i]) != 0 and len(lane_topo_form[i]) == 0:
            root_node_list.append(i)
    # Step 2. Make link graph

    link_data = data['link']
    link_topo_to = dict()
    link_topo_form = dict()

    for i in range(len(link_data)):
        link_topo_to[link_data[i]['link_id']] = []
        link_topo_form[link_data[i]['link_id']] = []

    for i in range(len(link_data)):
        if len(link_data[i]['output']) != 0:
            for name in link_data[i]['output']:
                link_topo_to[link_data[i]['link_id']].append(name)
        if len(link_data[i]['input']) != 0:
            for name in link_data[i]['input']:
                link_topo_form[link_data[i]['link_id']].append(name)

    lane_graph = {
        'sample_point': sample_point,
        "sample_point_vaild": sample_point_vaild,
        'topo_to': lane_topo_to,
        'topo_from': lane_topo_form,
        'root_node_list': root_node_list,
        'leaf_node_list': leaf_node_list,
    }

    link_graph = {
        'topo_to': link_topo_to,
        'topo_from': link_topo_form,
    }

    return lane_graph, link_graph

if __name__ == "__main__":
    json_path = '/home/wanjiaxu.wjx/workspace/mapping/code/MappingNet/datasets/nuscenes/val_model_inputs/singapore-hollandvillage_113bba80662946479dc5012fb9cc19de.json'

    data = json.load(open(json_path, 'r'))
    lane_graph, link_graph = make_graph_by_file(data)









    

    










