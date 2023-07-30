# -*- coding: utf-8 -*-

import dgl
import numpy as np
import torch
import networkx as nx

import matplotlib.pyplot as plt


def build_graph4ke(config: dict, from_e: bool):
    q = config['q']
    q = q.detach().cpu().numpy()
    know_num = config['know_num']
    exer_num = config['prob_num']
    node = exer_num + know_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    indices = np.where(q != 0)
    if from_e:
        for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
            edge_list.append((int(exer_id), int(know_id + exer_num - 1)))
    else:
        for exer_id, know_id in zip(indices[0].tolist(), indices[1].tolist()):
            edge_list.append((int(know_id + exer_num - 1), int(exer_id)))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


def build_graph4se(config: dict, from_s: bool):
    np_train = config['np_train']
    stu_num = config['stu_num']
    exer_num = config['prob_num']
    node = stu_num + exer_num
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    for index in range(np_train.shape[0]):
        stu_id = np_train[index, 0]
        exer_id = np_train[index, 1]
        if from_s:
            edge_list.append((int(stu_id + exer_num - 1), int(exer_id)))
        else:
            edge_list.append((int(exer_id), int(stu_id + exer_num - 1)))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g


def build_graph4di(config: dict):
    graph = config['directed_graph']
    know_num = config['know_num']
    g = dgl.DGLGraph()
    node = know_num
    g.add_nodes(node)
    edge_list = []
    src_idx_np, tar_idx_np = np.where(graph != 0)
    for src_indx, tar_index in zip(src_idx_np.tolist(), tar_idx_np.tolist()):
        edge_list.append((int(src_indx), int(tar_index)))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g
