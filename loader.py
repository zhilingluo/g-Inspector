# -*- coding: UTF-8 -*-

import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run g-Inspector")
    parser.add_argument('--data', nargs='?', default='mutag',
                        help='Input data name')
    parser.add_argument('--test-ratio', type=float, default=0.3,
                        help='Test ratio. Default is 0.3.')
    parser.add_argument('--instruction-length', type=int, default=3,
                        help='The shift instruction length m. Default is 3')
    parser.add_argument('--hidden-layers', type=int, default=7,
                        help='The hidden layers number T. Default is 7.')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Batch size of training. Default is 20.')
    parser.add_argument('--iter', default=20000, type=int,
                        help='Number of epochs. Default is 20000.')
    parser.add_argument('--agent', default=1, type=int,
                        help='Agent numbers in multi agent version. Default is 1')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def get_labels(labels_dir):
    labels = np.loadtxt(labels_dir)
    labels[labels < 0] = 0
    labelY = labels.astype(np.int32)
    return labelY


def get_graph(embedding_dir):
    print('get graph from ' + embedding_dir)
    num_embeddings = sum([len(files) for root, dirs, files in os.walk(embedding_dir)])
    graphs = []
    for i in range(num_embeddings):
        f = open(embedding_dir + "%s.embeddings" % i, "r")
        line = f.readline()
        head = line.strip().split(" ")
        node_num = head[0]
        node_dim = head[1]
        line = f.readline()

        vector_dict = {}
        while line:
            curline = line.strip().split(" ")
            vector_dict[int(curline[0])] = [float(s) for s in curline[1:]]
            line = f.readline()
        f.close()

        vector_dict = dict(sorted(vector_dict.items(), key=lambda e: e[0]))
        graphs.append(np.array(vector_dict.values()))

    graphs = np.array(graphs)
    return graphs


def cal_topk(graphs, topk):
    distances = []
    for i in range(len(graphs)):
        i_distances = []
        for j in range(len(graphs[i])):
            i_j_distances = {}
            for k in range(len(graphs[i])):
                if (j != k):
                    node_a = graphs[i][j]
                    node_b = graphs[i][k]
                    dist = np.linalg.norm(node_a - node_b)
                    i_j_distances[k] = dist
            lis = sorted(i_j_distances.items(), key=lambda e: e[1])
            if len(lis) >= topk:
                i_distances.append(list([lis[m][0] for m in range(topk)]))
            else:
                tmp = []
                for r in range(topk // len(lis)):
                    tmp.extend([lis[m][0] for m in range(len(lis))])
                tmp.extend([lis[m][0] for m in range(topk % len(lis))])
                i_distances.append(tmp)
        distances.append(np.array(i_distances))
    distances = np.array(distances)
    return distances


def node(graphs):
    nodes = graphs.size
    node_num = np.zeros((nodes))
    for i in range(nodes):
        node_num[i] = len(graphs[i])
    return node_num


def unifyGraphSize(org_graphs, max_nodes, node_dim, retype=False):
    graphs = np.zeros((len(org_graphs), max_nodes, node_dim))
    for i in range(len(org_graphs)):
        orgGraph = org_graphs[i]
        orgLen = len(orgGraph)

        if (max_nodes >= orgLen):
            newGraph = np.pad(orgGraph, ((0, (max_nodes - orgLen)), (0, 0),), 'constant', constant_values=(0, 0))
        else:
            newGraph = orgGraph[0:max_nodes]

        graphs[i] = newGraph

    if retype:
        graphs.astype(np.int)
    return graphs


class DataSet(object):
    """docstring for DataSet"""

    def __init__(self, graphs, labels, neighbors, node_numbers):
        self._graphs = graphs
        self._labels = labels
        self._neighbors = neighbors
        self._node_numbers = node_numbers
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = graphs.shape[0]

    @property
    def graphs(self):
        return self._graphs

    @property
    def labels(self):
        return self._labels

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._graphs = self._graphs[perm]
            self._labels = self._labels[perm]
            self._neighbors = self._neighbors[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._graphs[start:end], self._labels[start:end], self._neighbors[start:end], self._node_numbers[
                                                                                             start:end]


def read_data_sets(data, test_ratio):
    class DataSets(object):
        max_node_num = 0
        min_node_num = 0
        n_classes = 0
        node_dim = 0

    data_sets = DataSets()

    data_embedding_dir = 'dataset/' + data + '/'
    label_dir = 'dataset/' + data + '_labels.txt'

    graphs = get_graph(data_embedding_dir)
    labels = get_labels(label_dir)
    node_numbers = node(graphs)
    data_sets.max_node_num = np.max(node_numbers).astype(np.int)
    data_sets.min_node_num = np.min(node_numbers).astype(np.int)
    data_sets.n_classes = len(np.unique(labels))
    data_sets.node_dim = len(graphs[0][0])
    print("{}: \n \tmax node num: {} \n\tmin node num: {} \n\ttotal classes: {} \n\tnode embedding dimensions: {}".
          format(data, data_sets.max_node_num, data_sets.min_node_num, data_sets.n_classes, data_sets.node_dim))

    x_train, x_test, y_train, y_test = train_test_split(graphs, labels, test_size=test_ratio)

    data_sets.train = DataSet(unifyGraphSize(x_train, data_sets.max_node_num, data_sets.node_dim),
                              y_train,
                              unifyGraphSize(cal_topk(x_train, data_sets.min_node_num), data_sets.max_node_num,
                                             data_sets.min_node_num, retype=True),
                              node(x_train))
    data_sets.test = DataSet(unifyGraphSize(x_test, data_sets.max_node_num, data_sets.node_dim),
                             y_test,
                             unifyGraphSize(cal_topk(x_test, data_sets.min_node_num), data_sets.max_node_num,
                                            data_sets.min_node_num, retype=True),
                             node(x_test))
    return data_sets
