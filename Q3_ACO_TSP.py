# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q3_ACO_TSP.py
@Date: 2022/5/14
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 导入数据
def loadData(data_path):
    with open(data_path, "r", encoding='utf-8') as f:
        data = f.read()
        data = data.split()
        data_list = []
        for one in data:
            try:
                trans = int(one)
                data_list.append(trans)
            except:
                continue
        # print(data_list)
    return data_list

# 转化为邻接矩阵
def transform2matrix(data_list):
    graph_mat = np.zeros((8, 8))
    for i in range(0, len(data_list), 3):
        key1 = data_list[i] - 1
        key2 = data_list[i+1] - 1
        value = data_list[i+2]
        graph_mat[key1, key2] = value
        graph_mat[key2, key1] = value
    for i in range(8):
        for j in range(8):
            if i == j or graph_mat[i, j] == 0:
                graph_mat[i, j] = 99999
    return graph_mat

# 画路线图
def drawFigure(data_list, path, pos_ctrl):
    # 转化并记录待标记路线
    path_num = len(path) - 1
    path_list = []
    for i in range(path_num):
        path_list.append([int(path[i]), int(path[i+1])])
    # 创建图对象
    G1 = nx.Graph()
    G2 = nx.DiGraph()
    # 增加节点
    G1.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    G2.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    # 增加权重，数据格式（节点1，节点2，权重）
    for i in range(0, len(data_list), 3):
        G1.add_edge(data_list[i], data_list[i + 1], weight=data_list[i + 2])

    # 固定节点位置
    if pos_ctrl == 1:
        pos = {
            1: [11, 11],
            2: [6, 11],
            3: [6, 0.5],
            4: [1, 8],
            5: [1, 3],
            6: [11, 1.5],
            7: [13, 7],
            8: [6, 5]
        }
    elif pos_ctrl == 2:
        pos = {
            1: [12, 11],
            2: [6, 11],
            3: [8, 0.5],
            4: [1, 7],
            5: [1, 3],
            6: [11, 1.5],
            7: [13, 7],
            8: [6, 5]
        }
    # 重新获取权重序列
    weights = nx.get_edge_attributes(G1, "weight")
    # 画节点图
    nx.draw_networkx(G1, pos, with_labels=True, node_color='#dfd7d7', edge_color='#6b5152')
    # 画路径
    nx.draw_networkx_edges(G2, pos, edgelist=path_list, edge_color='#0000FF', width=2.0)
    # 画权重图
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=weights)
    # 展示
    plt.show()

# 蚁群算法
def ACO(graph_mat, data_list, pos_ctrl):
    # 初始化参数
    distance = graph_mat.copy()
    ant_num = 100
    city_num = 8
    alpha = 1
    beta = 2
    rho = 0.1
    gen = 0
    GENERATION = 200
    Q = 1

    # 信息素矩阵
    pheromone_table = np.ones((city_num, city_num))

    # 候选集矩阵
    candidate_table = np.zeros((ant_num, city_num)).astype(int)

    # 最短路径矩阵
    path_best = np.zeros((GENERATION, city_num))

    # 最短路径值
    distance_best = np.zeros(GENERATION)

    # 倒数矩阵
    distance_daoshu_table = 1.0 / distance

    while gen < GENERATION:
        # 令所有蚂蚁游历的初始城市为城市1
        candidate_table[:, 0] = 0
        length = np.zeros(ant_num)
        # 选择下一个城市
        for i in range(ant_num):
            # 删除已经访问的城市1
            unvisit_city = list(range(city_num))
            visited_city = candidate_table[i, 0]
            unvisit_city.remove(visited_city)
            for j in range(1, city_num):
                pro_trans = np.zeros(len(unvisit_city))
                # 计算选择下一城市的概率
                for k in range(len(unvisit_city)):
                    # (信息素浓度^alpha)*(城市适应度的倒数^beta)
                    pro_trans[k] = np.power(pheromone_table[visited_city][unvisit_city[k]], alpha) * np.power(
                        distance_daoshu_table[visited_city][unvisit_city[k]], beta)
                # 轮盘赌选择
                cumsum_probtrans = (pro_trans / sum(pro_trans)).cumsum()
                cumsum_probtrans -= np.random.rand()
                next_city = unvisit_city[list(cumsum_probtrans > 0).index(True)]
                # 得到下一个访问的城市
                candidate_table[i, j] = next_city
                unvisit_city.remove(next_city)
                length[i] += distance[visited_city][next_city]
                visited_city = next_city
            # 计算最后一个访问的城市与初始城市的距离，形成回路
            length[i] += distance[visited_city][candidate_table[i, 0]]

        # 记录每次迭代的最短路径及其值
        if gen == 0:
            distance_best[gen] = length.min()
            path_best[gen] = candidate_table[length.argmin()].copy()
        else:
            if length.min() > distance_best[gen - 1]:
                distance_best[gen] = distance_best[gen - 1]
                path_best[gen] = path_best[gen - 1].copy()
            else:
                distance_best[gen] = length.min()
                path_best[gen] = candidate_table[length.argmin()].copy()

        # 信息素更新
        change_pheromone_table = np.zeros((city_num, city_num))
        for i in range(ant_num):
            for j in range(city_num - 1):
                change_pheromone_table[candidate_table[i, j]][candidate_table[i][j + 1]] += Q / length[i]
            #最后一个城市和城市1的信息素增加量
            change_pheromone_table[candidate_table[i, j + 1]][candidate_table[i, 0]] += Q / length[i]
        #信息素更新
        pheromone_table = (1 - rho) * pheromone_table + change_pheromone_table
        gen += 1

    path_record = path_best[-1] + 1
    path_record = path_record.astype(int)
    path_print = str(path_record[0])
    for i in path_record[1:8]:
        path_print = path_print + '>>' + str(i)
    path_print = path_print + '>>' + str(path_record[0])
    print("蚁群算法的最优路径为:", path_print)
    print("最短回路距离为:", distance_best[-1])

    # 距离迭代图
    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(1, len(distance_best) + 1), distance_best)
    plt.xlabel("迭代代数")
    plt.ylabel("最短回路距离值")
    plt.show()

    # 画路线图
    plt.figure(2)
    path = list(path_best[-1]+1)
    path.append(1.0)
    drawFigure(data_list, path, pos_ctrl)

if __name__ == '__main__':
    # 初始化参数
    # pos_ctrl取1或2，与所选数据有关
    data_path = "B题附件/城市航线.txt"
    # data_path = "B题附件/城市航线_新_1.txt"
    pos_ctrl = 2

    # 导入数据
    data_list = loadData(data_path)

    # 转化为GraphDictionary形式
    graph_mat = transform2matrix(data_list)

    # ACO
    ACO(graph_mat, data_list, pos_ctrl)

