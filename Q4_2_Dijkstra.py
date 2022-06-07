# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q4_2_Dijkstra.py
@Date: 2022/6/1
"""
import networkx as nx
import matplotlib.pyplot as plt

# 导入数据
def loadData(data_path):
    with open(data_path, "r", encoding='utf-8') as f:
        data = f.read()
        list = data.split()
        data_list = []
        for one in list:
            try:
                trans = int(one)
                data_list.append(trans)
            except:
                try:
                    trans = float(one)
                    data_list.append(trans)
                except:
                    continue
        # print(data_list)
    return data_list

# 转化为邻接矩阵
def transform2matrix(data_list):
    graph_dic = {
        '1':{},
        '2':{},
        '3':{},
        '4':{},
        '5':{},
        '6':{},
        '7':{},
        '8':{}
    }
    for i in range(0, len(data_list), 3):
        key1 = str(data_list[i])
        key2 = str(data_list[i+1])
        value = data_list[i+2]
        graph_dic[key1].update({key2:value})
        graph_dic[key2].update({key1:value})
    return graph_dic

# Dijkstra算法
def dijkstra(graph, start, end):
    # 初始化距离矩阵
    distances = {}
    predecessors = {}
    # 获取图字典中的键
    nodes = graph.keys()
    for node in graph:
        distances[node] = float('inf')
        predecessors[node] = None
    # 新建已访问节点列表
    visited_list = []
    # 设置初始距离为0
    distances[start] = 0

    while len(visited_list) < len(nodes):
        # 记录剩余未访问点
        remain = {node: distances[node] for node in [node for node in nodes if node not in visited_list]}
        # 判断是否有更短距离
        closest = min(remain, key=distances.get)
        # 更新已访问节点列表
        visited_list.append(closest)
        for node in graph[closest]:
            if node != start or node != end:
                transfer_index = 1
            else:
                transfer_index = 0
            if distances[node] > distances[closest] + graph[closest][node] + TRANSFER_TIME * transfer_index:
                distances[node] = distances[closest] + graph[closest][node] + TRANSFER_TIME * transfer_index
                predecessors[node] = closest
    path = [end]
    while start not in path:
        path.append(predecessors[path[-1]])

    distances[end] -= TRANSFER_TIME
    return path[::-1], distances[end]

# 打印结果
def printResults(start, end):
    path_print = start
    for i in path[1:-1]:
        path_print = path_print + '>>' + str(i)
    path_print = path_print + '>>' + end
    print('从起点', start, '到终点', end, '的最短路径为:', path_print)
    print('总耗时为:', distance)

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


if __name__ == '__main__':
    # 初始化参数
    # pos_ctrl取1或2，与所选数据有关
    data_path = "B题附件/城市航线_时间.txt"
    pos_ctrl = 1
    start = '5'
    end = '7'
    TRANSFER_TIME = 1.5

    # 导入数据
    data_list = loadData(data_path)

    # 转化为GraphDictionary形式
    graph_dic = transform2matrix(data_list)

    # Dijkstra算法
    path, distance = dijkstra(graph_dic, start=start, end=end)

    # 打印结果
    printResults(start, end)

    # 画路线图
    drawFigure(data_list, path, pos_ctrl)
