# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q4_2_AddNewPaths.py
@Date: 2022/6/1
"""
import pandas as pd
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

# 画航线图
def drawFigure(data_list, new_path):
    # 画原始航线图(G1)、新增航线图(G2)
    # 转化并记录待标记路线
    path_num = len(new_path)
    # 创建图对象
    G1 = nx.Graph()
    G2 = nx.Graph()
    # 增加节点
    G1.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    G2.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    # 增加权重，数据格式（节点1，节点2，权重）
    for i in range(0, len(data_list), 3):
        G1.add_edge(data_list[i], data_list[i + 1], weight=data_list[i + 2])
    for i in range(path_num):
        G2.add_edge(new_path[i][0], new_path[i][1], weight=new_path[i][2])

    # 固定节点位置
    pos = {
        1: [13, 11],
        2: [6, 11],
        3: [8, 0.5],
        4: [1, 7],
        5: [1, 3],
        6: [11, 1.5],
        7: [13, 6],
        8: [6, 5]
    }
    # 重新获取权重序列
    weightsG1 = nx.get_edge_attributes(G1, "weight")
    weightsG2 = nx.get_edge_attributes(G2, "weight")
    # 画节点图、路径
    nx.draw_networkx(G1, pos, with_labels=True, node_color='#dfd7d7', edge_color='#6b5152', width=2.0)
    nx.draw_networkx(G2, pos, with_labels=True, node_color='#dfd7d7', edge_color='#0000FF')
    # 画权重图
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=weightsG1, font_size=8)
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=weightsG2, font_size=8)
    # 展示
    plt.show()

# 导出数据
def writeData(data_path_out, data_list, new_path):
    # 新建列表
    write_data = []
    write_headers = ['%数据集：八个城市之间的航线数据', '%作者：cinja', '%格式：起点，终点，距离/km']

    # 处理data_list
    for i in range(0, len(data_list), 3):
        one_str = str(data_list[i]) + ' ' + str(data_list[i+1]) + ' ' + str(data_list[i+2])
        write_data.append(one_str)

    # 处理new_path
    for i in new_path:
        one_str = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2])
        write_data.append(one_str)

    # 合并列表
    write_data = write_headers + write_data

    # 写数据
    with open(data_path_out, "w", encoding='utf-8') as f:
        for i in write_data:
            f.writelines(i)
            f.write('\n')


if __name__ == "__main__":
    # 初始化参数
    distance_path = 'Results/Distances.xls'
    data_path = "B题附件/城市航线_时间.txt"
    data_path_out = "B题附件/城市航线_新_2.txt"
    CITY_NUM = 8

    # 导入数据
    distance_mat = pd.read_excel(distance_path, index_col=0)
    data_list = loadData(data_path)

    # 新增航线
    new_path = [[1, 3, 2.22], [5, 7, 2.22]]

    # 画航线图
    drawFigure(data_list, new_path)

    # 导出新航线数据
    # writeData(data_path_out, data_list, new_path)