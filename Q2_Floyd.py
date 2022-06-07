# -*- coding: utf-8 -*-
"""
@Author: Cinja
@Software: PyCharm
@File: Q2_Floyd.py
@Date: 2022/5/13
"""
import numpy as np
import pandas as pd

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
            if i != j and graph_mat[i, j] == 0:
                graph_mat[i, j] = 99999
    return graph_mat

# 打印邻接矩阵
def printMatrix(matrix):
    for row in matrix:
        print(row)

# Floyd算法
def floyd(graph_mat, print_ctrl):
    distance_mat = graph_mat.copy()
    n = len(distance_mat[0])
    for k in range(n):
        if print_ctrl == 1:
            print('k=%d' % (k))
            printMatrix(distance_mat)
            print('-'*10)
        for i in range(n):
            for j in range(n):
                distance_mat[i][j] = min([distance_mat[i][j], distance_mat[i][k]+distance_mat[k][j]])
    print("各节点间的最短路径为：")
    printMatrix(distance_mat)

    return distance_mat


if __name__ == '__main__':
    # 初始化参数
    # save_ctrl为1时，保存距离矩阵结果
    # print_ctrl为1时，打印每一次迭代的距离矩阵
    data_path = "B题附件/城市航线.txt"
    # data_path = "B题附件/城市航线_新_1.txt"
    save_ctrl = 0
    print_ctrl = 1

    # 导入数据
    data_list = loadData(data_path)

    # 转化为GraphDictionary形式
    graph_mat = transform2matrix(data_list)

    # Floyd算法
    path_matrix = floyd(graph_mat, print_ctrl)

    # 保存距离矩阵结果
    if save_ctrl == 1:
        save_df = pd.DataFrame(path_matrix)
        # save_df.to_excel('Results/Q2_Floyd_Distances.xls')
        save_df.to_excel('Results/Q4_1_Floyd_Distances.xls')

