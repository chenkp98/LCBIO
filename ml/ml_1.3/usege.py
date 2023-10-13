#!/usr/bin/env python
# coding: utf-8

import argparse

def usage():
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="Description：机器学习在高通量测序中的应用")

    # 添加命令行参数选项
    parser.add_argument("-p", "--parameters", required=True, help="配置文件路径，必需选项")
    parser.add_argument("-f", "--file_type", required=False, help="分析数据文件路径，必须选项，支持文件格式csv,txt,excel,pickle")
    parser.add_argument("-t", "--test_size", type=float, required=False, default=0.2, help="训练集和测试集的划分比例，默认为0.2")
    parser.add_argument("-r", "--random_state", type=int, required=False, default=420, help="随机种子，保证每次运行的结果一样，默认为420")
    parser.add_argument("-s", "--select", type=bool, required=True, default=True, help="是否进行特征选择降维，默认为是")
    parser.add_argument("-w", "--min_child_weight",type=float, required=False, default=1, help="决定叶子节点继续划分的最小样本权重总和，默认为1")
    parser.add_argument("-d", "--max_depth", required=False, default=6, help="决定每个树的最大深度，默认为6")
    parser.add_argument("-a", "--reg_alpha", required=False, help="L1正则化项的权重")
    parser.add_argument("-l", "--reg_lambda", required=False, help="L2正则化项的权重")
    parser.add_argument("-k", "--kernel", required=False, default='rbf', help="支持向量机核函数，默认为高斯核函数")
    parser.add_argument("-c", "--C", type=float, required=False,help="正则化参数，控制了分类错误的惩罚程度")
    parser.add_argument("Version", action="store_true", help="1.3")
    parser.add_argument("Contact", action="store_true", help="1018849303@qq.com")
    parser.add_argument("Last Update",  action="store_true", help="10/12/2023")


    # 解析命令行参数
    args = parser.parse_args()

    # 获取解析后的参数值
    parameters = args.parameters

if __name__ == "__main__":
    usage()