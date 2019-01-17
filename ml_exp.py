# coding=utf-8
import pandas as pd
from math import log2
import random

positive_value = 'recurrence-events'
negative_value = 'no-recurrence-events'
unknown_value = '?'


class Node:
    def __init__(self):
        self.decision_attribute = ''    # 决策属性
        self.test_branch = {}           # 分支测试 {决策属性值：子树} 字典
        self.leaf_label = '非叶结点'    # 叶结点标签


def print_tree(root, key_value_string, target_attribute):
    if root.leaf_label != '非叶结点':
        if root.leaf_label:
            print(key_value_string + '\b\b\b\b' + ' → ' + '[' + target_attribute + ' = ' + positive_value + ']')
        else:
            print(key_value_string + '\b\b\b\b' + ' → ' + '[' + target_attribute + ' = ' + negative_value + ']')
    if root.leaf_label == '非叶结点':
        for v in root.test_branch.keys():
            tmp = key_value_string + '[' + root.decision_attribute + ' = ' + str(v) + '] ' + ' ∧ '
            print_tree(root.test_branch[v], tmp, target_attribute)
    return


def entropy(s, target_attribute):
    # 计算信息熵
    if s.empty:
        return 0
    ent = 0
    s_size = len(s)
    classes = s[target_attribute].unique()
    for i in classes:
        pi = (s[target_attribute] == i).sum()/s_size
        if pi != 0:
            ent += -pi * log2(pi)
    return ent


def information_gain(s, target_attribute, attribute):
    # 计算信息增益
    s_size = len(s)                            # |S|
    entropy_s = entropy(s, target_attribute)   # Entropy(S)
    values = s[attribute].unique()
    weighted_entropy_summary = 0
    for v in values:                           # ∑
        s_v = s[s[attribute] == v]             # Sv
        s_size_v = len(s_v)                    # |Sv|
        entropy_s_v = entropy(s_v, target_attribute)
        weighted_entropy_summary += s_size_v * entropy_s_v / s_size
    return entropy_s - weighted_entropy_summary


def id3_build_tree(examples, target_attribute, attributes):
    # 使用ID3算法构建决策树
    root = Node()
    if (examples[target_attribute] == positive_value).all():
        root.leaf_label = True
        return root
    if (examples[target_attribute] == negative_value).all():
        root.leaf_label = False
        return root
    if not attributes:
        root.leaf_label = examples[target_attribute].mode()[0] == positive_value
        return root
    ig = []
    for attribute in attributes:
        ig.append(information_gain(examples, target_attribute, attribute))
    a = attributes[ig.index(max(ig))]
    # print(a)
    root.decision_attribute = a
    values = examples[a].unique()
    for vi in values:
        examples_vi = examples[examples[a] == vi]
        if examples.empty:
            new_node = Node()
            new_node.leaf_label = examples[target_attribute].mode()[0] == positive_value
        else:
            new_node = id3_build_tree(examples_vi, target_attribute, [i for i in attributes if i != a])
        if vi != unknown_value:
            root.test_branch.update({vi: new_node})
    return root


def id3_prune(examples, target_attribute, root, tree):
    # 对决策树进行剪枝
    if root.leaf_label != '非叶结点':
        return root
    for v in root.test_branch.keys():
        root.test_branch[v] = id3_prune(examples, target_attribute, root.test_branch[v], tree)
    old_correctness = id3_correctness(tree, examples, target_attribute)
    root.leaf_label = examples[target_attribute].mode()[0] == positive_value
    new_correctness = id3_correctness(tree, examples, target_attribute)
    if new_correctness > old_correctness:
        return root
    else:
        root.leaf_label = '非叶结点'
        return root


def id3_classify(root, example):
    # 使用ID3算法得到的决策树对样本进行分类
    while root.leaf_label == '非叶结点':
        if example[root.decision_attribute] in root.test_branch:
            root = root.test_branch[example[root.decision_attribute]]
        elif example[root.decision_attribute] == unknown_value:
            root = root.test_branch[random.choice(list(root.test_branch))]
        else:
            return '拒分'
    if root.leaf_label:
        return positive_value
    else:
        return negative_value


def id3_correctness(root, example_test, target_attribute):
    # 计算ID3算法在测试集上的准确度
    test_size = len(example_test)
    correct = 0
    for i in range(test_size):
        real_result = example_test.loc[i, target_attribute]
        id3_result = id3_classify(root, example_test.loc[i, :])
        if real_result == id3_result:
            correct += 1
    return correct/test_size


def divide_train_test(cancer, fraction, fraction_train):
    # 以fraction的比例划分训练，测试集
    # 以fraction_train的比例划分训练集中用于建树和用于剪枝的部分
    threshold = int(fraction * len(cancer))
    threshold_train = int(fraction_train * threshold)
    train1 = cancer.loc[0: threshold_train, :]
    train1 = train1.reset_index(drop=True)
    train2 = cancer.loc[threshold_train: threshold, :]
    train2 = train2.reset_index(drop=True)
    test = cancer.loc[threshold: len(cancer), :]
    test = test.reset_index(drop=True)
    # print(test)
    return train1, train2, test


def save_tree(tree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(tree, fw)
    fw.close()


def load_tree(filename):
    import pickle
    fr = open(filename, 'rb')
    tree = pickle.load(fr)
    fr.close()
    return tree


target_attribute = 'Class'                                                            # 设置目标属性值
attributes = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
          'deg-malig', 'breast', 'breast-quad', 'irradiat']                           # 设置属性值
breast_cancer = pd.read_csv('breast-cancer.data', names=attributes).sample(frac=1)    # sample方法打乱数据集
breast_cancer = breast_cancer.reset_index(drop=True)
breast_cancer_train1, breast_cancer_train2, breast_cancer_test = divide_train_test(breast_cancer, 1, 0.6)

attributes.remove(target_attribute)                                                   # 获得非目标属性值

id3_tree = id3_build_tree(breast_cancer_train1, target_attribute, attributes)         # 建立决策树
id3_tree = id3_prune(breast_cancer_train2, target_attribute, id3_tree, id3_tree)      # 剪枝
print_tree(id3_tree, '', target_attribute)                                            # 打印决策树
save_tree(id3_tree, 'id3_tree.tree')                                                  # 将决策树保存为二进制文件
print('train1: ', len(breast_cancer_train1), 'train2: ', len(breast_cancer_train2), 'test: ', len(breast_cancer_test))
print('分类准确度', id3_correctness(id3_tree, breast_cancer_train2, target_attribute))   # 计算准确度   最高：0.8275862068965517
