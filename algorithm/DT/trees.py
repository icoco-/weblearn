# coding=utf-8

from math import log
import operator
import treePlotter


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return data_set, labels


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# myDat, labels = create_data_set()
# myDat[0][-1] = 'maybe'
# print calc_shannon_ent(myDat)


# data_set 待划分的数据集
# axis 待划分的数据特征
# value 特征的返回值
def split_data_set(data_set, axis, value):
    ret_data_set = []  # 避免修改数据集
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


# extend append
a = [1, 2, 3]
b = [4, 5, 6]
# a.append(b)
# a.extend(b)
# print a

# split_data test
myDat, labels = create_data_set()


# print split_data_set(myDat, 0, 0)
# print split_data_set(myDat, 0, 1)

# 计算信息增益最大的特征
def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]  # create a list of all the examples of this feature
        unique_vals = set(feat_list)  # get a set of unique values
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)  # 获取所有feature[i]=value 的数据
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature


print choose_best_feature_to_split(myDat)


# 返回出现次数最多的分类名称
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


#
def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]  # 分类结果，yes no
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]  # stop splitting when all of the classes are equal
    if len(data_set[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majority_cnt(class_list)  # 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组，这里返回出现次数最多的类
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


# print create_tree(myDat, labels)


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

myTree = create_tree(myDat, ['no surfacing', 'flippers'])
print classify(myTree, labels, [1, 0])


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# 隐形眼镜测试数据
fr = open('data/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = create_tree(lenses, lensesLabels)
# print lensesTree
# treePlotter.createPlot(lensesTree)


