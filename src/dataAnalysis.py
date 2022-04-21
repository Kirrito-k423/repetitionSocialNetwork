from datetime import date
from inspect import Parameter
from platform import node
from tokenize import group
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.profiler import profile
from rich.progress import track
from data import DataBase
import pickle
import sys
from tsjPython.tsjCommonFunc import *

# 需要入门 PyTorch Geometric
# 不介意可以看我写的 http://home.ustc.edu.cn/~shaojiemike/posts/pytorchgeometric
nodeNum = 1000
edgeNum = 0  # 保存的是两倍的数量
topicNum = 0
groupNum = 0
batchSize = 32
N_EPOCHS = 500
echo2Print = 1
threshold = 0.5
trainLR = 0.0001


def groupMemberNum(dataset, batch_size, groupName):
    groupNumDict = dict()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for id_batch, (trainGroup_batch, label_batch) in enumerate(dataloader):
        tmpGroupNum = 0
        # print(label_batch.size())
        for i in range((label_batch.size())[1]):
            if label_batch[0][i] == 1:
                tmpGroupNum += 1
        if tmpGroupNum in groupNumDict:
            groupNumDict[tmpGroupNum] += 1
        else:
            groupNumDict[tmpGroupNum] = 1
    passPrint(groupName)
    print("group的成员数：group个数")
    pPrint(groupNumDict)


def memberJoinGroupNum(dataset, batch_size, groupName):
    groupNumDict = {}
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for id_batch, (trainGroup_batch, label_batch) in enumerate(dataloader):
        valuePrint(label_batch.size())
        label_batch = torch.sum(label_batch, 0)
        valuePrint(label_batch.size())
        # print(label_batch.size())
        for i in range(label_batch.size()[0]):
            tmpGroupNum = int(label_batch[i])
            if tmpGroupNum in groupNumDict:
                groupNumDict[tmpGroupNum] += 1
            else:
                groupNumDict[tmpGroupNum] = 1
    passPrint(groupName)
    print("每个人参加group数：人数")
    pPrint(groupNumDict)


def readDataFromDatabase(nodeNum):
    try:
        f = open("nodeNum_" + str(nodeNum) + ".tmp", "rb")
        # Do something with the file
        [
            dataset,
            prediction_dataset,
            edge_index,
            nodeNum,
            edgeNum,
            topicNum,
            groupNum,
            predictGroupNum,
        ] = pickle.load(f)
        f.close()
    except IOError:
        print("File nodeNum = {} not accessible".format(nodeNum))
        db = DataBase()
        [
            dataset,
            prediction_dataset,
            edge_index,
            nodeNum,
            edgeNum,
            topicNum,
            groupNum,
            predictGroupNum,
        ] = db.exampleDataFrom(
            nodeNum, percent=0.8, simple_topics=True, use_top_city=True, citynum=1,min_group_num=3
        )
        f = open("nodeNum_" + str(nodeNum) + ".tmp", "wb")
        pickle.dump(
            [
                dataset,
                prediction_dataset,
                edge_index,
                nodeNum,
                edgeNum,
                topicNum,
                groupNum,
                predictGroupNum,
            ],
            f,
        )
        f.close()
    return [
        dataset,
        prediction_dataset,
        edge_index,
        nodeNum,
        edgeNum,
        topicNum,
        groupNum,
        predictGroupNum,
    ]


def main():
    global nodeNum, edgeNum, topicNum, groupNum
    args = sys.argv[1:]
    if len(args) == 1:
        nodeNum = args[0]
    [
        dataset,
        prediction_dataset,
        edge_index,
        nodeNum,
        edgeNum,
        topicNum,
        groupNum,
        predictGroupNum,
    ] = readDataFromDatabase(nodeNum)
    print(
        "nodeNum,edgeNum,topicNum,groupNum predictGroupNum {} {} {} {} {}".format(
            nodeNum, edgeNum, topicNum, groupNum, predictGroupNum
        )
    )
    # print(dataset)
    groupMemberNum(dataset, 1, "trainGroup")
    memberJoinGroupNum(dataset, groupNum, "trainGroup")
    groupMemberNum(prediction_dataset, 1, "predictGroup")
    memberJoinGroupNum(prediction_dataset, predictGroupNum, "predictGroup")


if __name__ == "__main__":
    main()
