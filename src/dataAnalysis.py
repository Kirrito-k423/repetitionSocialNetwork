from datetime import date
from inspect import Parameter
from platform import node
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
from pprint import pprint

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


def main():
    global nodeNum, edgeNum, topicNum, groupNum
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
        ] = db.exampleDataFrom(nodeNum, percent=0.8, simple_topics=True)
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

    print(
        "nodeNum,edgeNum,topicNum,groupNum predictGroupNum {} {} {} {} {}".format(
            nodeNum, edgeNum, topicNum, groupNum, predictGroupNum
        )
    )
    print(dataset)
    groupNumDict = dict()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
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
    print("group的成员数：group个数")
    pprint(groupNumDict)

    groupNumDict = {}
    dataloader = DataLoader(dataset, batch_size=groupNum, shuffle=True)
    for id_batch, (trainGroup_batch, label_batch) in enumerate(dataloader):
        print(label_batch.size())
        label_batch = torch.sum(label_batch, 0)
        print(label_batch.size())
        # print(label_batch.size())
        for i in range(label_batch.size()[0]):
            tmpGroupNum = int(label_batch[i])
            if tmpGroupNum in groupNumDict:
                groupNumDict[tmpGroupNum] += 1
            else:
                groupNumDict[tmpGroupNum] = 1
    print("每个人参加group数：人数")
    pprint(groupNumDict)


if __name__ == "__main__":
    main()
