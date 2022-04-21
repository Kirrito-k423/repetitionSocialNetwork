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
from dataAnalysis import readDataFromDatabase

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


class DSI(MessagePassing):
    def __init__(self, NodeNum, TopicNum, EdgeNum, device, BatchSize):
        super().__init__(aggr="mul")  # "mul" aggregation (Step 5).
        self.nodePrefferVector = nn.Parameter(
            torch.rand(NodeNum, TopicNum, requires_grad=True)
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.Wij = nn.Parameter(torch.rand(EdgeNum, 1, requires_grad=True))  # W_{ij}
        self.Sig = torch.nn.Sigmoid()
        self.device = device
        self.NodeNum = NodeNum
        self.TopicNum = TopicNum
        self.BatchSize = BatchSize

    def forward(self, trainGroup_batch, edge_index, threshold_H):
        # edge_index has shape [2, E]
        self.BatchSize = trainGroup_batch.size()[0]
        edge_index = BatchExpand(edge_index, self.BatchSize)
        edge_index, _ = remove_self_loops(edge_index)

        # print(trainGroup_batch)
        tmpCosInput = trainGroup_batch[0].expand(self.NodeNum, self.TopicNum)
        for i in range(self.BatchSize - 1):
            tmpCosInput = torch.cat(
                (
                    tmpCosInput,
                    trainGroup_batch[i + 1].expand(self.NodeNum, self.TopicNum),
                ),
                0,
            )

        expandNodePrefferVector = torch.cat(
            [self.nodePrefferVector] * self.BatchSize, 0
        )
        f = self.cos(tmpCosInput, expandNodePrefferVector)  # [nodeNum * 1]
        fMinusH = f - torch.cat([threshold_H] * self.BatchSize, 1)
        fMinusHtranspose = fMinusH.t()
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=fMinusHtranspose, f=f)

    def message(self, x_j, f):
        # x_j has shape [E, out_channels]
        L = self.Sig(x_j)
        tmpCat = torch.cat((L, torch.cat([self.Wij] * self.BatchSize)), 1)
        LW = torch.prod(tmpCat, 1)
        x_j = (1 - LW).view(1, -1).t()
        return x_j

    def update(self, aggr_out, f):
        # aggr_out has shape [N, out_channels]
        ht = aggr_out.t()
        return [self.Sig(f - ht), ht.detach()]


# def exampleDateFrom():
#     # PyG保存的有向图，因此有 4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
#     # [2, edgeNum]
#     edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

#     # trainGroup features
#     # [trainGroupNum , topicNum]
#     x = torch.tensor([[1, 2, 3], [0, 0, 0], [1, 0, 0]], dtype=torch.float,)

#     # label
#     # [trainGroupNum, nodeNum]
#     label = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float)
#     # Use torch.utils.data to create a DataLoader
#     # that will take care of creating batches
#     dataset = TensorDataset(x, label)
#     print(x.size())
#     return [dataset, edge_index]


def BatchExpand(edge_index, batch_size):
    tmp = edge_index
    for i in range(batch_size - 1):
        tmp = tmp + nodeNum
        edge_index = torch.cat((edge_index, tmp), 1)
    # print(edge_index)
    return edge_index


def meanBatchOut(batchOut):
    tmpList = torch.split(batchOut, nodeNum, 1)
    tmpSum = torch.zeros(1, nodeNum).cuda()
    for i in range(len(tmpList)):
        tmpSum += tmpList[i]
    return tmpSum / len(tmpList)


def trainNet(dataset, edge_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = DSI(nodeNum, topicNum, edgeNum, device, batchSize)
    net.to(device)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=trainLR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )

    # Print model's state_dict
    print("\nModel's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    print("\nparam.cuda()")
    # In pseudo code (this won't work with nested nn.Module
    with torch.no_grad():
        for name, param in net.named_parameters():
            print(param.cuda())

    # Print optimizer's state_dict
    print("\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    threshold_H = torch.rand(1, nodeNum, dtype=torch.float)
    # dataset.to(device)
    edge_index, threshold_H = (
        edge_index.to(device),
        threshold_H.to(device),
    )

    log_writer = SummaryWriter()
    # with SummaryWriter(comment="LeNet") as w:
    #     gpuTestData = dataset[0][0].to(device)
    #     w.add_graph(net, (gpuTestData, edge_index, threshold_H,))

    beforeLoss = 2333
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     with_flops=True,
    # ) as prof:
    for epoch in track(
        range(N_EPOCHS), total=N_EPOCHS, description="epoch"
    ):  # loop over the dataset multiple times
        # if epoch % 200 == 0:
        #     print(f"Epoch {epoch + 1}\n-------------------------------")

        # 由于loss是全体Group算一次，所以Batch大小为总大小。
        # 应该是不需要batch的
        testNum = 0
        correctNum = 0
        for id_batch, (trainGroup_batch, label_batch) in enumerate(dataloader):
            # trainGroup_batch = trainGroup_batch.reshape(1, topicNum * batchSize)
            trainGroup_batch = trainGroup_batch.to(device)
            # print(label_batch.size())
            label_batch = label_batch.reshape(1, nodeNum * label_batch.size()[0])
            label_batch = label_batch.to(device)
            # print("222222222", flush=True)
            # print(trainGroup_batch, flush=True)
            optimizer.zero_grad()
            [predict, tmpthreshold_H] = net(trainGroup_batch, edge_index, threshold_H)
            threshold_H = meanBatchOut(tmpthreshold_H)
            loss = criterion(predict, label_batch)
            loss.backward(retain_graph=True)
            optimizer.step()  # Does the update
            batchTestNum = (label_batch.size())[1]
            testNum += batchTestNum
            for i in range(batchTestNum):
                if label_batch[0][i] == (predict[0][i] > threshold):
                    correctNum += 1
                if label_batch[0][i] == 1:
                    positiveLabelNum += 1
                elif label_batch[0][i] == 0:
                    negativeLabelNum += 1
        if epoch % echo2Print == 0:
            print("epoch: %d, loss: %f" % (epoch, loss))
        # print(threshold_H.size())
        # print(label.size())
        # print(list(range(1, nodeNum)))

        # label_img = torch.rand(nodeNum, 3, 10, 10)
        # log_writer.add_embedding(
        #     threshold_H,
        #     tag="threshold_H",
        #     metadata=list(range(1, nodeNum + 1)),
        #     label_img=label_img,
        #     global_step=epoch,
        # )
        log_writer.add_scalar("Loss/train", float(loss), epoch)
        log_writer.add_scalar("accuracy", correctNum / testNum, epoch)
        # print(predict.size())
        # print(label_batch.size())
        log_writer.add_pr_curve("pr_curve", label_batch, predict, epoch)
        predict01 = torch.rand(predict.size()[0], predict.size()[1])
        for i in range(predict.size()[0]):
            for j in range(predict.size()[1]):
                predict01[i][j] = predict[i][j] > threshold
        # print(predict.size())
        log_writer.add_pr_curve("pr_curve_01", label_batch, predict01, epoch)
        if abs(beforeLoss - loss) < 10e-7:
            break
        beforeLoss = loss
    # print(prof.table())
    # prof.export_chrome_trace("torch_trace.json")
    # prof.export_stacks("torch_cpu_stack.json", metric="self_cpu_time_total")
    # prof.export_stacks("torch_cuda_stack.json", metric="self_cuda_time_total")
    torch.save(net.state_dict(), "./saveNet/save.pt")
    torch.save(threshold_H, "./saveNet/Htensor.pt")


def testNet(dataset, edge_index, gpuDevice):
    # return 0
    device = torch.device(gpuDevice if torch.cuda.is_available() else "cpu")
    print(device)

    net = DSI(nodeNum, topicNum, edgeNum, device, batchSize)
    net.load_state_dict(torch.load("./saveNet/save.pt"))
    net.to(device)
    net.eval()
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    threshold_H = torch.load("./saveNet/Htensor.pt")
    edge_index, threshold_H = (
        edge_index.to(device),
        threshold_H.to(device),
    )
    testNum = 0
    correctNum = 0
    positiveLabelNum = 0
    negativeLabelNum = 0
    for id_batch, (trainGroup_batch, label_batch) in enumerate(dataloader):
        # trainGroup_batch = trainGroup_batch.reshape(1, topicNum * batchSize)
        trainGroup_batch = trainGroup_batch.to(device)
        # print(label_batch.size())
        label_batch = label_batch.reshape(1, nodeNum * label_batch.size()[0])
        label_batch = label_batch.to(device)
        # print("222222222", flush=True)
        # print(trainGroup_batch, flush=True)
        [predict, tmpthreshold_H] = net(trainGroup_batch, edge_index, threshold_H)
        threshold_H = meanBatchOut(tmpthreshold_H)
        batchTestNum = (label_batch.size())[1]
        testNum += batchTestNum
        print(testNum)
        # print(label_batch.size())
        # print(predict.size())
        for i in range(batchTestNum):
            if label_batch[0][i] == (predict[0][i] > threshold):
                correctNum += 1
            if label_batch[0][i] == 1:
                positiveLabelNum += 1
            elif label_batch[0][i] == 0:
                negativeLabelNum += 1
    print("test accuracy: %f" % (correctNum / testNum))
    print(
        "positiveLabelNum: %d negativeLabelNum: %d ratio: %f"
        % (positiveLabelNum, negativeLabelNum, positiveLabelNum / negativeLabelNum)
    )


def main():
    global nodeNum, edgeNum, topicNum, groupNum
    args = sys.argv[1:]
    # args is a list of the command line args
    if args[0] == "predict":
        skip_training = 1
        skip_predict = 0
    elif args[0] == "skip":
        skip_training = 1
        skip_predict = 1
    else:
        skip_training = 0
        skip_predict = 0

    if len(args) == 1:
        gpuDevice = args[1]
    elif len(args) == 2:
        gpuDevice = args[1]
        nodeNum=int(args[2])
    else:
        gpuDevice = "cuda:0"
   
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
    if skip_training == 0:
        print("train")
        # trainNet(dataset, edge_index)
    else:
        print("skip train")
    if skip_predict == 0:
        print("predict")
        testNet(prediction_dataset, edge_index, gpuDevice)
    else:
        print("skip predict")


if __name__ == "__main__":
    main()
