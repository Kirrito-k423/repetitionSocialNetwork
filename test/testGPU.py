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

# 需要入门 PyTorch Geometric
# 不介意可以看我写的 http://home.ustc.edu.cn/~shaojiemike/posts/pytorchgeometric
nodeNum = 3
edgeNum = 2  # 无向边就是4
topicNum = 3
groupNum = 2
batchSize = 1
N_EPOCHS = 10000


class DSI(MessagePassing):
    def __init__(self, NodeNum, TopicNum, EdgeNum, device):
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

    def forward(self, trainGroup_batch, edge_index, threshold_H):
        # edge_index has shape [2, E]

        edge_index, _ = remove_self_loops(edge_index)

        # batchInputNum = (trainGroup_batch.size())[0]
        # for i in range(batchInputNum):
        i = 0
        tmpCosInput = trainGroup_batch.expand(self.NodeNum, self.TopicNum)
        # tmpCosInput = tmpCosInput.to(self.device)
        print(tmpCosInput)
        print(trainGroup_batch)
        # print(self.nodePrefferVector)
        f = self.cos(tmpCosInput, self.nodePrefferVector)  # [nodeNum * 1]
        fMinusH = f - threshold_H
        # 因为只有一个所以没有
        fMinusHtranspose = fMinusH.t()
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=fMinusHtranspose, f=f)

    def message(self, x_j, f):
        # x_j has shape [E, out_channels]
        L = self.Sig(x_j)
        tmpCat = torch.cat((L, self.Wij), 1)
        LW = torch.prod(tmpCat, 1)
        x_j = (1 - LW).view(1, -1).t()
        return x_j

    def update(self, aggr_out, f):
        # aggr_out has shape [N, out_channels]
        # f = self.lin(x)
        ht = aggr_out.t()
        return [self.Sig(f - ht), ht.detach()]


def exampleDateFrom():
    # 由于是无向图，因此有 4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
    # [2, edgeNum]
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    # trainGroup features
    # [trainGroupNum , topicNum]
    x = torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.float,)

    # label
    # [trainGroupNum, nodeNum]
    label = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float)
    # Use torch.utils.data to create a DataLoader
    # that will take care of creating batches
    dataset = TensorDataset(x, label)
    print(x.size())
    return [dataset, edge_index]


def trainNet(dataset, edge_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = DSI(nodeNum, topicNum, 2 * edgeNum, device)
    # net = DSI(nodeNum, topicNum, device)
    net.to(device)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # In pseudo code (this won't work with nested nn.Module
    with torch.no_grad():
        for name, param in net.named_parameters():
            print(param.cuda())
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
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
    with SummaryWriter(comment="LeNet") as w:
        w.add_graph(net, (dataset[0][0], edge_index, threshold_H,))

    beforeLoss = 2333
    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # 由于loss是全体Group算一次，所以Batch大小为总大小。
        # 应该是不需要batch的
        for id_batch, (trainGroup_batch, label_batch) in enumerate(dataloader):
            trainGroup_batch = trainGroup_batch.to(device)
            label_batch = label_batch.to(device)
            print("222222222", flush=True)
            print(trainGroup_batch, flush=True)
            optimizer.zero_grad()
            [predict, threshold_H] = net(trainGroup_batch, edge_index, threshold_H)
            loss = criterion(predict, label_batch)
            loss.backward(retain_graph=True)
            optimizer.step()  # Does the update
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
            # print(predict.size())
            # print(label_batch.size())
            log_writer.add_pr_curve("pr_curve", label_batch, predict, epoch)
            if abs(beforeLoss - loss) < 10e-7:
                break
            beforeLoss = loss
    torch.save(net.state_dict(), "./saveNet/save.pt")
    torch.save(threshold_H, "./saveNet/Htensor.pt")


def testNet(dataset, edge_index):
    return 0
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # net = DSI(nodeNum, topicNum)
    # net.load_state_dict(torch.load("./saveNet/save.pt"))
    # net.to(device)
    # net.eval()
    # threshold_H = torch.load("./saveNet/Htensor.pt")
    # x, edge_index, label, threshold_H = (
    #     x.to(device),
    #     edge_index.to(device),
    #     label.to(device),
    #     threshold_H.to(device),
    # )
    # [predict, threshold_H2] = net(x, edge_index, threshold_H)
    # testNum = (label.size())[0]
    # correctNum = 0
    # for i in range(testNum):
    #     if label[i][0] == (predict[i][0] > 0.5):
    #         correctNum += 1
    # print("test accuracy: %f" % (correctNum / testNum))


def main():
    [dataset, edge_index] = exampleDateFrom()
    trainNet(dataset, edge_index)
    # testNet(dataset, edge_index)


if __name__ == "__main__":
    main()
