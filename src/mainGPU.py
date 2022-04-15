from datetime import date
from inspect import Parameter
from platform import node
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn as nn
from tensorboardX import SummaryWriter

# 需要入门 PyTorch Geometric
# 不介意可以看我写的 http://home.ustc.edu.cn/~shaojiemike/posts/pytorchgeometric
nodeNum = 3
edgeNum = 2  # 无向边就是4
topicNum = 3
groupNum = 2  # groupNum = BatchSize
BatchSize = 2


class DSI(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mul")  # "mul" aggregation (Step 5).
        self.lin = torch.nn.Linear(
            in_channels, out_channels, bias=False
        )  # [topicNum*BatchSize,BatchSize]
        self.message_lin = torch.nn.Linear(
            out_channels, out_channels, bias=False
        )  # W_{ij}
        self.Sig = torch.nn.Sigmoid()

    def forward(self, x, edge_index, threshold_H):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = remove_self_loops(edge_index)

        x = x.view(nodeNum, topicNum * groupNum)
        print(x.size())
        f = self.lin(x)
        L = self.Sig(f - threshold_H)
        print(L.size())
        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=L, f=f)

    def message(self, x_j, f):
        # x_j has shape [E, out_channels]
        return self.message_lin(x_j)

    def update(self, aggr_out, f):
        # aggr_out has shape [N, out_channels]
        # f = self.lin(x)
        return [self.Sig(f - aggr_out), aggr_out.detach()]


def exampleDateFrom():
    # 由于是无向图，因此有 4 条边：(0 -> 1), (1 -> 0), (1 -> 2), (2 -> 1)
    # [2, edgeNum]
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    # 节点的特征
    # [nodeNum , topicNum*groupNum]
    x = torch.tensor(
        [[[0, 1], [1, 1], [0, 0]], [[0, 1], [1, 1], [0, 1]], [[0, 1], [1, 1], [1, 0]]],
        dtype=torch.float,
    )

    # label
    # [nodeNum, groupNum]
    label = torch.tensor([[1, 1], [0, 0], [1, 0]], dtype=torch.float)
    print(x.size())
    return [x, edge_index, label]


def trainNet(x, edge_index, label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = DSI(topicNum * BatchSize, BatchSize)
    net.to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )
    threshold_H = torch.rand(nodeNum, 1, dtype=torch.float)
    x, edge_index, label, threshold_H = (
        x.to(device),
        edge_index.to(device),
        label.to(device),
        threshold_H.to(device),
    )
    log_writer = SummaryWriter()
    with SummaryWriter(comment="LeNet") as w:
        w.add_graph(net, (x, edge_index, threshold_H,))
    beforeLoss = 2333
    for epoch in range(10000):  # loop over the dataset multiple times
        # 由于loss是全体Group算一次，所以Batch大小为总大小。
        # 应该是不需要batch的
        optimizer.zero_grad()
        [predict, threshold_H] = net(x, edge_index, threshold_H)
        loss = criterion(predict, label)
        loss.backward(retain_graph=True)
        optimizer.step()  # Does the update
        print("epoch: %d, loss: %f" % (epoch, loss))
        log_writer.add_scalar("Loss/train", float(loss), epoch)
        if abs(beforeLoss - loss) < 10e-7:
            break
        beforeLoss = loss


def main():
    [x, edge_index, label] = exampleDateFrom()
    trainNet(x, edge_index, label)


if __name__ == "__main__":
    main()
