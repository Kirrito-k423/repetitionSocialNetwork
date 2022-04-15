from torch.utils.tensorboard import SummaryWriter
import numpy as np


labels = np.random.randint(2, size=10)  # binary label
print(labels)
predictions = np.random.rand(10)
print(predictions)
writer = SummaryWriter()
writer.add_pr_curve("pr_curve", labels, predictions, 0)
writer.close()

# 增加了精确回忆曲线。绘制精度召回曲线可以让您了解模型在不同阈值设置下的性能。使用此功能，您可以为每个目标提供地面真值标记（T/F）和预测置信度（通常是模型的输出）。TensorBoard UI将允许您以交互方式选择阈值。
