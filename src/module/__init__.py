# TODO: 调试adapter等PEFT，使其效果提升。或者可以调大模型参数量。
# TODO: T5配置YOFO，为什么sparse下不去？
# TODO: MoEYOFO
# TODO: YOFOの对比学习
# TODO: Conditioned YOFO
# TODO: YOFO收敛速度与分类不一致，间接导致YOFO准确率不高
import torch.nn

torch.nn.BatchNorm1d