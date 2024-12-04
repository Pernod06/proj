import torch
import torch.nn as nn
import math

from caffe2.experiments.python.device_reduce_sum_bench import SumElements


# 位置编码
class PositionalEncoder(nn.Module):
    '''d_model 模型维度'''
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        # 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000**((2*i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000*((2*(i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一点
        x = x * math.sqrt(self.d_model)

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x