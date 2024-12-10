import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F


class CAE(nn.Module):
    """
    全连接自编码器
    """

    def __init__(self, dropout_prob=0.2):
        super().__init__()
        # 卷积层1
        self.conv1_enc = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(dropout_prob)  # 加入 Dropout
        )
        self.pool1_enc = nn.MaxPool2d(2, 2, return_indices=True)

        # 卷积层2
        self.conv2_enc = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Dropout(dropout_prob)  # 加入 Dropout
        )
        self.pool2_enc = nn.MaxPool2d(2, 2, return_indices=True)
        # self.attention = selfAttention(in_channels=256)
        # 卷积层3
        self.conv3_enc = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU()
        )
        self.pool3_enc = nn.MaxPool2d(2, 2, return_indices=True)

        # 隐层再提取一次特征
        self.hidden = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # 全连接层
        self.fc1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout 在 fc1 和 fc2 之间

        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout 在 fc2 和 fc3 之间

        self.fc3 = nn.Linear(128, 3)
        self.dropout3 = nn.Dropout(dropout_prob)  # Dropout 在 fc3 和 fc4 之间

        self.fc4 = nn.Linear(3, 128)

        self.fc5 = nn.Linear(128, 512)

        self.fc6 = nn.Linear(512,1024)

        # 上采样层3
        self.pool3_dec = nn.MaxUnpool2d(2, 2)
        self.conv3_dec = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # 上采样层2
        self.pool2_dec = nn.MaxUnpool2d(2, 2)
        self.conv2_dec = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # 上采样层1
        self.pool1_dec = nn.MaxUnpool2d(2, 2)
        self.conv1_dec = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.Tanh(),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1_enc(x)
        x, indices1 = self.pool1_enc(x)
        x = self.conv2_enc(x)
        x, indices2 = self.pool2_enc(x)
        # attention=selfAttention(in_channels=x.shape[1])
        # if x.device.type == 'cuda':
        #     attention=attention.to("cuda")
        # x=attention(x)
        x = self.conv3_enc(x)
        x, indices3 = self.pool3_enc(x)
        # print(x.shape,indices3.shape)
        x = self.hidden(x)
        ###加入全连接层，然后完了再reshape为合适的形状
        # 动态计算全连接层输入和输出大小

        # print(x.shape)
        batch_size, c, h, w = x.shape
        # print(f"中间层的{x.shape}")
        x = x.view(batch_size, -1)  # 展平成一维向量
        x = self.fc1(x)  # 全连接层1
        x = F.relu(x)
        x = self.dropout1(x)  # 加入Dropout

        x = self.fc2(x)  # 全连接层2
        x = F.relu(x)
        x = self.dropout2(x)  # 加入Dropout

        x = self.fc3(x)  # 全连接层3
        x = F.relu(x)
        x = self.dropout3(x)  # 加入Dropout

        x = self.fc4(x)  # 全连接层4
        x = F.relu(x)

        x = self.fc5(x)  # 全连接层4
        x = F.relu(x)

        x = self.fc6(x)  # 全连接层4
        x = F.relu(x)

        x = x.view(batch_size, c, h, w)  # 重塑为卷积输入形状

        encoded_fea = x
        # print(x.shape)
        x = self.pool3_dec(x, indices3)
        x = self.conv3_dec(x)
        x = self.pool2_dec(x, indices2)
        x = self.conv2_dec(x)
        x = self.pool1_dec(x, indices1)
        x = self.conv1_dec(x)
        decoded_fea = x
        return encoded_fea, decoded_fea

    def _initialize_weights(self):
        # self.modules()返回的是整个网络模块从左到右，从最大模块到最小模块的全部信息
        # 例如1->(2->3,4->5,6->7)，则会包含1->(2->3,4->5,6->7)，2->3，3，4->5，5，6->7，7
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



