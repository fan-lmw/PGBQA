import torch
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt  # plotting library
import numpy as np  # this module is useful to work with numerical arrays
import random  # this module will be used to select random samples from a collection
import os  # this module will be used just to create directories in the local filesystem
from dataset import MyDataset
import time
import torch.nn as nn
from torch.nn import DataParallel
import argparse
from torchvision import models
from torch.nn import DataParallel

class GramMatrixLoss(nn.Module):
    def __init__(self):
        super(GramMatrixLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        # 计算x的Gram矩阵
        Gx = self.compute_gram_matrix(x)
        # 计算y的Gram矩阵
        Gy = self.compute_gram_matrix(y)
        # 计算两个Gram矩阵之间的差异
        loss = self.loss(Gx, Gy)
        return loss

    def compute_gram_matrix(self, x):
        # 计算Gram矩阵
        (b, ch, h, w) = x.size()
        # 扩展维度以便可以进行矩阵乘法
        x = x.view(b, ch, h * w)
        # 计算Gram矩阵
        G = torch.bmm(x, x.transpose(1, 2)) / (ch * h * w)
        return G


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        # 计算风格损失
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Training function
def train_epoch(convEncoder, device, dataloader, loss_fn,loss_gmm, optimizer):
    # Set train mode for both the encoder and the decoder
    convEncoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Decode data
        _, decoded_data = convEncoder(image_batch)

        # Evaluate loss
        loss1 = loss_fn(decoded_data, image_batch)
        loss2=loss_gmm(decoded_data, image_batch)
        loss=loss1+loss2

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

def main(args):
    # 超参数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    EPOCH = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr

    # 数据路径
    train_path = args.train_path
    result_pth = args.result_pth
    os.makedirs(result_pth, exist_ok=True)

    # 根据 patchsize 选择不同的模型
    if args.patchsize == 32:
        from models.model_new_32 import CAE  # 如果 patchsize 为 32，加载 model_new_32 模型
    elif args.patchsize == 64:
        from models.model_new_64 import CAE  # 如果 patchsize 为 64，加载 model_new_64 模型
    else:
        raise ValueError("Unsupported patchsize. Please use 32 or 64.")  # 不支持的 patchsize

    # 训练数据
    train_dataset = MyDataset(
        path=train_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
    )
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    convEncoder = CAE(dropout_prob=0.2)
    print(convEncoder)

    # 定义损失函数
    loss_fn = torch.nn.L1Loss()

    # 定义优化器
    optim = torch.optim.Adam(convEncoder.parameters(), lr=LR, weight_decay=1e-05)

    # 检查是否有GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # 如果有多个GPU，则使用DataParallel进行分布式训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        convEncoder = DataParallel(convEncoder)  # 使用 DataParallel 进行多卡训练

    # 将模型移至GPU/CPU
    convEncoder.to(device)

    loss_gmm = GramMatrixLoss()

    # 训练
    history = {'train_loss': []}
    txt_pth = os.path.join(result_pth, "train_log.txt")
    f = open(txt_pth, "w+")

    for epoch in range(EPOCH):
        train_loss = train_epoch(convEncoder, device, train_loader, loss_fn,loss_gmm, optim)
        history['train_loss'].append(train_loss)
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        out_log = t + "\t" + 'EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, EPOCH, train_loss)
        print(out_log)
        f.writelines(out_log + '\n')
        f.flush()

        if (epoch+1) % 50 == 0 :
            if torch.cuda.device_count() > 1:
                torch.save(convEncoder.module.state_dict(), os.path.join(result_pth, str(epoch+1).rjust(5, '0') + ".pth"))
            else :
                torch.save(convEncoder.state_dict(), os.path.join(result_pth, str(epoch+1).rjust(5, '0') + ".pth"))

    # 训练loss可视化
    plt.figure(figsize=(10, 8))
    plt.semilogy(history['train_loss'], label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(result_pth, "loss.jpg"))
    f.close()


if __name__ == '__main__':
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Train CAE model")
    parser.add_argument('--train_path', type=str, required=True, help="Path to the training data")
    parser.add_argument('--result_pth', type=str, required=True, help="Directory to save the model and logs")
    parser.add_argument('--epochs', type=int, default=600, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--device_id', type=str, default='0', help="CUDA device ID (e.g., '0' for GPU 0)")
    parser.add_argument('--patchsize', type=int, choices=[32, 64], required=True, help="Patch size (32 or 64)")
    args = parser.parse_args()

    # 调用训练主函数
    main(args)


